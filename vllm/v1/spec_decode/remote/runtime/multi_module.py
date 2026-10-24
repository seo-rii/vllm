# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authoritative target inputs for a native MTP module chain.

Later modules' trailing KV rows depend on proposed tokens. Replay the affected
real target rows together with the new accepted prefix, including after reduced
effective K. Native MTP receives compact inputs with zero rejections and
performs its usual token shift and all configured module forwards.
"""

from dataclasses import dataclass, replace

import torch

from vllm.v1.spec_decode.remote.capabilities import TargetFeatureKind
from vllm.v1.spec_decode.remote.protocol import (
    SequenceKey,
    SpeculatorStatusCode,
    TargetFeatureSchema,
)
from vllm.v1.spec_decode.remote.runtime.adapter import Features
from vllm.v1.spec_decode.remote.runtime.step import DraftStepPlan, DraftStepRow
from vllm.v1.spec_decode.remote.server import AdapterError


@dataclass(frozen=True)
class _History:
    position: int
    features: tuple[torch.Tensor | None, ...]
    speculative: bool


@dataclass(frozen=True)
class MultiModuleHistoryUpdate:
    key: SequenceKey
    history: _History
    previous: _History | None
    lifetime: object


class MultiModuleReplay:
    """At most depth-1 independently owned target rows per sequence generation.

    ``prepare_row`` only reads committed history. Stage updates before a
    forward, commit them after its completion, or discard the returned tuple
    on failure. No uncommitted tensor is retained by this object.
    """

    def __init__(self, schema: TargetFeatureSchema) -> None:
        self._schema = schema
        lookahead = next(
            slot
            for slot in schema.slots
            if slot.kind == TargetFeatureKind.NEXT_PREFILL_TOKENS.value
        )
        if (
            not lookahead.per_sequence
            or len(lookahead.trailing_shape) != 1
            or lookahead.trailing_shape[0] < 2
        ):
            raise ValueError("multi-module MTP requires depth-wide prefill lookahead")
        self._history_rows = lookahead.trailing_shape[0] - 1
        self._position_slot = next(
            i
            for i, slot in enumerate(schema.slots)
            if slot.kind == TargetFeatureKind.POSITIONS.value and not slot.per_sequence
        )
        self._history: dict[SequenceKey, _History] = {}
        self._lifetimes: dict[SequenceKey, object] = {}

    @property
    def num_sequences(self) -> int:
        return len(self._history)

    def prepare_row(
        self, row: DraftStepRow, features: Features
    ) -> tuple[DraftStepRow, Features]:
        if (
            row.num_tokens <= 0
            or row.num_replayed
            or row.num_sampled < 0
            or not 0 <= row.num_rejected < row.num_tokens
            or (row.num_sampled == 0 and row.num_rejected != 0)
        ):
            raise AdapterError(
                SpeculatorStatusCode.SEQUENCE_DESYNC, "invalid MTP execution row"
            )
        if len(features) != len(self._schema.slots):
            raise ValueError("MTP features do not match the negotiated schema")
        for slot, tensor in zip(self._schema.slots, features):
            if tensor is None:
                if not slot.optional:
                    raise ValueError(f"missing MTP feature {slot.kind}")
            elif not slot.per_sequence and tensor.shape[0] != row.num_tokens:
                raise ValueError(f"MTP feature {slot.kind} has the wrong token count")
        positions = features[self._position_slot]
        assert positions is not None
        if positions.ndim != 1:
            raise ValueError("MTP positions must be scalar")
        start = int(positions[0].item())
        history = self._history.get(row.key)
        expected_start = 0 if history is None else history.position + 1
        if start != expected_start or not torch.equal(
            positions,
            torch.arange(
                start,
                start + row.num_tokens,
                dtype=positions.dtype,
                device=positions.device,
            ),
        ):
            raise AdapterError(
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                "MTP target positions do not continue the completed history",
            )
        valid = row.num_tokens - row.num_rejected
        if history is not None and history.speculative and valid != row.num_sampled:
            raise AdapterError(
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                "MTP acceptance does not match the target prefix",
            )
        replay = (
            min(start, max(0, self._history_rows + 1 - row.num_sampled))
            if history is not None and history.speculative
            else 0
        )
        prepared: Features = []
        for i, (slot, tensor) in enumerate(zip(self._schema.slots, features)):
            if slot.per_sequence or tensor is None:
                prepared.append(tensor)
                continue
            prefix = tensor[:valid]
            if replay:
                assert history is not None
                previous = history.features[i]
                if previous is None or previous.shape[0] < replay:
                    raise AdapterError(
                        SpeculatorStatusCode.SEQUENCE_DESYNC,
                        "MTP replay requires the previous target features",
                    )
                prefix = torch.cat((previous[-replay:].to(tensor.device), prefix))
            prepared.append(prefix)
        return (
            replace(
                row, num_tokens=valid + replay, num_rejected=0, num_replayed=replay
            ),
            prepared,
        )

    def stage_history(
        self, plan: DraftStepPlan, features: Features
    ) -> tuple[MultiModuleHistoryUpdate, ...]:
        """Own the completed target suffix before native scratch is used."""
        histories = []
        for i, row in enumerate(plan.rows):
            end = int(plan.query_start_loc[i + 1]) - row.num_rejected
            last = end - 1
            start = int(plan.query_start_loc[i]) + row.num_replayed
            if last < start:
                raise ValueError("MTP history needs a new target row")
            position = int(plan.positions[last])
            keep = min(position + 1, self._history_rows)
            previous = self._history.get(row.key)
            owned: list[torch.Tensor | None] = []
            for slot_index, (slot, tensor) in enumerate(
                zip(self._schema.slots, features)
            ):
                if tensor is None or slot.per_sequence:
                    owned.append(None)
                    continue
                suffix = tensor[max(start, end - keep) : end].detach()
                needed = keep - suffix.shape[0]
                if needed:
                    old = None if previous is None else previous.features[slot_index]
                    if old is None or old.shape[0] < needed:
                        raise AdapterError(
                            SpeculatorStatusCode.SEQUENCE_DESYNC,
                            "MTP history requires the previous target features",
                        )
                    suffix = torch.cat((old[-needed:].to(tensor.device), suffix))
                else:
                    suffix = suffix.clone()
                owned.append(suffix)
            history = _History(
                position=position,
                features=tuple(owned),
                speculative=row.num_sampled > 0,
            )
            histories.append((row.key, history))
        return tuple(
            MultiModuleHistoryUpdate(
                key,
                history,
                self._history.get(key),
                self._lifetimes.setdefault(key, object()),
            )
            for key, history in histories
        )

    def commit_history(self, updates: tuple[MultiModuleHistoryUpdate, ...]) -> None:
        for update in updates:
            if (
                self._lifetimes.get(update.key) is not update.lifetime
                or self._history.get(update.key) is not update.previous
            ):
                raise ValueError("stale MTP history update")
        for update in updates:
            self._history[update.key] = update.history

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        for key in keys:
            self._history.pop(key, None)
            self._lifetimes.pop(key, None)
