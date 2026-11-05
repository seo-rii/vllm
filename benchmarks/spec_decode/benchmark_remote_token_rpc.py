# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure native token-draft RPC cost with explicitly synthetic target outcomes.

The manifest contains schema_version=1, hello (the server's TokenHello contract),
models (string model/revision identities) and server_source (string source IDs).
This client runs no target model and cannot measure target acceptance or serving
throughput. Run identical arguments over loopback and a trusted cross-host path.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgspec

from vllm.v1.spec_decode.remote.runtime.token_state import TokenAdvanceRow
from vllm.v1.spec_decode.remote.token_protocol import TokenHello
from vllm.v1.spec_decode.remote.token_session import TokenOnlyRemoteSession


class Manifest(msgspec.Struct):
    hello: TokenHello
    models: dict[str, str]
    server_source: dict[str, str]
    schema_version: int = 1


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--endpoint", required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output-json", type=Path, required=True)
    result.add_argument("--allow-cross-node", action="store_true")
    result.add_argument(
        "--transport-label", choices=("loopback", "tcp", "ssh-tunnel"), required=True
    )
    result.add_argument("--requests", type=_positive_int, default=8)
    result.add_argument("--rounds-per-request", type=_positive_int, default=12)
    result.add_argument("--warmup-requests", type=int, default=1)
    result.add_argument("--k", type=_positive_int)
    result.add_argument("--prompt-length", type=_positive_int, default=16)
    result.add_argument("--prompt-token", type=int, default=3)
    result.add_argument("--bonus-token", type=int, default=4)
    result.add_argument(
        "--accepted-prefix", default="all", help="Synthetic accepted count: all or 0..K"
    )
    result.add_argument("--request-timeout-ms", type=_positive_int, default=5000)
    result.add_argument("--startup-timeout-ms", type=_positive_int, default=30000)
    result.add_argument(
        "--target-token-ms", type=_positive_float, nargs="*", default=[]
    )
    return result


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return dict(
        mean=statistics.fmean(ordered),
        min=ordered[0],
        p50=percentile(0.5),
        p95=percentile(0.95),
        p99=percentile(0.99),
        max=ordered[-1],
    )


def _run_request(
    session: TokenOnlyRemoteSession,
    sequence_id: int,
    batch_ids,
    args: argparse.Namespace,
    k: int,
    accepted_prefix: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = session.deadline()
    started = time.perf_counter_ns()
    key = session.open_sequence(sequence_id, deadline=deadline)
    opened = time.perf_counter_ns()
    prompt = (args.prompt_token,) * args.prompt_length
    records = []
    try:
        chunk_size = session.hello.max_num_batched_tokens
        for offset in range(0, len(prompt), chunk_size):
            chunk = prompt[offset : offset + chunk_size]
            session.prefill(
                key,
                chunk,
                is_final=offset + len(chunk) == len(prompt),
                deadline=deadline,
            )
        prefilled = time.perf_counter_ns()
        for round_id in range(args.rounds_per_request):
            state = session.records[key]
            offered = len(state.proposal_tokens)
            accepted = min(accepted_prefix, offered)
            delta = state.proposal_tokens[:accepted] + (args.bonus_token,)
            batch_id = next(batch_ids)
            row = TokenAdvanceRow(
                key, state.committed, state.proposal_id, offered, accepted, delta
            )
            before = time.perf_counter_ns()
            output = session.collect(
                session.dispatch(batch_id, (row,), k, deadline=session.deadline())
            )
            elapsed_ms = (time.perf_counter_ns() - before) / 1e6
            valid = output.valid_lengths[0]
            if not session.active(key) or valid != k:
                raise RuntimeError("RPC did not produce the requested active draft")
            records.append(
                dict(
                    request=sequence_id,
                    round=round_id,
                    rpc_ms=elapsed_ms,
                    previous_offered_tokens=offered,
                    synthetic_accepted_tokens=accepted,
                    synthetic_emitted_tokens=accepted + 1,
                    valid_proposal_tokens=valid,
                    proposal_token_ids=list(output.token_ids[0][:valid]),
                    input_token_payload_bytes=len(delta) * 8,
                    input_state_payload_bytes=5 * 8,
                    output_token_payload_bytes=len(output.token_ids[0]) * 8,
                    output_state_payload_bytes=2 * 4 + 2 * 8,
                )
            )
    finally:
        before_close = time.perf_counter_ns()
        session.close_sequences((key,), deadline=session.deadline())
        close_ms = (time.perf_counter_ns() - before_close) / 1e6
    return (
        dict(
            request=sequence_id,
            open_ms=(opened - started) / 1e6,
            prefill_ms=(prefilled - opened) / 1e6,
            prefill_token_payload_bytes=len(prompt) * 4,
            close_ms=close_ms,
        ),
        records,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    with args.manifest.open("rb") as manifest_file:
        raw_manifest = manifest_file.read((64 << 10) + 1)
    if len(raw_manifest) > 64 << 10:
        raise ValueError("manifest exceeds the 64 KiB limit")
    manifest = msgspec.json.decode(raw_manifest, type=Manifest)
    if (
        manifest.schema_version != 1
        or not manifest.models
        or not manifest.server_source
    ):
        raise ValueError("manifest requires schema1, model IDs and server source IDs")
    hello = manifest.hello
    if hello.method != "draft_model" or hello.transport != "inline":
        raise ValueError("benchmark requires an inline token-only contract")
    if not all(
        (hello.target_fingerprint, hello.draft_fingerprint, hello.tokenizer_fingerprint)
    ):
        raise ValueError("manifest model/tokenizer fingerprints must be populated")
    if (
        args.warmup_requests < 0
        or args.requests + args.warmup_requests > 1024
        or args.rounds_per_request * (args.requests + args.warmup_requests) > 65536
    ):
        raise ValueError("benchmark exceeds the 1024-request/65536-round bounds")
    k = hello.num_speculative_tokens if args.k is None else args.k
    if (
        args.prompt_length > 65536
        or hello.num_speculative_tokens > 64
        or hello.max_num_seqs > 4096
        or hello.max_num_batched_tokens > 1 << 20
    ):
        raise ValueError("benchmark prompt and frame capacities exceed safe bounds")
    accepted = k if args.accepted_prefix == "all" else int(args.accepted_prefix)
    if not 0 <= accepted <= k <= hello.num_speculative_tokens:
        raise ValueError("synthetic accepted prefix and K must fit the contract")
    if any(
        not 0 <= token < hello.vocab_size
        for token in (args.prompt_token, args.bonus_token)
    ):
        raise ValueError("prompt/bonus token IDs must fit the negotiated vocabulary")
    final_end = (
        args.prompt_length + 1 + (args.rounds_per_request - 1) * (accepted + 1) + k
    )
    if final_end > hello.max_model_len:
        raise ValueError("prompt, synthetic commits and rollout exceed model context")
    hello = msgspec.structs.replace(
        hello,
        verifier_instance_id=f"rpc-bench-{uuid.uuid4().hex}",
        cross_node=False,
        byte_order="",
    )
    requests, rounds = [], []
    batch_ids = itertools.count()
    started = time.perf_counter_ns()
    session = TokenOnlyRemoteSession(
        args.endpoint,
        hello,
        failure_policy="error",
        request_timeout_ms=args.request_timeout_ms,
        startup_timeout_ms=args.startup_timeout_ms,
        allow_cross_node=args.allow_cross_node,
    )
    try:
        session.connect()
        connection_ms = (time.perf_counter_ns() - started) / 1e6
        for sequence_id in range(args.warmup_requests + args.requests):
            request, records = _run_request(
                session, sequence_id, batch_ids, args, k, accepted
            )
            if sequence_id >= args.warmup_requests:
                requests.append(request)
                rounds.extend(records)
        if session.records:
            raise RuntimeError("benchmark retained sequence credit after close")
    finally:
        session.close()
    total_rpc_ms = sum(row["rpc_ms"] for row in rounds)
    emitted = sum(row["synthetic_emitted_tokens"] for row in rounds)
    mean_emitted = emitted / len(rounds)
    mean_rpc_ms = total_rpc_ms / len(rounds)
    versions = {}
    for package in ("vllm", "torch", "transformers", "huggingface_hub", "msgspec"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    runtime_file = Path(inspect.getfile(TokenOnlyRemoteSession)).resolve()
    source = runtime_file.parents[4]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=source, capture_output=True
    )
    source_head = head.stdout.strip() if head.returncode == 0 else None
    if source_head is None and (source / ".source-head").is_file():
        source_head = (source / ".source-head").read_text().strip()
    return dict(
        schema_version=1,
        measured_at=datetime.now(UTC).isoformat(),
        experiment="native token-draft RPC; synthetic target outcomes",
        transport=dict(
            label=args.transport_label,
            endpoint=args.endpoint,
            cross_node=args.allow_cross_node,
        ),
        models=manifest.models,
        server_source=manifest.server_source,
        client_source=dict(
            head_sha=source_head,
            tracked_diff_sha256=hashlib.sha256(diff.stdout).hexdigest()
            if diff.returncode == 0
            else None,
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            runtime_session_sha256=hashlib.sha256(
                runtime_file.read_bytes()
            ).hexdigest(),
        ),
        environment=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            libraries=versions,
        ),
        contract=msgspec.to_builtins(session.hello),
        manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        settings=dict(
            requests=args.requests,
            rounds_per_request=args.rounds_per_request,
            warmup_requests=args.warmup_requests,
            prompt_length=args.prompt_length,
            prompt_token=args.prompt_token,
            bonus_token=args.bonus_token,
            k=k,
            synthetic_accepted_prefix=accepted,
            request_timeout_ms=args.request_timeout_ms,
            startup_timeout_ms=args.startup_timeout_ms,
        ),
        connection_ms=connection_ms,
        latency_ms=dict(
            round_rpc=_distribution([row["rpc_ms"] for row in rounds]),
            open=_distribution([row["open_ms"] for row in requests]),
            prefill=_distribution([row["prefill_ms"] for row in requests]),
            close=_distribution([row["close_ms"] for row in requests]),
        ),
        totals=dict(
            valid_proposal_tokens=sum(row["valid_proposal_tokens"] for row in rounds),
            synthetic_accepted_tokens=sum(
                row["synthetic_accepted_tokens"] for row in rounds
            ),
            synthetic_emitted_tokens=emitted,
            target_verified_tokens=None,
            prefill_token_payload_bytes=sum(
                row["prefill_token_payload_bytes"] for row in requests
            ),
            input_token_payload_bytes=sum(
                row["input_token_payload_bytes"] for row in rounds
            ),
            input_state_payload_bytes=sum(
                row["input_state_payload_bytes"] for row in rounds
            ),
            output_token_payload_bytes=sum(
                row["output_token_payload_bytes"] for row in rounds
            ),
            output_state_payload_bytes=sum(
                row["output_state_payload_bytes"] for row in rounds
            ),
        ),
        break_even=dict(
            zero_verification_lower_bound_target_token_ms=total_rpc_ms / emitted,
            assumed_target_token_times=[
                dict(
                    target_only_token_ms=value,
                    remaining_verification_ms_per_round=mean_emitted * value
                    - mean_rpc_ms,
                )
                for value in args.target_token_ms
            ],
            assumptions=(
                "Serial draft RPC plus target verification; synthetic acceptance. "
                "A positive budget is necessary, not sufficient, for a target speedup."
            ),
        ),
        limitations=[
            "No target executes; acceptance and emitted tokens are synthetic.",
            "RPC timing includes native draft, serialization, network and queue delay; "
            "it is not isolated network latency.",
            "Tensor payload bytes exclude control metadata, framing, TCP and SSH.",
            "Break-even excludes connection/open/prefill/close setup. It does not "
            "establish serving throughput, overlap or GPU cost efficiency.",
        ],
        request_records=requests,
        round_records=rounds,
    )


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=args.output_json.parent,
            prefix=args.output_json.name + ".",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(result, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(args.output_json)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            dict(
                output=str(args.output_json),
                round_rpc_ms=result["latency_ms"]["round_rpc"],
                target_verified_tokens=None,
            )
        )
    )


if __name__ == "__main__":
    main()
