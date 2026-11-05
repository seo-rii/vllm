# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch

from vllm.utils.import_utils import resolve_obj_by_qualname

# Keep this light: vllm.config imports the speculator configs in this package.
if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.spec_decode.contracts import DraftBackend
    from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
    from vllm.v1.worker.gpu.states import RequestState


def init_speculator(
    vllm_config: "VllmConfig",
    device: torch.device,
    req_states: "RequestState",
):
    """Build the speculator named by `speculative_config.speculator_config`.

    The speculator is constructed with `(vllm_config, device)`, plus
    `req_states` if its constructor accepts it.
    """
    from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if getattr(speculative_config, "remote_draft", None) is not None or (
        getattr(getattr(speculative_config, "execution", None), "placement", None)
        == "remote"
    ):
        if speculative_config.method == "draft_model":
            from vllm.v1.worker.gpu.spec_decode.token_remote import (
                TokenOnlyRemoteSpeculator,
            )

            return TokenOnlyRemoteSpeculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.remote import RemoteSpeculator

        return RemoteSpeculator(vllm_config, device)
    speculator_config = speculative_config.speculator_config
    qualname = speculator_config.speculator_class
    if qualname is None:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
    speculator_cls = resolve_obj_by_qualname(qualname)
    if not (
        isinstance(speculator_cls, type) and issubclass(speculator_cls, BaseSpeculator)
    ):
        raise TypeError(
            f"speculator_class '{qualname}' of {type(speculator_config).__name__} "
            f"must be a subclass of {BaseSpeculator.__module__}."
            f"{BaseSpeculator.__qualname__}."
        )
    # BaseSpeculator doesn't define the constructor signature.
    factory = cast(Callable[..., "BaseSpeculator"], speculator_cls)
    if "req_states" in inspect.signature(factory).parameters:
        return factory(vllm_config, device, req_states=req_states)
    return factory(vllm_config, device)


def init_draft_backend(
    vllm_config: "VllmConfig", speculator: "BaseSpeculator"
) -> "DraftBackend | None":
    """Opt in without replacing the native object used by runner lifecycle hooks."""
    spec = vllm_config.speculative_config
    assert spec is not None
    if spec.use_remote_draft():
        return cast("DraftBackend", speculator)
    enabled = (
        spec.use_unified_draft_backend()
        if hasattr(spec, "use_unified_draft_backend")
        else getattr(spec, "execution", None) is not None
    )
    if not enabled:
        return None
    from vllm.v1.worker.gpu.spec_decode.local_backend import LocalDraftBackend

    return LocalDraftBackend(speculator)
