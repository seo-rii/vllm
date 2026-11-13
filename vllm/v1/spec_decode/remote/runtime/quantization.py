# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native weight formats admitted by standalone execution contracts."""

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


@dataclass(frozen=True)
class NativeWeightProfile:
    format: Literal["unquantized", "fp8_per_tensor_dynamic"]
    compute_dtype: torch.dtype


def resolve_native_weight_profile(
    model_config: "ModelConfig",
    *,
    allow_fp8_per_tensor: bool = False,
    context: str = "standalone",
) -> NativeWeightProfile:
    quantization = getattr(model_config.hf_config, "quantization_config", None)
    if model_config.quantization is None and quantization is None:
        return NativeWeightProfile("unquantized", model_config.dtype)
    if not allow_fp8_per_tensor:
        raise ValueError(f"{context} requires unquantized weights")
    detail = f"{context} requires serialized per-tensor FP8 with dynamic activations"
    if (
        model_config.quantization != "fp8"
        or not isinstance(quantization, dict)
        or quantization.get("quant_method") != "fp8"
        or quantization.get("activation_scheme") != "dynamic"
    ):
        raise ValueError(detail)
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    native = Fp8Config.from_config(quantization)
    if (
        not native.is_checkpoint_fp8_serialized
        or native.weight_block_size is not None
        or native.store_dtype is not None
    ):
        raise ValueError(detail)
    return NativeWeightProfile("fp8_per_tensor_dynamic", model_config.dtype)


def require_complete_native_weights(
    config: "VllmConfig", profile: NativeWeightProfile
) -> "VllmConfig":
    """Enable native parameter tracking on the effective FP8 draft loader."""
    if profile.format == "unquantized":
        return config
    from vllm.config import replace
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.utils import get_draft_load_config

    spec = config.speculative_config
    assert spec is not None
    load = get_draft_load_config(config)
    extra = load.model_loader_extra_config
    if extra.get("enable_weights_track") is False:
        raise ValueError("standalone FP8 requires enable_weights_track=True")
    strict = replace(
        load, model_loader_extra_config={**extra, "enable_weights_track": True}
    )
    if not isinstance(get_model_loader(strict), DefaultModelLoader):
        raise ValueError("standalone FP8 requires the native default weight loader")
    # SpeculativeConfig already contains normalized fields that are not valid
    # constructor inputs. Preserve them and the shared forward-context objects.
    resource = copy(config)
    resource.speculative_config = copy(spec)
    resource.speculative_config.draft_load_config = strict
    return resource


def validate_shared_weight(
    name: str, tensor: torch.Tensor, expected_shape: tuple[int, int]
) -> None:
    if tuple(tensor.shape) != expected_shape or tensor.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError(
            f"standalone {name} requires an unquantized {expected_shape} weight; "
            f"got {tuple(tensor.shape)} {tensor.dtype}"
        )


def validate_loaded_weight_profile(
    model: torch.nn.Module, profile: NativeWeightProfile
) -> dict[str, object]:
    """Inspect native loaded storage without reading or synchronizing tensors."""
    from vllm import envs
    from vllm.model_executor.kernels.linear.scaled_mm import FP8ScaledMMLinearKernel
    from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
        MarlinFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.platforms import current_platform

    quantized = 0
    kernels: set[str] = set()
    storage: set[str] = set()
    for name, layer in model.named_modules():
        if profile.format == "fp8_per_tensor_dynamic" and isinstance(
            layer, VocabParallelEmbedding
        ):
            validate_shared_weight(
                name,
                layer.weight,
                (layer.num_embeddings_per_partition, layer.embedding_dim),
            )
            if layer.weight.dtype != profile.compute_dtype:
                raise ValueError(f"standalone shared {name} violates its compute dtype")
        method = getattr(layer, "quant_method", None)
        if not isinstance(method, Fp8LinearMethod):
            continue
        native = method.quant_config
        if (
            profile.format != "fp8_per_tensor_dynamic"
            or not native.is_checkpoint_fp8_serialized
            or native.activation_scheme != "dynamic"
            or native.weight_block_size is not None
            or native.store_dtype is not None
            or method.input_dtype != profile.compute_dtype
        ):
            raise ValueError(f"loaded FP8 layer {name} violates its weight profile")
        kernel = method.fp8_linear
        expected_dtype = (
            torch.int32 if method.use_marlin else current_platform.fp8_dtype()
        )
        scale = layer.weight_scale
        if (
            not isinstance(kernel, FP8ScaledMMLinearKernel)
            or method.use_marlin != isinstance(kernel, MarlinFP8ScaledMMLinearKernel)
            or layer.weight.dtype != expected_dtype
            or scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or not scale.numel()
            or scale.device != layer.weight.device
        ):
            raise ValueError(f"loaded FP8 layer {name} has incompatible native storage")
        # Packed Marlin storage cannot be used by the native BF16 fallback.
        if method.use_marlin and envs.VLLM_BATCH_INVARIANT:
            raise ValueError("standalone Marlin FP8 requires VLLM_BATCH_INVARIANT=0")
        quantized += 1
        kernels.add(type(kernel).__name__)
        storage.add(str(layer.weight.dtype))
    if profile.format == "fp8_per_tensor_dynamic" and quantized == 0:
        raise ValueError("standalone FP8 profile loaded no FP8 projection kernels")
    return dict(
        format=profile.format,
        compute_dtype=str(profile.compute_dtype),
        quantized_linear_modules=quantized,
        native_kernels=sorted(kernels),
        storage_dtypes=sorted(storage),
    )
