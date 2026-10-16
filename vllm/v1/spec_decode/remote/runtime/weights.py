# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize target embedding/head without loading its transformer layers."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open

from vllm.transformers_utils.repo_utils import hf_api

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def selected_checkpoint_weights(
    source: str, revision: str | None, names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Fetch only the safetensor shards containing the requested named tensors.

    Missing tensors are omitted so tied embeddings can be resolved explicitly.
    Network/authentication errors propagate; they never trigger a full download.
    """
    local = Path(source).is_dir()

    def resolve(filename: str) -> Path:
        if local:
            return Path(source) / filename
        return Path(hf_api().hf_hub_download(source, filename, revision=revision))

    try:
        index = resolve("model.safetensors.index.json")
        with index.open() as handle:
            weight_map = json.load(handle)["weight_map"]
    except (FileNotFoundError, EntryNotFoundError):
        single = resolve("model.safetensors")
        with safe_open(single, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            return {
                name: handle.get_tensor(name) for name in names if name in available
            }

    selected = {name: weight_map[name] for name in names if name in weight_map}
    result: dict[str, torch.Tensor] = {}
    for filename in sorted(set(selected.values())):
        with safe_open(resolve(filename), framework="pt", device="cpu") as handle:
            for name, shard in selected.items():
                if shard == filename:
                    result[name] = handle.get_tensor(name)
    return result


def materialize_target_shell(
    model_config: "ModelConfig",
    device: torch.device,
    *,
    shared_weights_source: str = "auto",
) -> torch.nn.Module:
    """Create only the modules the existing local draft loader shares."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    source = (
        model_config.model if shared_weights_source == "auto" else shared_weights_source
    )
    revision = model_config.revision if shared_weights_source == "auto" else None
    tensors = selected_checkpoint_weights(
        source, revision, ("model.embed_tokens.weight", "lm_head.weight")
    )
    embed = tensors.get("model.embed_tokens.weight")
    head = tensors.get("lm_head.weight")
    if embed is None:
        raise ValueError(f"{source} has no model.embed_tokens.weight")
    if head is None:
        if getattr(model_config.hf_config, "tie_word_embeddings", False):
            head = embed
        else:
            raise ValueError(f"{source} has no lm_head.weight and is not tied")
    shape = (model_config.get_vocab_size(), model_config.get_hidden_size())
    for name, tensor in (("embedding", embed), ("LM head", head)):
        if tuple(tensor.shape) != shape or not tensor.is_floating_point():
            raise ValueError(
                f"standalone {name} requires an unquantized {shape} weight; "
                f"got {tuple(tensor.shape)} {tensor.dtype}"
            )
    shell = torch.nn.Module()
    shell.model = torch.nn.Module()
    with device:
        shell.model.embed_tokens = VocabParallelEmbedding(
            *shape, params_dtype=model_config.dtype, quant_config=None
        )
        shell.lm_head = ParallelLMHead(
            *shape, params_dtype=model_config.dtype, quant_config=None
        )
    shell.model.embed_tokens.weight_loader(shell.model.embed_tokens.weight, embed)
    shell.lm_head.weight_loader(shell.lm_head.weight, head)
    return shell
