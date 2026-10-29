# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-derived contract shared by the verifier and standalone runtime."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex as re
from huggingface_hub import ResolvedRevision

from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import FeatureSlot, TargetFeatureSchema

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class Eagle3FeatureContract:
    aux_layer_ids: tuple[int, ...] | None
    input_width: int
    output_width: int

    def validate_model(self, model: Any) -> None:
        """Check the loaded native projection before allocating its KV cache."""
        inner = model.model
        use_aux = self.aux_layer_ids is not None
        if bool(getattr(inner, "use_aux_hidden_state", True)) != use_aux:
            raise ValueError(
                "loaded EAGLE3 hidden-state mode differs from its contract"
            )
        if inner.config.hidden_size != self.output_width:
            raise ValueError("loaded EAGLE3 hidden width differs from its contract")
        if use_aux:
            if (
                inner.fc.input_size != self.input_width
                or inner.fc.output_size != self.output_width
            ):
                raise ValueError(
                    "loaded EAGLE3 projection width differs from its contract"
                )
        elif hasattr(inner, "fc") or self.input_width != self.output_width:
            raise ValueError("loaded single-state EAGLE3 requires equal hidden widths")


@dataclass(frozen=True)
class RuntimeContract:
    schema: TargetFeatureSchema
    capabilities: SpeculatorPlacementCapabilities
    target_fingerprint: str
    tokenizer_fingerprint: str
    draft_fingerprint: str
    eagle3_features: Eagle3FeatureContract | None = None
    requires_target_lm_head: bool = True


@dataclass(frozen=True)
class _MTPModelContract:
    target_architectures: tuple[str, ...]
    max_speculative_tokens: int | None = None
    requires_target_lm_head: bool = True
    canonical_text_mrope: bool = False


# These native implementations consume one target hidden vector and own their
# attention KV. Other MTP architectures need a separate state/feature contract.
_MTP_MODELS = {
    "MiMoMTPModel": _MTPModelContract(("MiMoForCausalLM",), 1),
    "MiMoV2MTPModel": _MTPModelContract(("MiMoV2FlashForCausalLM",)),
    "ErnieMTPModel": _MTPModelContract(("Ernie4_5_MoeForCausalLM",), 1),
    "Qwen3NextMTP": _MTPModelContract(("Qwen3NextForCausalLM",)),
    "Qwen3_5MTP": _MTPModelContract(("Qwen3_5ForCausalLM",), canonical_text_mrope=True),
    "Qwen3_5MoeMTP": _MTPModelContract(
        ("Qwen3_5MoeForCausalLM",), canonical_text_mrope=True
    ),
    "Glm4MoeMTPModel": _MTPModelContract(("Glm4MoeForCausalLM",)),
    "Glm4MoeLiteMTPModel": _MTPModelContract(("Glm4MoeLiteForCausalLM",)),
    "DeepSeekMTPModel": _MTPModelContract(("DeepseekV3ForCausalLM",)),
    "ExaoneMoeMTP": _MTPModelContract(("ExaoneMoeForCausalLM", "ExaoneMoEForCausalLM")),
    "NemotronHMTPModel": _MTPModelContract(
        ("NemotronHForCausalLM", "NemotronHPuzzleForCausalLM")
    ),
    "HYV3MTPModel": _MTPModelContract(("HYV3ForCausalLM",)),
    "Step3p5MTP": _MTPModelContract(
        ("Step3p5ForCausalLM",), requires_target_lm_head=False
    ),
}


def _validate_mtp_contract(target: Any, spec: Any) -> _MTPModelContract:
    draft = spec.draft_model_config
    config = draft.hf_config
    architectures = getattr(config, "architectures", None) or []
    if len(architectures) != 1 or architectures[0] not in _MTP_MODELS:
        raise ValueError(f"unsupported standalone MTP architecture: {architectures}")
    architecture = architectures[0]
    contract = _MTP_MODELS[architecture]
    target_architectures = getattr(target.hf_config, "architectures", None) or []
    if (
        len(target_architectures) != 1
        or target_architectures[0] not in contract.target_architectures
    ):
        raise ValueError(f"{architecture} requires its matching target architecture")
    if (
        target.get_hidden_size() != draft.get_hidden_size()
        or target.get_vocab_size() != draft.get_vocab_size()
    ):
        raise ValueError("standalone MTP requires matching target hidden/vocab sizes")
    if target.max_model_len > draft.max_model_len:
        raise ValueError(
            "standalone MTP requires draft max_model_len "
            f"({draft.max_model_len}) to cover target max_model_len "
            f"({target.max_model_len}); reduce the target context limit"
        )
    num_modules = getattr(
        config,
        "mtp_num_hidden_layers"
        if contract.canonical_text_mrope
        else "num_nextn_predict_layers",
        None,
    )
    if type(num_modules) is not int or num_modules < 1:
        raise ValueError(
            "standalone MTP requires a positive checkpoint MTP module count"
        )
    if architecture == "ExaoneMoeMTP" and num_modules > 1:
        if spec.num_speculative_tokens != num_modules:
            raise ValueError(
                "Exaone checkpoint MTP modules require configured K "
                f"to equal the module count ({num_modules})"
            )
        layers = getattr(config, "mtp_layer_types", None)
        if (
            not isinstance(layers, (list, tuple))
            or len(layers) != num_modules
            or any(
                layer not in ("full_attention", "sliding_attention") for layer in layers
            )
        ):
            raise ValueError("Exaone MTP layers require one full/SWA type per module")
        module_windows = getattr(config, "mtp_sliding_windows", None)
        windows = module_windows or getattr(config, "sliding_windows", None)
        if module_windows and (
            not isinstance(module_windows, (list, tuple))
            or len(module_windows) != num_modules
        ):
            raise ValueError("Exaone MTP windows require one entry per module")
        if windows is not None and not isinstance(windows, (list, tuple)):
            raise ValueError("Exaone MTP windows require an array of per-layer values")
        for index, layer in enumerate(layers):
            if layer != "sliding_attention":
                continue
            if windows is not None and index >= len(windows):
                raise ValueError("Exaone MTP windows do not cover every sliding layer")
            window = (
                windows[index]
                if windows is not None
                else getattr(config, "sliding_window", None)
            )
            if type(window) is not int or window <= 0:
                raise ValueError("Exaone MTP sliding layers require positive windows")
    elif architecture == "Step3p5MTP":
        layers = getattr(config, "mtp_layer_types", None)
        mlp_layers = getattr(config, "mtp_mlp_layer_types", None)
        if (
            not isinstance(layers, (list, tuple))
            or len(layers) != num_modules
            or len(set(layers)) != 1
            or layers[0] not in ("full_attention", "sliding_attention")
            or mlp_layers != ["dense"] * num_modules
        ):
            raise ValueError(
                "standalone Step3.5 requires homogeneous full/SWA dense MTP layers"
            )
        if num_modules > 1 and (
            spec.num_speculative_tokens != 1 or layers[0] != "sliding_attention"
        ):
            raise ValueError(
                "standalone Step3.5 multiple checkpoint heads require SWA and K=1"
            )
        rope = getattr(config, "rope_parameters", None) or {}
        if not isinstance(rope.get(layers[0]), dict):
            raise ValueError(
                "standalone Step3.5 requires MTP attention RoPE parameters"
            )
        if layers[0] == "sliding_attention":
            window = getattr(config, "sliding_window", None)
            heads = getattr(config, "num_sliding_attention_heads", None)
            if (
                type(window) is not int
                or window < 1
                or type(heads) is not int
                or heads < 1
            ):
                raise ValueError(
                    "standalone Step3.5 SWA requires positive window/head geometry"
                )
        else:
            heads = getattr(config, "num_attention_heads", None)
            if getattr(config, "is_heterogeneous", False):
                per_layer = getattr(config, "per_layer_config", ())
                if len(per_layer) < config.num_hidden_layers + num_modules:
                    raise ValueError(
                        "standalone Step3.5 requires explicit MTP head geometry"
                    )
                heads = per_layer[config.num_hidden_layers].num_attention_heads
        kv_heads = getattr(config, "num_attention_groups", None)
        if (
            type(heads) is not int
            or type(kv_heads) is not int
            or kv_heads <= 1
            or heads < 1
            or heads % kv_heads
            or getattr(config, "att_impl_type", "GQA") != "GQA"
        ):
            raise ValueError("standalone Step3.5 requires divisible GQA head groups")
    elif num_modules != 1:
        raise ValueError(
            "standalone MTP requires one checkpoint MTP module, "
            "or an Exaone module chain"
        )
    if (
        contract.max_speculative_tokens is not None
        and spec.num_speculative_tokens > contract.max_speculative_tokens
    ):
        raise ValueError(f"{architecture} supports exactly one speculative token")
    for model in (target, draft):
        hf_config = model.hf_config
        rope = (
            getattr(hf_config, "rope_parameters", None)
            or getattr(hf_config, "rope_scaling", None)
            or {}
        )
        if (
            (
                getattr(model, "is_multimodal_model", False)
                and not (contract.canonical_text_mrope and model is draft)
            )
            or (
                getattr(model, "uses_mrope", False)
                and not contract.canonical_text_mrope
            )
            or getattr(hf_config, "text_config", None) is not None
            or ("mrope_section" in rope and not contract.canonical_text_mrope)
            or getattr(hf_config, "hc_mult", 1) != 1
        ):
            raise ValueError(
                "standalone MTP requires scalar positions and text features"
            )
    if contract.canonical_text_mrope:
        if spec.use_multi_module_mtp():
            raise ValueError("standalone Qwen3.5 requires one checkpoint MTP head")
        if target.uses_mrope:
            raise ValueError("standalone Qwen3.5 requires native flat-text positions")
        if getattr(target.hf_text_config, "mtp_num_hidden_layers", None) != num_modules:
            raise ValueError(
                "standalone Qwen3.5 requires matching checkpoint head counts"
            )
        for model in (target, draft):
            if model.uses_mrope and model.mrope_num_dims != 3:
                raise ValueError(
                    "standalone Qwen3.5 requires three canonical text axes"
                )
        # Native flat-text target initialization strips these inherited fields;
        # the native MTP proposer still feeds scalar positions to its head.
        ropes = [
            {
                key: value
                for key, value in model.hf_text_config.rope_parameters.items()
                if key not in ("mrope_section", "mrope_interleaved")
            }
            for model in (target, draft)
        ]
        if ropes[0] != ropes[1]:
            raise ValueError(
                "standalone Qwen3.5 requires matching text RoPE parameters"
            )
        # The native Qwen head is constructed from the target text config.
        # Draft metadata must describe the same geometry and expert layout.
        configs = []
        for model, rope in zip((target, draft), ropes):
            text = model.hf_text_config.to_dict()
            for field in (
                "architectures",
                "model_type",
                "n_predict",
                "_name_or_path",
                "transformers_version",
                "_commit_hash",
            ):
                text.pop(field, None)
            text["rope_parameters"] = rope
            configs.append(text)
        if configs[0] != configs[1]:
            raise ValueError("standalone Qwen3.5 requires matching native text configs")
    if architecture == "NemotronHMTPModel":
        pattern = getattr(config, "mtp_hybrid_override_pattern", "")
        if not pattern or "*" not in pattern or set(pattern) - {"*", "E"}:
            raise ValueError("standalone Nemotron MTP requires attention/MoE layers")
    if architecture == "Glm4MoeLiteMTPModel" and hasattr(config, "index_topk"):
        raise ValueError("standalone GLM MLA does not support a sparse indexer")
    if architecture == "MiMoV2MTPModel":
        for model in (target, draft):
            hf_config = model.hf_config
            if any(
                type(getattr(hf_config, field, None)) is not int
                or getattr(hf_config, field) != width
                for field, width in (
                    ("head_dim", 192),
                    ("swa_head_dim", 192),
                    ("v_head_dim", 128),
                    ("swa_v_head_dim", 128),
                )
            ):
                raise ValueError("standalone MiMo-V2 Flash requires QK192/V128 DiffKV")
            for prefix in ("", "swa_"):
                heads = getattr(hf_config, prefix + "num_attention_heads", None)
                kv_heads = getattr(hf_config, prefix + "num_key_value_heads", None)
                if (
                    type(heads) is not int
                    or type(kv_heads) is not int
                    or heads < 1
                    or kv_heads < 1
                    or heads % kv_heads
                ):
                    raise ValueError(
                        "standalone MiMo-V2 Flash requires divisible Q/KV heads"
                    )
            window = getattr(hf_config, "sliding_window_size", None)
            if type(window) is not int or window < 1:
                raise ValueError(
                    "standalone MiMo-V2 Flash requires a positive SWA window"
                )
        pattern = getattr(target.hf_config, "hybrid_layer_pattern", None)
        if (
            not isinstance(pattern, (list, tuple))
            or len(pattern) != target.get_total_num_hidden_layers()
            or any(type(layer) is not int or layer not in (0, 1) for layer in pattern)
        ):
            raise ValueError(
                "standalone MiMo-V2 Flash requires a full global/SWA layer pattern"
            )
    if architecture == "DeepSeekMTPModel":
        for model in (target, draft):
            if hasattr(model.hf_config, "index_topk"):
                raise ValueError(
                    "standalone DeepSeek MTP does not support a sparse indexer"
                )
    if architecture in (
        "DeepSeekMTPModel",
        "MiMoV2MTPModel",
        "Step3p5MTP",
        "Qwen3_5MTP",
        "Qwen3_5MoeMTP",
    ):
        for model in (target, draft):
            if (
                model.using_transformers_backend()
                or model.quantization is not None
                or getattr(model.hf_config, "quantization_config", None) is not None
                or str(model.dtype) != "torch.bfloat16"
            ):
                raise ValueError(
                    f"standalone {architecture} requires native unquantized bfloat16"
                )
    if architecture in ("DeepSeekMTPModel", "MiMoV2MTPModel"):
        # These native heads construct some or all layers from target metadata.
        # Compare only fields read there; the remaining draft fields stay local.
        defaults = (
            dict(
                num_hidden_layers=None,
                num_nextn_predict_layers=None,
                n_routed_experts=None,
                n_shared_experts=None,
                n_group=None,
            )
            if architecture == "DeepSeekMTPModel"
            else dict(
                layernorm_epsilon=None,
                intermediate_size=None,
                hidden_act=None,
                swa_num_attention_heads=None,
                swa_num_key_value_heads=None,
                swa_head_dim=None,
                swa_v_head_dim=None,
                attention_value_scale=None,
                sliding_window_size=-1,
                attention_bias=None,
                add_swa_attention_sink_bias=False,
                max_position_embeddings=32768,
                partial_rotary_factor=1.0,
            )
        )
        configs = []
        for hf_config in (target.hf_config, config):
            value = {
                field: getattr(hf_config, field, default)
                for field, default in defaults.items()
            }
            if architecture == "MiMoV2MTPModel":
                value["swa_rope_theta"] = getattr(
                    hf_config,
                    "swa_rope_theta",
                    getattr(hf_config, "rope_theta", 1000000),
                )
            configs.append(value)
        if configs[0] != configs[1]:
            raise ValueError(
                f"standalone {architecture} requires matching native configs "
                "for target-derived MTP layers"
            )
    return contract


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def checkpoint_revision(source: str, revision: str | None) -> str | None:
    """Identify loaded weights independently of their configuration repository."""
    if Path(source).exists():
        return None
    if isinstance(revision, ResolvedRevision):
        if revision._repo_id not in (None, source) or revision._repo_type != "model":
            raise ValueError(
                f"checkpoint revision was resolved for a different repository: {source}"
            )
        revision = revision.resolved
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "remote drafting requires an immutable checkpoint revision "
            f"or local checkpoint artifacts: {source}"
        )
    return revision


def _model_identity(model: Any) -> dict[str, Any]:
    weights_source = getattr(model, "model_weights", "")
    if weights_source and weights_source != model.model:
        raise ValueError(
            "remote drafting does not support a separate model_weights source; "
            "the advertised checkpoint must be the loaded weight source"
        )
    config = model.hf_config.to_dict()
    for field in ("_name_or_path", "transformers_version", "_commit_hash"):
        config.pop(field, None)
    path = Path(model.model)
    identity: dict[str, Any]
    if path.is_dir():
        filenames = sorted(
            str(file.relative_to(path))
            for pattern in (
                "*.safetensors",
                "*.bin",
                "*.pt",
                "*.gguf",
                "*.safetensors.index.json",
                "*.bin.index.json",
            )
            for file in path.rglob(pattern)
        )
        identity = {"artifacts": _local_artifact_digests(path, filenames)}
    elif path.exists():
        identity = {"artifact": _local_artifact_digests(path.parent, [path.name])}
    else:
        identity = {
            "repository": model.model,
            "revision": checkpoint_revision(model.model, model.revision),
        }
    return {
        "checkpoint": identity,
        "config": config,
        "dtype": str(model.dtype),
        "quantization": model.quantization,
    }


def _local_artifact_digests(path: Path, filenames: list[str]) -> dict[str, str]:
    """Hash deployed bytes at startup; mutable paths are never a cache key."""
    if not filenames:
        raise ValueError(f"no verifiable checkpoint/tokenizer artifacts in {path}")
    digests = {}
    for name in filenames:
        file = path / name
        before = file.stat()
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        after = file.stat()
        if any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise ValueError(
                f"checkpoint artifact changed while fingerprinting: {file}"
            )
        digests[name] = digest.hexdigest()
    return digests


def _tokenizer_identity(target: Any) -> dict[str, Any]:
    path = Path(target.tokenizer)
    if path.is_dir():
        filenames = [
            name
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "vocab.txt",
                "merges.txt",
                "tokenizer.model",
            )
            if (path / name).is_file()
        ]
        return {"artifacts": _local_artifact_digests(path, filenames)}
    return {
        "tokenizer": target.tokenizer,
        "revision": checkpoint_revision(target.tokenizer, target.tokenizer_revision),
    }


def _resolve_eagle3_features(vllm_config: "VllmConfig") -> Eagle3FeatureContract:
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    spec = vllm_config.speculative_config
    assert spec is not None and spec.draft_model_config is not None
    target = vllm_config.model_config
    draft = spec.draft_model_config
    config = draft.hf_config
    target_type = target.hf_config.model_type
    architectures = config.architectures or []
    native_target = {
        "llama": "LlamaForCausalLM",
        "qwen2": "Qwen2ForCausalLM",
        "mistral": "MistralForCausalLM",
    }.get(target_type)
    if native_target is not None:
        name = native_target.removesuffix("ForCausalLM")
        if target.hf_config.architectures != [native_target] or architectures not in (
            ["Eagle3LlamaForCausalLM"],
            ["LlamaForCausalLMEagle3"],
        ):
            raise ValueError(
                f"standalone {name} EAGLE3 requires native {name} "
                "and Llama EAGLE3 models"
            )
        if any(
            model.using_transformers_backend()
            or model.is_multimodal_model
            or model.is_encoder_decoder
            or model.uses_mrope
            or not getattr(model.hf_text_config, "is_causal", True)
            or model.quantization is not None
            for model in (target, draft)
        ):
            raise ValueError(
                f"standalone {name} EAGLE3 requires unquantized causal text models"
            )
        if target.get_vocab_size() != draft.get_vocab_size():
            raise ValueError(
                f"standalone {name} EAGLE3 requires equal target/draft vocabularies"
            )
        if target.get_hidden_size() != draft.get_hidden_size():
            raise ValueError(
                f"standalone {name} EAGLE3 requires equal target/draft hidden widths"
            )
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError(f"standalone {name} EAGLE3 requires target TP1")
        remote = spec.get_remote_draft_config()
        if remote is not None and (
            remote.transport not in ("auto", "inline")
            or len(spec.get_remote_draft_endpoints()) != 1
        ):
            raise ValueError(f"standalone {name} EAGLE3 requires one inline endpoint")
        if target_type in ("qwen2", "mistral"):
            for model in (target, draft):
                layers = getattr(model.hf_text_config, "layer_types", None)
                if (
                    model.get_sliding_window() is not None
                    or vllm_config.cache_config.sliding_window is not None
                    or model.attention_chunk_size is not None
                    or (
                        layers is not None
                        and (
                            not isinstance(layers, (list, tuple))
                            or len(layers) != model.get_total_num_hidden_layers()
                            or any(layer != "full_attention" for layer in layers)
                        )
                    )
                ):
                    raise ValueError(
                        f"standalone {name} EAGLE3 requires full attention "
                        "without a target, draft or global sliding window"
                    )
                if getattr(model.hf_text_config, "dual_chunk_attention_config", None):
                    raise ValueError(
                        f"standalone {name} EAGLE3 does not support "
                        "dual-chunk attention"
                    )
                if getattr(model.hf_text_config, "ada_rms_norm_t_cond", False):
                    raise ValueError(
                        f"standalone {name} EAGLE3 does not support conditional state"
                    )
    elif target_type not in ("gpt_oss", "qwen3"):
        raise ValueError(
            "standalone EAGLE3 supports GPT-OSS, Qwen3, Llama, Qwen2 "
            "and Mistral targets"
        )

    target_width, draft_width = target.get_hidden_size(), draft.get_hidden_size()
    eagle = getattr(config, "eagle_config", None) or {}
    if not eagle.get("use_aux_hidden_state", True):
        if target_width != draft_width:
            raise ValueError(
                "single-state EAGLE3 requires equal target/draft hidden widths"
            )
        return Eagle3FeatureContract(None, target_width, draft_width)

    layer_ids = get_eagle3_aux_layers_from_config(spec)
    num_layers = target.get_total_num_hidden_layers()
    if layer_ids is None:
        # The admitted native target classes inherit this SupportsEagle3 default.
        layer_ids = (2, num_layers // 2, num_layers - 3)
    if any(type(layer) is not int for layer in layer_ids):
        raise ValueError("EAGLE3 capture positions must be integers")
    if any(not 0 <= layer <= num_layers for layer in layer_ids):
        raise ValueError("EAGLE3 capture position is outside the target layer range")
    # EagleModelMixin emits captures in traversal order and removes duplicates.
    layer_ids = tuple(sorted(set(layer_ids)))

    # Match the native draft's projection, which has a separate count contract
    # from the target capture selector above.
    if architectures in (["Eagle3Qwen3ForCausalLM"], ["PeagleQwen3ForCausalLM"]):
        count = getattr(config, "num_aux_layers", None)
        if count is None:
            count = getattr(config, "num_aux_hidden_states", None)
        projected_ids = getattr(
            config, "eagle_aux_hidden_state_layer_ids", None
        ) or eagle.get("eagle_aux_hidden_state_layer_ids")
    else:
        count = getattr(config, "num_aux_hidden_states", None)
        projected_ids = eagle.get("eagle_aux_hidden_state_layer_ids")
    if count is None:
        count = len(projected_ids) if projected_ids else 3
    if type(count) is not int or count != len(layer_ids):
        raise ValueError(
            "EAGLE3 auxiliary projection count does not match selected "
            f"target layers: {count} != {len(layer_ids)}"
        )
    projected_width = getattr(config, "target_hidden_size", draft_width)
    if type(projected_width) is not int or projected_width != target_width:
        raise ValueError(
            "EAGLE3 projection target hidden width differs from the target"
        )
    return Eagle3FeatureContract(layer_ids, target_width * count, draft_width)


def create_runtime_contract(vllm_config: "VllmConfig") -> RuntimeContract:
    """Resolve the supported feature layout without loading any model weights."""
    spec = vllm_config.speculative_config
    if spec is None or spec.draft_model_config is None:
        raise ValueError("standalone drafting requires a resolved speculative config")
    target = vllm_config.model_config
    draft = spec.draft_model_config
    eagle3_features = None
    requires_target_lm_head = True
    if spec.method == "eagle3":
        eagle3_features = _resolve_eagle3_features(vllm_config)
    elif spec.method == "mtp":
        requires_target_lm_head = _validate_mtp_contract(
            target, spec
        ).requires_target_lm_head
        if draft.hf_config.architectures[0] in (
            "DeepSeekMTPModel",
            "MiMoV2MTPModel",
            "Step3p5MTP",
            "Qwen3_5MTP",
            "Qwen3_5MoeMTP",
        ):
            name = draft.hf_config.architectures[0]
            if (
                vllm_config.parallel_config.tensor_parallel_size != 1
                or spec.draft_parallel_config.tensor_parallel_size != 1
            ):
                raise ValueError(f"standalone {name} requires target TP1 and draft TP1")
            remote = spec.get_remote_draft_config()
            if remote is not None and (
                remote.transport not in ("auto", "inline")
                or len(spec.get_remote_draft_endpoints()) != 1
            ):
                raise ValueError(f"standalone {name} requires one inline endpoint")
            if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16") or (
                spec.kv_cache_dtype is not None
                and spec.kv_cache_dtype not in ("auto", "bfloat16")
            ):
                raise ValueError(f"standalone {name} requires bfloat16 KV")
    else:
        raise ValueError(f"unsupported standalone method: {spec.method}")
    if spec.parallel_drafting or spec.draft_sample_method != "greedy":
        raise ValueError("standalone runtime requires sequential greedy drafting")
    if spec.use_multi_module_mtp():
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("standalone multi-module MTP requires target TP1")
        remote = spec.get_remote_draft_config()
        if remote is not None and (
            remote.transport not in ("auto", "inline", "cuda_ipc")
            or len(spec.get_remote_draft_endpoints()) != 1
        ):
            raise ValueError("standalone multi-module MTP requires one endpoint")

    hidden_size = target.get_hidden_size()
    dtype = str(target.dtype).removeprefix("torch.")
    use_aux = eagle3_features is not None and eagle3_features.aux_layer_ids is not None
    slots = [
        FeatureSlot(TargetFeatureKind.TOKEN_IDS.value, "int32", ()),
        FeatureSlot(TargetFeatureKind.POSITIONS.value, "int64", ()),
    ]
    if use_aux:
        assert eagle3_features is not None
        slots.append(
            FeatureSlot(
                TargetFeatureKind.AUX_HIDDEN_STATES.value,
                dtype,
                (eagle3_features.input_width,),
            )
        )
    else:
        slots.append(
            FeatureSlot(
                TargetFeatureKind.LAST_HIDDEN_STATES.value, dtype, (hidden_size,)
            )
        )
    num_lookahead = vllm_config.num_prefill_lookahead_tokens
    slots.append(
        FeatureSlot(
            TargetFeatureKind.NEXT_PREFILL_TOKENS.value,
            "int64",
            (num_lookahead,),
            optional=True,
            per_sequence=True,
        )
    )
    schema = TargetFeatureSchema(schema_id=1, slots=tuple(slots))
    layout = [(s.kind, s.dtype, s.trailing_shape) for s in slots]
    feature_identity: dict[str, Any] = (
        {"eagle3_features": asdict(eagle3_features)}
        if eagle3_features is not None
        else {}
    )
    if not requires_target_lm_head:
        feature_identity["requires_target_lm_head"] = False
    target_identity = _model_identity(target)
    return RuntimeContract(
        schema=schema,
        eagle3_features=eagle3_features,
        requires_target_lm_head=requires_target_lm_head,
        capabilities=SpeculatorPlacementCapabilities(
            state_dependency="own_kv",
            required_features=tuple(s.kind for s in slots if not s.optional),
            num_prefill_lookahead=num_lookahead,
            supports_multi_module=spec.use_multi_module_mtp(),
            standalone_weights="materializable",
        ),
        target_fingerprint=_fingerprint(
            {"target": target_identity, "layout": layout, **feature_identity}
        ),
        tokenizer_fingerprint=_fingerprint(
            {
                "identity": _tokenizer_identity(target),
                "mode": target.tokenizer_mode,
                "vocab_size": target.get_vocab_size(),
                "special_tokens": {
                    k: target_identity["config"].get(k)
                    for k in ("bos_token_id", "eos_token_id", "pad_token_id")
                },
            }
        ),
        draft_fingerprint=_fingerprint(
            {
                "draft": _model_identity(draft),
                "layout": layout,
                "method": spec.method,
                "num_speculative_tokens": spec.num_speculative_tokens,
                **feature_identity,
            }
        ),
    )
