# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.v1.spec_decode.remote.runtime_fixtures import (
    cpu_parallel_state as cpu_parallel_state,
)
from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
    is_mtp_completeness_check_enabled,
)


def test_disable_mtp_completeness_check_is_scoped():
    assert is_mtp_completeness_check_enabled()

    with pytest.raises(RuntimeError), disable_mtp_completeness_check():
        assert not is_mtp_completeness_check_enabled()
        raise RuntimeError

    assert is_mtp_completeness_check_enabled()


@pytest.mark.parametrize("omitted", [None, "q_proj", "k_proj", "v_proj"])
@pytest.mark.parametrize("tracking", [None, True, False])
@pytest.mark.usefixtures("cpu_parallel_state")
def test_qwen35_moe_mtp_requires_every_attention_projection(
    tmp_path, omitted, tracking
):
    """Other QKV slices and complete experts cannot conceal a missing slice."""
    from tests.v1.e2e.spec_decode.remote.test_remote_mtp import _save_qwen35_mtp_model
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_model_parallel, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils.torch_utils import set_default_torch_dtype

    source, _ = _save_qwen35_mtp_model(tmp_path, routed=True)
    config = EngineArgs(
        model=source,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=128,
        enforce_eager=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        model_loader_extra_config={"enable_weights_track": tracking},
    ).create_engine_config()
    assert config.speculative_config is not None
    draft = config.speculative_config.draft_model_config
    path = tmp_path / "model" / "model.safetensors"
    reference = load_file(path)
    weights = reference.copy()
    if omitted is not None:
        missing = f"mtp.layers.0.self_attn.{omitted}.weight"
        del weights[missing]
        save_file(weights, path, metadata={"format": "pt"})
    loader = get_model_loader(config.load_config)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cpu"),
    ):
        destroy_model_parallel()
        initialize_model_parallel(backend="gloo")
        model = loader.create_model(config, draft)
        projection = model.model.layers[0].self_attn.qkv_proj
        with torch.no_grad():
            projection.weight.fill_(-2)
        if omitted is not None and tracking is not False:
            with pytest.raises(ValueError, match=missing):
                loader.load_weights(model, draft)
            return
        loader.load_weights(model, draft)
        expected = torch.cat(
            [
                torch.full_like(reference[f"mtp.layers.0.self_attn.{part}.weight"], -2)
                if part == omitted
                else reference[f"mtp.layers.0.self_attn.{part}.weight"]
                for part in ("q_proj", "k_proj", "v_proj")
            ]
        )
        assert torch.equal(projection.weight, expected)
