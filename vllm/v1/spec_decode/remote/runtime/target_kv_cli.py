# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the experimental same-host Gemma4 E2B snapshot assistant."""

import argparse

import torch

from vllm.config import AttentionConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.spec_decode.remote.runtime.target_kv_gpu import Gemma4SnapshotExecutor
from vllm.v1.spec_decode.remote.target_kv_server import TargetKVDraftServer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-revision")
    parser.add_argument("--draft-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-feature-tokens", type=int, default=64)
    parser.add_argument("--language-model-only", action="store_true", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    config = EngineArgs(
        model=args.target_model,
        revision=args.target_revision,
        tokenizer_revision=args.target_revision,
        trust_remote_code=args.trust_remote_code,
        language_model_only=args.language_model_only,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_feature_tokens,
        block_size=16,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN),
        speculative_config={
            "model": args.draft_model,
            "revision": args.draft_revision,
            "method": "mtp",
            "num_speculative_tokens": 1,
            "draft_tensor_parallel_size": 1,
        },
    ).create_engine_config()
    executor = Gemma4SnapshotExecutor(config, device=torch.device(args.device))
    contract = executor.contract
    server = TargetKVDraftServer(
        executor,
        contract_fingerprint=contract.fingerprint,
        hidden_size=contract.hidden_size,
        vocab_size=contract.vocab_size,
        expected_specs=contract.expected_specs,
        snapshot_budget=contract.snapshot_budget,
        receiver_budget=contract.receiver_budget,
    )
    try:
        endpoint = server.start(args.endpoint)
        print(f"Gemma4 target-KV draft ready at {endpoint}", flush=True)
        server.wait()
        if not server.healthy:
            raise RuntimeError("Gemma4 draft runtime is unhealthy; restart required")
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        executor.close()


if __name__ == "__main__":
    main()
