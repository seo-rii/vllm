# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch the v3 independent draft runtime on one explicitly selected GPU."""

import argparse


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--endpoint", required=True)
    result.add_argument("--target-model", required=True)
    result.add_argument("--target-revision")
    result.add_argument("--draft-model", required=True)
    result.add_argument("--draft-revision")
    result.add_argument("--num-speculative-tokens", type=int, default=3)
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--policy", choices=("on_demand", "outcome_cache"), default="on_demand"
    )
    result.add_argument("--cache-max-branches", type=int, default=1)
    result.add_argument("--cache-kv-budget-mib", type=int, default=64)
    result.add_argument("--cache-copy-chunk-blocks", type=int, default=1)
    result.add_argument("--draft-kv-cache-gib", type=float, default=0.25)
    result.add_argument("--max-model-len", type=int, default=256)
    result.add_argument("--max-batch-size", type=int, default=4)
    result.add_argument("--max-feature-tokens", type=int, default=256)
    result.add_argument("--dtype", default="bfloat16")
    result.add_argument("--block-size", type=int, default=16)
    result.add_argument("--attention-backend")
    result.add_argument("--target-attention-backend")
    result.add_argument("--trust-remote-code", action="store_true")
    return result


def create_adapter(args):
    import torch

    from vllm.config import AttentionConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.spec_decode.remote.runtime.token_cache import TokenCacheOptions
    from vllm.v1.spec_decode.remote.runtime.token_gpu import make_token_gpu_adapter

    cache = None
    if args.policy == "outcome_cache":
        cache = TokenCacheOptions(
            max_branches=args.cache_max_branches,
            kv_budget_bytes=args.cache_kv_budget_mib << 20,
            copy_chunk_blocks=args.cache_copy_chunk_blocks,
        )
    elif (
        args.cache_max_branches,
        args.cache_kv_budget_mib,
        args.cache_copy_chunk_blocks,
    ) != (1, 64, 1):
        raise ValueError("cache budgets require --policy outcome_cache")

    config = EngineArgs(
        model=args.target_model,
        revision=args.target_revision,
        tokenizer_revision=args.target_revision,
        tokenizer_mode="hf",
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_batch_size,
        max_num_batched_tokens=args.max_feature_tokens,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum[args.target_attention_backend]
            if args.target_attention_backend
            else None
        ),
        block_size=args.block_size,
        speculative_config={
            "model": args.draft_model,
            "revision": args.draft_revision,
            "method": "draft_model",
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
            "attention_backend": args.attention_backend,
        },
    ).create_engine_config()
    return make_token_gpu_adapter(
        config,
        device=torch.device(args.device),
        kv_cache_bytes=int(args.draft_kv_cache_gib * 1024**3),
        outcome_cache=cache,
    )


def main() -> None:
    args = parser().parse_args()
    from vllm.v1.spec_decode.remote.protocol import RemoteServerLimits
    from vllm.v1.spec_decode.remote.token_server import TokenDraftServer

    adapter = create_adapter(args)
    server = TokenDraftServer(
        adapter,
        limits=RemoteServerLimits(
            args.max_batch_size,
            args.max_feature_tokens,
            args.max_batch_size,
            args.max_model_len,
        ),
    )
    try:
        endpoint = server.start(args.endpoint)
        print(f"TOKEN_DRAFT_READY {endpoint}", flush=True)
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    if not server.healthy:
        raise RuntimeError("token draft runtime requires a process restart")


if __name__ == "__main__":
    main()
