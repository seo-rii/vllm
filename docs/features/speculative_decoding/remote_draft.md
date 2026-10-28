# Remote draft placement (experimental)

Remote placement separates a target-conditioned speculator from the target
process. The target remains responsible for verification and sampling. Placement
is configured independently of the speculative method; native local execution
remains the default.

The standalone GPU runtime provides same-host target-conditioned execution with
independent draft KV. It supports greedy, eager MRV2 execution, target TP1 or TP2
and draft TP1. The protocol test adapter is for development only.

For existing full-attention token-only target/draft combinations, target prefix
caching can replay cached tokens into a fresh draft sequence; see
[cached target prefixes](../../design/unified_speculation_runtime.md#cached-target-prefixes).
If preparing a cached prefix runs out of host memory, `target_only` disables
remote drafting for that request until it finishes. Other requests keep their
remote sequences; `error` propagates the allocation failure.
The same single-endpoint TP1 combination has an experimental eager MRV2
[async verifier scheduling mode](../../design/unified_speculation_runtime.md#async-verifier-scheduling).
Other remote backends retain their synchronous scheduling requirement.
For dense full-attention token-only models, an explicit draft TP2 mode uses two
native distributed draft ranks with target TP1, BF16 weights/KV, eager
synchronous execution and one inline endpoint. Its prefix caching and async
scheduling remain disabled. See the
[token-only topology and launch contract](../../design/unified_speculation_runtime.md#token-only-state-contract).

## Configuration and lifecycle

`speculative_config.remote_draft` takes a single `endpoint`, a `transport`
(`auto`, `inline` or `cuda_ipc`) and a `failure_policy` (`target_only` or `error`). `auto`
selects inline. Use a Unix socket or loopback TCP endpoint on the same host.
The protocol does not authenticate peers; non-loopback server binding requires
explicit `--allow-remote-connections`.

The default request budget is 100 ms and the initial connection budget is
30,000 ms. A request timeout under `target_only` permanently disables remote
speculation for affected requests, returning zero valid drafts while the target
continues. `error` surfaces the failure. Reconnection may admit new identities;
it does not rebuild the old requests' draft KV. Generation and session fencing
prevent late replies from reactivating retired state. Replies for another session,
request, or batch are discarded before their ordering counters or draft tokens are
used.

For token-only placement, errors while closing a failed socket preserve the
selected failure policy. Cancellation and shutdown retire pending proposals
even when socket cleanup reports an error.

Protocol v2.2 assigns one server execution deadline to each pipelined prefill
group. Waiting behind an earlier chunk consumes that group's remaining budget.
With older peers, the client waits for each prefill ACK before sending the next
chunk, using the same round deadline. Already executing GPU work is not cancelled.
When buffered final prompt chunks require multiple draft forwards, they also
share the proposal deadline; expired rows do not start another forward.
The server also checks the remaining budget after importing CUDA IPC features,
before starting prefill or proposal computation.

Socket cleanup errors encountered while handling a transport failure do not
replace that failure or interrupt the configured fallback.

Feature schema and frame byte budgets are checked before tensor serialization.
Connections, partial frames, pending bytes and adapter waiters have server-side
bounds. A shutdown grace timeout does not authorize freeing GPU storage whose
completion is unknown; the process must be restarted in that state.

A separate [outcome cache](#all-accepted-outcome-cache) supports one dense
Qwen3 verifier. N:M routing, pooled caching and cross-node execution remain
unsupported. Moving a speculator to another GPU does not by itself overlap its
computation with target verification or guarantee higher throughput.

## All-accepted outcome cache

`execution.policy="outcome_cache"` is an experimental opt-in for unquantized BF16
full-attention Qwen3 target/draft models with BF16 KV and identical tokenizer artifacts.
Use greedy sampling, eager MRV2, synchronous scheduling, target/draft TP1 and
one same-host inline endpoint. Other models, remote pooling and async scheduling
remain outside this policy's initial support.
Combining `outcome_cache` with draft TP2 is rejected before model loading;
use `on_demand` for two draft ranks.

Start the existing token-only server with explicit prediction budgets:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 .venv/bin/python -m vllm.v1.spec_decode.remote.runtime.token_cli \
  --endpoint unix:///tmp/token-draft.sock --device cuda:1 \
  --target-model '<qwen3-target>' --draft-model '<tokenizer-identical-qwen3-draft>' \
  --policy outcome_cache --num-speculative-tokens 3 \
  --cache-max-branches 1 --cache-kv-budget-mib 64 --cache-copy-chunk-blocks 1 \
  --draft-kv-cache-gib 0.25 --max-model-len 256 --max-feature-tokens 256
```

The cache budget caps additional physical branch pages within the draft KV
arena. Branch slots and native copy scratch are initialized before serving.
Executor buffers include the extra branch slots; `--max-batch-size` and
`--max-feature-tokens` continue to bound foreground admission. The cache shares
whole immutable committed prefix pages and copies only the mutable tail before
predicting one all-accepted continuation. Shared pages remain reserved in the
physical arena until both the live and branch leases release them; sharing does
not expand the arena or reuse another request's KV.
Foreground work has priority, and an unfinished or wrong prediction falls back
to ordinary on-demand drafting for that round.

Configure the target with `enforce_eager=True`, `async_scheduling=False` and:

```json
{
  "method": "draft_model",
  "model": "<tokenizer-identical-qwen3-draft>",
  "num_speculative_tokens": 3,
  "execution": {
    "placement": "remote",
    "policy": "outcome_cache",
    "endpoints": ["unix:///tmp/token-draft.sock"],
    "transport": "inline"
  }
}
```

Both peers must negotiate this policy. Ready predictions and consumed hits are
logged separately; successful output alone does not demonstrate a cache hit.
This policy does not promise lower latency or throughput gains.
The [unpaced A100 measurements](https://github.com/seo-rii/vllm/pull/27) have median per-repeat cache/on-demand
throughput ratios of 0.925 and 0.954 for Qwen3-0.6B at batches 1 and 4, and
0.885 and 0.958 for Qwen3-8B with a 0.6B draft at those batch sizes. Batch 1
consumed no cached proposals; batch 4 consumed three across both model pairs
and all repeats. These workloads show no throughput benefit. Keep the default
on-demand policy unless measurements for the intended workload justify the
extra page copies and draft forwards; record readiness delay, copy volume and
extra draft work alongside consumed hits.

For a token-only remote backend, host-side counters can be read without GPU
synchronization through the existing worker RPC. Passing a callable requires
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` before constructing `LLM`; use this opt-in
for trusted local diagnostic callbacks. Production defaults remain unchanged:

```python
def cache_stats(worker):
    return worker.model_runner.speculator.get_cache_stats()

stats = llm.collective_rpc(cache_stats)
```

`received_cache_hits` counts validated cache-origin responses, and `cache_hits`
counts those installed as draft proposals. `cache_verified_proposals` and
`cache_accepted_tokens` count their subsequent target verification and acceptance.
Server ready/hit logs include cumulative counters so these events can be matched
to actual background progress. A correctness test may use the in-process engine
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`) to pause between public `step()` calls until
a real ready event. A multiprocessing core advances independently of frontend
`step()` calls. Such paced execution must not be reported as performance.
Measure cache off/on latency and throughput separately without extra waiting or
progress polling in the timed region.

## Standalone GPU example

For MiMo-7B-RL MTP K=1, start the draft process on its assigned GPU:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 .venv/bin/python -m vllm.v1.spec_decode.remote.server \
  --adapter gpu --endpoint unix:///tmp/vllm-draft.sock --device cuda:1 \
  --target-model XiaomiMiMo/MiMo-7B-RL --draft-model XiaomiMiMo/MiMo-7B-RL \
  --method mtp --num-speculative-tokens 1 --dtype bfloat16 \
  --max-batch-size 8 --max-model-len 4096 --max-feature-tokens 512 \
  --draft-kv-cache-gib 1 --block-size 16 --attention-backend FLASH_ATTN \
  --target-attention-backend FLASH_ATTN
```

Use immutable target/draft revisions for reproducible evaluations. The server
loads and warms its actual prefill/decode path before listening. Required shared
weights are validated without loading the target transformer stack.

Configure the target on GPU 0 with the matching model, revisions and limits:

```python
from vllm import LLM

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-RL",
    max_model_len=4096,
    max_num_seqs=8,
    max_num_batched_tokens=512,
    enable_prefix_caching=False,
    async_scheduling=False,
    enforce_eager=True,
    attention_config={"backend": "FLASH_ATTN"},
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
        "draft_sample_method": "greedy",
        "attention_backend": "FLASH_ATTN",
        "remote_draft": {
            "endpoint": "unix:///tmp/vllm-draft.sock",
            "transport": "inline",
            "request_timeout_ms": 5000,
            "failure_policy": "error",
        },
    },
)
```

Set `VLLM_USE_V2_MODEL_RUNNER=1` for the target too. Both processes must agree on
target/draft device assignments and checkpoint/schema identities. The example's
5000 ms budget is explicit; it does not establish that the 100 ms default is
suitable for long prefills. EAGLE3 currently accepts GPT-OSS and Qwen3 targets;
see [own-KV MTP contracts](remote_mtp.md) for the experimental MTP families.
Prefix caching, input streaming, async scheduling, multimodal input, LoRA and
`draft_lm_head_quantization` are rejected for this backend. Native local speculation retains its own support.

## CUDA IPC feature transport

On a single NVIDIA host, `transport="cuda_ipc"` explicitly selects bounded
CUDA IPC arenas for target features. `auto` continues to select inline.
The receiver copies features into independently owned GPU storage before
acknowledging a lease; this is not zero-copy. Both processes must see the
source GPU, and the actual device UUIDs and peer-copy support are validated.

Producer arenas are reserved before target KV profiling. A timeout, disconnect
or cancel does not prove a device copy has finished: uncertain storage is
quarantined, further admission is fenced, and the process requires restart.
Ordinary request failures preserve the selected target-only or error policy.

For a historical two-GPU correctness, throughput and memory comparison, see
[validation snapshot](remote_placement_validation.md). Its measurements belong
to the recorded source revision and do not validate subsequent commits.
