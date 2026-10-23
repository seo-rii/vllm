# Shared token-only draft workers

Independent verifier engines can share same-host token-only draft workers.
Each request stays on the worker selected when its remote sequence opens.
Workers combine proposals from different verifiers into a ragged GPU batch,
with one execution thread owning their mutable model and KV state.

This opt-in path uses `method="draft_model"`, greedy sampling, inline transport,
target TP1 and draft TP1. It retains the token-only runtime's model and tokenizer
contracts. Target-conditioned EAGLE/MTP, Gemma target-KV snapshots, predictive
caches, cross-node execution and live sequence migration are not supported by
this pool.

## Configuration

Start each worker with the same target, draft and tokenizer artifacts. Select
its device explicitly; the number of endpoints does not select GPU placement.
For example, a worker can be started with:

```bash
python -m vllm.v1.spec_decode.remote.runtime.token_cli \
  --endpoint unix:///run/vllm/draft0.sock \
  --target-model <target-checkpoint> \
  --draft-model <draft-checkpoint> \
  --device cuda:0 \
  --num-speculative-tokens 3 \
  --max-batch-size 8 \
  --max-feature-tokens 512 \
  --max-model-len 4096 \
  --max-sequences-per-verifier 4
```

Start another worker at a different endpoint and pass both to each verifier:

```json
{
  "method": "draft_model",
  "model": "<draft-checkpoint>",
  "num_speculative_tokens": 3,
  "execution": {
    "placement": "remote",
    "policy": "on_demand",
    "endpoints": [
      "unix:///run/vllm/draft0.sock",
      "unix:///run/vllm/draft1.sock"
    ],
    "transport": "inline",
    "failure_policy": "target_only",
    "request_timeout_ms": 100
  }
}
```

Between one and sixteen distinct endpoints are accepted. A single endpoint
retains the original session path and can also be shared by multiple verifiers
(N:1). Other remote methods continue to require exactly one endpoint. Each
worker must advertise sufficient per-batch limits for the verifier; the workers
do not add their individual limits together during handshake.

## Routing and failure

The client chooses the eligible worker with the fewest locally assigned
sequences and rotates ties. Server admission independently enforces actual
sequence, KV and queue capacity. Routing does not use GPU utilization or claim
knowledge of assignments made by other verifiers.

Proposal rows are partitioned by assignment and their results are restored to
the original verifier order. Each worker has one socket driver. Drivers progress
concurrently against the same client deadline, including request preparation;
splitting a batch never restarts its timeout for each endpoint.

Under `target_only`, a failed worker's existing sequences become terminal and
return zero valid drafts. Healthy worker results remain usable. New requests
can use a healthy worker, but existing own-KV sequences are never migrated or
revived after an uncertain failure. Under `error`, a worker failure is surfaced
after all submitted endpoint jobs have been drained. Reconnection retains the
underlying session's backoff and generation fencing.

Closing unrelated requests while a proposal is pending defers their close until
collection. Closing a participating request cancels the pending result and fences
its participating sessions. Cancellation waits for submitted I/O to settle; it
does not interrupt a running CUDA kernel. The deadline is a soft work budget,
not a process response-time guarantee.

## Worker admission and batching

The worker schedules verifier queues in round-robin order. A short coalescing
window groups compatible pending proposals; lifecycle operations remain ordered
with that connection's proposals. Correction tokens are packed first, followed
by one packed forward per live draft depth. Row validation and KV admission
failures remain per-row. A shared GPU execution failure invalidates the affected
rows rather than retrying partially written KV.

| Option | Default | Scope |
| --- | --- | --- |
| `--max-pending-jobs` | 64 | Queued plus executing foreground jobs |
| `--max-queued-token-bytes` | 64 MiB | Foreground payload bytes, including executing work |
| `--max-jobs-per-verifier` | 8 | Foreground jobs for one verifier identity |
| `--max-token-bytes-per-verifier` | 16 MiB | Foreground payload bytes for one verifier |
| `--max-sequences-per-verifier` | Worker sequence limit | Live request slots per verifier |
| `--batch-wait-ms` | 1 ms | Maximum coalescing wait before starting a batch |

These queue limits supplement transport receive limits, connection admission and
the worker's global sequence/KV limits. They do not reserve every verifier's
advertised maximum simultaneously. Bounded cleanup credits remain available
when foreground admission is full so disconnects can retire their own state.

Shutdown drains work under the existing grace period. If its GPU owner has not
finished, the worker retains model and buffer ownership and requires process
replacement; it does not free storage underneath executing work.

Sharing workers and batching do not by themselves establish a throughput gain.
Measure accepted drafts, fallback causes, per-verifier queue latency and goodput,
alongside target/draft memory and the total GPU budget.
