# Shared own-KV draft workers

Independent verifier engines can share same-host token-only or target-conditioned
own-KV draft workers.
Each request stays on the worker selected when its remote sequence opens.
Workers combine proposals from different verifiers into a ragged GPU batch,
with one execution thread owning their mutable model and KV state.

Token-only workers use `method="draft_model"`, inline transport, target TP1 and
draft TP1. Own-KV EAGLE3/MTP workers retain their existing model contracts, target
TP1/TP2 and draft TP1, and accept inline or explicit CUDA IPC transport. Both
require sequential greedy drafting. Gemma target-KV snapshots, predictive
caches, cross-node execution and live sequence migration remain unsupported.

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
(N:1). Target-KV snapshot methods still require exactly one endpoint. Each
worker must advertise sufficient per-batch limits for the verifier; the workers
do not add their individual limits together during handshake.

For EAGLE3/MTP, start workers with the existing
[`--adapter gpu` command](remote_mtp.md) at distinct endpoints and use the same
`execution` object with the corresponding `method`, model and K. The GPU server
uses a bounded owner queue for both single-endpoint and pooled clients. All
workers must provide the same checkpoint and feature contract.

With `transport="cuda_ipc"`, every endpoint must connect at startup. The target
reserves one export arena per endpoint before KV profiling; their memory costs
add together. Feature partitions wait on the producer CUDA event, and each
driver retains its tensors until the session completes or quarantines the
transfer. An IPC session whose arena was quarantined cannot reconnect and
allocate a replacement arena during serving.

## Routing and failure

The client chooses the eligible worker with the fewest locally assigned
sequences and rotates ties. Server admission independently enforces actual
sequence, KV and queue capacity. Routing does not use GPU utilization or claim
knowledge of assignments made by other verifiers.

Proposal rows are partitioned by assignment and their results are restored to
the original verifier order. Each worker has one socket driver. Drivers progress
concurrently against the same client deadline, including request preparation;
splitting a batch never restarts its timeout for each endpoint. New opens and
prefill chunks run on those drivers too, so a stalled preparation on one worker
does not delay launching a healthy worker's proposal.

Under `target_only`, a failed worker's existing sequences become terminal and
return zero valid drafts. Healthy worker results remain usable. New requests
can use a healthy worker, but existing own-KV sequences are never migrated or
revived after an uncertain failure. Under `error`, a worker failure is surfaced
after all submitted endpoint jobs have been drained. Reconnection retains the
underlying session's backoff and generation fencing.
New-request admission probes disconnected workers on their own socket drivers
after backoff while healthy workers continue serving. A worker becomes eligible
again only after handshake completes. Rejoining enables new sequences; old
terminal identities remain fenced.

Closing unrelated requests while a proposal is pending defers their close until
collection. Closing a participating request cancels the pending result and fences
its participating sessions. Cancellation waits for submitted I/O to settle; it
does not interrupt a running CUDA kernel. The deadline is a soft work budget,
not a process response-time guarantee.

## Worker admission and batching

The worker schedules verifier queues in round-robin order. A short coalescing
window groups compatible pending proposals; lifecycle operations remain ordered
with that connection's proposals. Token-only correction tokens are packed first,
followed by one packed forward per live draft depth. Target-conditioned workers
combine rounds with equal effective K, preserving per-sequence and packed-token
feature ordering; K=0 still updates context without a speculative rollout.
Row validation and KV admission
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

The token byte flags above apply to token-only workers. For `--adapter gpu`,
use `--max-queued-feature-bytes` and `--max-feature-bytes-per-verifier` (both
default to 1 GiB); accounting includes IPC descriptor payload sizes. The other
queue options have the same meaning.

These queue limits supplement transport receive limits, connection admission and
the worker's global sequence/KV limits. They do not reserve every verifier's
advertised maximum simultaneously. Bounded cleanup credits remain available
when foreground admission is full so disconnects can retire their own state.

Shutdown drains work under the existing grace period. If its GPU owner has not
finished, the worker retains model and buffer ownership and requires process
replacement; it does not free storage underneath executing work. A cleanly
stopped owner can be started again, including after a recoverable warmup or
endpoint-bind failure. Fatal GPU errors still require process replacement.

Sharing workers and batching do not by themselves establish a throughput gain.
Measure accepted drafts, fallback causes, per-verifier queue latency and goodput,
alongside target/draft memory and the total GPU budget.
