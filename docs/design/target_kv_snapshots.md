# Target-owned KV snapshots for Q-only draft replay

This design provides bounded tensor snapshots and an experimental remote serving
backend for target-state MTP. Gemma4 remains outside the own-KV remote contract;
its dedicated backend is described below. Native local Gemma4 behavior is unchanged.

Gemma4 assistant attention is Q-only: it reads KV from the last non-shared target
layer of the matching attention type. Its full-attention and sliding-window
layers can have different head dimensions. Sending only target hidden states to
an own-KV draft executor cannot reproduce this contract.

## Snapshot and receiver ownership

`vllm.v1.spec_decode.remote.target_kv` provides these operations:

- `gemma4_source_layers` resolves native target-to-assistant layer mappings.
- `select_draft_position` selects the last accepted query row and its materialized
  context length. Unfinished prefill produces no draft query.
- `snapshot_target_kv` copies the live full/SWA ranges into private page storage.
- `import_target_kv` validates the caller's expected sequence generation, session,
  round, materialized length and layer specs, then makes independent receiver
  caches and block tables.
- `import_target_kv_batch` imports multiple requests into one cache per source
  layer, with disjoint physical page ranges and one block-table row per request.
  Only the zero null page is shared. The singleton API retains its original shape.

Snapshots retain absolute logical page indices. Receiver pages are compacted
physically while block tables preserve those logical positions, including null
holes before a sliding window and native block-table alignment padding. Partial
window prefixes and rejected/uncommitted tails are zeroed. A target page can be
reused after the synchronous snapshot copy completes without changing the
snapshot; separate imports also cannot overwrite each other.

The primitive requires plain BF16 full or sliding attention, equal K/V head
widths within each layer, and supported native layer-compact views. It rejects
quantized, compressed, sparse, mixed-device and unsupported stride contracts.
Different source layers may have different head widths and block sizes.
Scheduler blocks must be converted to kernel-page indices before calling it.

## Budgets and completion

Export and receiver budgets are separate. Page and byte limits cover all layers,
receiver null pages and padded block tables. All destinations are allocated
before source reads begin. A recoverable copy failure fences submitted work;
unknown completion raises `SnapshotCompletionError` and requires abandoning the
runtime. A context-fatal CUDA error must not trigger additional CUDA cleanup.

These are synchronous correctness primitives. The caller must order target
writes on the current stream and prevent concurrent target block reuse until
snapshotting returns. A receiver must remain alive until its consumer finishes;
a consumer on another stream must record its allocations on that stream.
Cancellation is not evidence that GPU readers have completed.

The byte limits bound allocations made by each primitive, not the total target
model allocator peak. Target KV, export snapshot and imported receiver cache may
coexist. The remote integration profiles the export capacity before target KV allocation
and bounds receiver allocations separately. Native local execution does not use
these synchronous copies.

## Validation

CPU tests use native KV specs and allocator views to check source mutation,
independent imports, heterogeneous layers, exact page/byte boundaries, window
remapping, rejection tails, stale identity/length, allocation failure and copy
completion. CUDA tests compare the actual Triton attention consumer against the
original paged cache, including native head-width/window tiling. A skip means the
CUDA path was not validated on that host.

The serving backend uses an explicit paged-KV protocol, bounded transport,
receiver cache binding, embedding-only target weight loading and MRV2 lifecycle
integration. Existing v2 hidden-feature and v3 token-only messages keep their
semantics. Exporting mutable target arenas directly is outside this design.

Actual model and serving validation must identify the exact source/model
revisions, positive acceptance, window-boundary cases, fallback and cleanup.
Primitive tests alone do not establish remote model correctness.

## Experimental Gemma4 remote serving

A dedicated MRV2 backend and assistant-only GPU process exchange snapshots
through protocol v4; v2 own-KV and v3
token-only peers reject its semantics. It derives Gemma4 target/assistant hidden
sizes, layer counts, attention heads, per-layer KV heads and head widths, sliding
windows, vocabulary and target KV owners from their native checkpoint configs.
Assistant head widths, KV head counts and windows must match their selected
target owners; query heads must form valid GQA groups. Native page unification
is preserved. Physical page padding, noncausal attention and incompatible
assistant architectures are rejected before weights are loaded. Snapshot and
receiver capacity for the configured batch and maximum context must fit the
fixed budgets, including worst-case sliding-window alignment. Larger batches
or KV geometry may require a shorter configured context.

Execution requires BF16 weights/KV, target TP1, draft TP1, K1, eager
`TRITON_ATTN`, scheduler block size 16 and `1 <= max_num_seqs <= 16`. Context is capped at
8192 tokens. Prefix caching, async scheduling, multimodal inputs, input
streaming, LoRA, quantization and CUDA IPC are rejected. Ordinary output
streaming remains supported. The native local factory is unchanged.

The config-derived contract is not evidence that every Gemma4 checkpoint has
been evaluated. Real model validation must identify the target and assistant
revisions; CPU geometry and snapshot tests do not establish model accuracy.
Other target-state architectures require their own state adapter. The server
constructs one Q-only query per participating request in a single assistant
forward, preserving differing context lengths and original verifier row order.

Each round exports the last accepted target hidden row, the recovery token and
private full/SWA KV pages. The selected materialized length excludes rejected
positions. No assistant weights are loaded on the verifier. The draft process
loads the assistant and the target embedding, preserving the assistant's own LM
head; it does not load the target transformer or target LM head.

The server retains no request KV between proposals. K0 and unfinished prefill
need no assistant forward because the next snapshot contains authoritative
state. Sequence generation, connection epoch, monotonically increasing round
and materialized length fence results. There is one active verifier connection
with bounded open request identities. Unrelated closes wait until a pending
batch reply has been collected; cancelling a participating request fences that
connection. Import storage is released only after the assistant consumer completes.

HELLO negotiates a checkpoint/geometry fingerprint and bounded snapshot capacity.
A control manifest for the entire batch precedes fixed-shape data frames. Each frame's byte cap is
checked before receiving its body, and both export and receiver budgets are
checked before tensor allocation. The limits are 1024 pages and 128 MiB for the
whole batch, not per request; receiver accounting includes one null page per
source layer and rectangular padded tables. The verifier profiles all retained
exports and hidden-row compaction before allocating target KV. The assistant
warms up singleton and configured maximum batches before listening.

The CLI defaults to one sequence. Singleton clients retain the existing v4 wire
path. A batched client requires the server to acknowledge its requested capacity
before using batch messages; an older singleton server rejects this handshake.
Both processes must set compatible sequence limits. This is batching within one
verifier, not N:M snapshot routing or cross-client scheduling.

Timeout and cancellation close the connection. Affected requests remain
terminal target-only, or raise under `failure_policy="error"`; a late reply
cannot reactivate them. A new request can reconnect. A recoverable assistant
forward OOM fences all rows submitted in that forward while preserving unrelated
open requests. A verifier preparation or transport failure fences its connection.
Socket cleanup errors preserve the selected failure policy and retire cancelled
proposals. If registering a new request runs out of memory, its opened sequence
credits are returned and the request remains terminal target-only until it ends.
New requests can use the recovered capacity.
Unknown CUDA completion and context-fatal errors propagate and
require process restart. Closing a socket never serves as a GPU completion fence.

This implementation copies context on every round and synchronizes the host.
It is an opt-in correctness baseline, not an async-first performance path.
`VLLM_GPU_SYNC_CHECK=error` is incompatible with its synchronous snapshots.
Memory saved on the target and latency/throughput must be measured separately;
long-context snapshot traffic can outweigh the cost of a small local assistant.

Start the draft process with the same staged model paths used by the verifier:

```bash
python -m vllm.v1.spec_decode.remote.runtime.target_kv_cli \
  --target-model /models/gemma-4-E2B-it \
  --draft-model /models/gemma-4-E2B-it-assistant \
  --endpoint unix:///tmp/gemma-draft.sock --device cuda:1 \
  --language-model-only --max-model-len 640 --max-feature-tokens 128 \
  --max-num-seqs 4
```

Then start the verifier (device visibility is configured separately per process):

```bash
vllm serve /models/gemma-4-E2B-it \
  --language-model-only --dtype bfloat16 --enforce-eager \
  --max-model-len 640 --max-num-seqs 4 --max-num-batched-tokens 128 \
  --block-size 16 --no-enable-prefix-caching --no-async-scheduling \
  --attention-config '{"backend":"TRITON_ATTN"}' \
  --speculative-config '{"method":"mtp","model":"/models/gemma-4-E2B-it-assistant","num_speculative_tokens":1,"draft_tensor_parallel_size":1,"remote_draft":{"endpoint":"unix:///tmp/gemma-draft.sock","transport":"inline","failure_policy":"error","request_timeout_ms":1000}}'
```

The same GPU can host both processes for functional validation if their combined
memory fits. It does not demonstrate an advantage from placement on another GPU.
The explicit timeout above includes snapshot preparation, copying and socket I/O;
it is an example configuration, not a measured recommendation. This is a soft
deadline for accepting a result. Blocking CUDA completion fences cannot be
preempted by the host timeout, so it does not bound the time until the call
returns. A process supervisor must enforce hard shutdown bounds. Unconfirmed
copy completion makes the runtime unusable and keeps storage quarantined until
the owning process exits.
