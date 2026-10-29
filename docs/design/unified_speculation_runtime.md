# Draft backends and token-only runtime

The native local speculator remains the default. An opt-in backend boundary
adds local wrappers, same-step dispatch/collect and a token-only standalone
draft. Full-attention token-only execution additionally supports bounded
target-prefix replay and eager MRV2 async scheduling. Predictive caching, N:M
pooling, stochastic remote sampling, general prefix reuse, cross-node transport
and draft TP expansion remain outside this implementation.

## Native local execution

With neither `execution` nor `remote_draft`, the factory returns the existing
speculator and the runner uses its existing proposal path. An explicit local
configuration opts into a sidecar around that same object:

```json
{
  "method": "draft_model",
  "model": "<compatible-draft-checkpoint>",
  "num_speculative_tokens": 3,
  "execution": {"placement": "local", "policy": "on_demand"}
}
```

The sidecar does not reload weights, initialize a process group, serialize
features, or synchronize CUDA. The runner retains the original speculator for
its type-specific hooks, attention metadata, workspace, and KV ownership. Local
model capabilities are not restricted by the remote support matrix.

`SpeculatorOutput` separates token capacity from per-row valid lengths. The
local wrapper caps those lengths by effective K and stops at the first invalid
token. Conversion to the runner's `-1` representation occurs at installation.
This conversion has an opt-in cost. It does not imply that every existing
native speculator reduces its compute when K shrinks.

## Same-step backend contract

`DraftStep` freezes request order, ragged query lengths, prefill flags, and
effective K. `LocalExecutionContext` retains the native inputs separately from
wire messages. Its bindings are frozen; its tensors and InputBatch are borrowed,
not immutable copies. The caller must not mutate or reuse them before collection.

The backend exposes dispatch, collect, cancel, request close, and shutdown.
Exactly one handle may be pending. Handles are owner-bound and consumed once;
double collection, foreign handles, and stale completion are caller errors.
Local completion means work was enqueued in the native stream ordering, not that
the host synchronized the device.

The runner keeps dispatch and collect adjacent inside the
existing workspace lane, after authoritative sampling and request postprocessing.
It establishes an explicit boundary without claiming additional compute overlap.
The separately gated token-only async mode retains this same-step ordering.
Moving collection across another forward requires a separate snapshot and
buffer-lease design.

All target TP ranks invoke backend and lifecycle methods in the same order.
Rank-zero preparation and transport errors are carried to the collect-side CPU
broadcast. This handles runtime failures; it cannot recover arbitrary asymmetric
API misuse, such as one rank never collecting.

Cancelling a local handle discards its result without closing the native request.
Cancelling uncertain remote work fences the connection and its pending result.
Requests whose own-KV state is no longer known remain terminal locally. CUDA IPC
leases retain their existing quarantine rules; cancellation does not prove that
a remote reader finished.

## Configuration and protocol boundaries

Execution separates placement from the speculative method. Only `on_demand`
policy and one endpoint are accepted. Local execution rejects remote endpoint,
transport, failure-policy, and timeout options. Supplying both legacy
`remote_draft` and `execution` is an error. Legacy remote configuration keeps its
existing defaults and v2 semantics; `transport="auto"` still means inline.

```json
{
  "method": "eagle3",
  "model": "<supported-draft-checkpoint>",
  "num_speculative_tokens": 3,
  "execution": {
    "placement": "remote",
    "policy": "on_demand",
    "endpoints": ["unix:///run/vllm/draft.sock"],
    "transport": "cuda_ipc",
    "failure_policy": "target_only"
  }
}
```

The existing v2 target-conditioned runtime remains responsible for its checkpoint
capabilities, feature schema, state fencing, deadlines, and IPC ownership.
Independent AR execution uses a separate v3 semantic contract. Required token
deltas and proposal identities are not optional v2 fields, and no request is
retried through another protocol after an ambiguous send.

Like v2.1, v3.1 adds an optional `execution_timeout_ms` to prefill and proposal
commands. The client refreshes this relative budget after sending token frames;
the server measures queue wait against its own monotonic clock and checks expiry
under the adapter lock before prefill and before each proposal row. Expired rows
return `TIMEOUT` and release their own KV and request slots, preserving other
sessions. This prevents starting stale queued work; it does not interrupt an
already executing forward or impose a hard GPU completion deadline. Both peers
must support the field for this guarantee. A v3.0 peer remains wire-compatible
but ignores or omits it, retaining the previous queue-execution behavior.

Capabilities belong to the concrete v2 and v3 runtime contracts. Owning KV
does not imply independence from target features or support for future prediction.

## Token-only state contract

Token-only execution consumes authoritative token deltas, not hidden features.
Its state distinguishes:

| Quantity | Meaning |
| --- | --- |
| C | Authoritative committed token count |
| M | Materialized committed KV prefix |
| S | Materialized end, including uncommitted draft tail |
| R | Reserved capacity, not evidence of valid KV |

At successful update boundaries M equals C, including K=0. A proposal of length
K may have only K-1 draft tokens materialized. Full acceptance can therefore
require forwarding both the last draft and the target's bonus token. Outcome
application validates the exact previous proposal, base prefix, offered count,
accepted prefix, token ranges, and generation before mutating state. Rejected
tail KV cannot be treated as committed state.

A standalone loader and forward path reuse native model execution without a
target shell or a second inference engine. The supported mode is greedy,
text-only native decoder models with homogeneous causal full attention,
same-host inline, target TP1 and draft TP1, with no heterogeneous vocabulary or
parallel drafting. Qwen3, Llama and Mistral configured with `sliding_window=null`
are examples; the runtime checks model capabilities and the loaded draft's
actual KV specification rather than requiring a model-type name. Loaded drafts
must accept token IDs and positions alone and use the default decoder model
state. Target and
draft may have different hidden sizes, layer counts and attention geometry.
Sliding-window, chunked, sparse/MLA, recurrent/hybrid, encoder-decoder,
multimodal and Transformers-backend execution remain unsupported. Select the built-in
`tokenizer_mode="hf"` explicitly.
Tokenizer compatibility is checked against the draft checkpoint's own artifacts,
including token mapping, special tokens, and frontend metadata. This initial
contract requires identical tokenizer artifact sets; a chat-template difference
also changes the fingerprint. Checkpoint identity uses immutable repository
revisions or hashes of deployed local weight files, independent of their path.

Start the v3 server through
`python -m vllm.v1.spec_decode.remote.runtime.token_cli --help`. Stage both
checkpoints and tokenizer assets at immutable revisions before launch. The server
requires the same target, draft, context limit, K, dtype and tokenizer contract
as the verifier. It completes model warmup before exposing its socket.

For the verifier, combine `tokenizer_mode="hf"` with this speculative configuration:

```json
{
  "method": "draft_model",
  "model": "<staged-compatible-draft-checkpoint>",
  "revision": "<immutable-draft-revision>",
  "num_speculative_tokens": 3,
  "execution": {
    "placement": "remote",
    "policy": "on_demand",
    "endpoints": ["unix:///run/vllm/token-draft.sock"],
    "transport": "inline",
    "failure_policy": "error"
  }
}
```

Use greedy sampling and retain target TP1. Prefix caching and async scheduling
are available only for the full-attention combinations below; other remote
combinations require them disabled. A same-GPU
two-process run is a lifecycle test; normal single-GPU placement remains native
local execution. Neither `auto` transport nor GPU visibility selects placement.

### Cached target prefixes

Set `enable_prefix_caching=True` on the verifier when both target and draft use
an existing native causal full-attention token-only contract, target/draft TP1
and one inline endpoint. This includes full-attention Qwen3, Llama and Mistral
configurations. Sliding-window, MLA and recurrent state, pooled placement,
input streaming and target-conditioned MTP/EAGLE placement remain excluded.

On a target cache hit, the worker retains the authoritative CPU token prefix
and replays it into a fresh draft sequence before sending the scheduled suffix.
Replay chunks respect the negotiated token/frame limits and share the original
round deadline with open, suffix prefill and proposal. The draft still computes
that prefix; this does not reuse another request's draft KV or guarantee lower
TTFT. A long replay may need a larger explicit request budget.

Finish, cancellation and preemption retire the draft generation and its pending
CPU prefix. Resuming a preempted request replays its current authoritative
prefix, including any committed output tokens. A replay failure follows the
existing terminal target-only or strict policy; a late ACK cannot resume it.

### Async verifier scheduling

Set `async_scheduling=True` and `enforce_eager=True` on the verifier, with
`VLLM_USE_V2_MODEL_RUNNER=1`, for an existing native causal full-attention
token-only target/draft combination. Target and draft must use TP1 and one inline
endpoint. The existing greedy, tokenizer and model/weight contracts still apply.
Prefix caching may be enabled independently when the cached-prefix contract
above is satisfied.

The backend keeps dispatch and collect adjacent, with at most one pending
proposal. Async verifier scheduling does not move collection across another
forward or make a single request's target and draft computations overlap.
Logical valid draft lengths remain separate from the scheduler's physical
placeholder slots; an invalid slot is never an authoritative token delta.
The existing terminal fallback, generation fencing and request retirement rules
remain in effect.

Target-conditioned MTP/EAGLE, Gemma target-KV snapshots, recurrent or MLA state,
sliding-window attention, pooled endpoints, CUDA graphs and larger TP are
excluded from this async mode. Native local execution retains its own support.
This mode is experimental; no performance improvement is established.

### Batched token execution

The state machine validates each sequence's proposal identity and reserves its
KV independently, then packs correction tokens into ragged native forwards.
Each following draft depth batches the rows that still need tokens, including
batches with different effective K and client round IDs. Both token and request
scratch limits bound every forward. Sampled IDs return through one batched CPU
copy per forward rather than one scalar synchronization per row.

Admission and deadline failures stay local to the affected row. An actual GPU
forward failure invalidates every row submitted in that forward; it is not
retried against potentially modified KV. A context-fatal or completion-unknown
failure prevents further GPU submissions. Deadlines govern admission and
submission boundaries, not a hard time bound on an already executing kernel.
This batches on-demand work; it does not precompute outcomes or create target
and draft overlap.

## Validation

Record the tested source, immutable model/tokenizer revisions, effective
configuration, hardware and results using
`benchmarks/spec_decode/validation_manifest.example.json`. Keep CPU, socket,
CUDA primitive and model/HTTP evidence separate.

Compare target-only, native local, explicit local wrapper and supported remote
modes. Check output tokens together with actual draft/acceptance counts and
fallback reasons; output equality alone can conceal an unused draft backend.
Same-GPU runs do not establish cross-GPU transfer or target TP2 behavior.

Performance comparisons include TTFT, TPOT, tail latency, throughput, target
and draft memory, and additional GPU cost. Dispatch/collect separation alone
does not demonstrate compute overlap.

## Related work

[RFC #42109](https://github.com/vllm-project/vllm/issues/42109) proposes standalone
drafting, outcome caches and pooling. Its
[fixed PoC snapshot](https://github.com/laviier/vllm/commit/2132e0599aba8dd252b6e1880547c6617b709a2c)
is a design reference for the backend boundary and token-state contracts.

[Speculative Speculative Decoding v1](https://arxiv.org/abs/2603.03251v1)
motivates outcome precomputation. Appendix E's feature-dependency discussion is
why target-conditioned shadow prediction remains separate from exact on-demand
state. Token agreement alone cannot promote approximate KV into exact live KV.
