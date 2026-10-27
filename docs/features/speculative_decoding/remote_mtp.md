# Standalone own-KV MTP

The experimental standalone runtime can place a target-conditioned MTP head in
another process. It uses the native MRV2 MTP forward path and transports target
hidden states. The target still verifies every proposed token. This placement
can reduce target GPU memory use; it does not imply faster decoding or overlap
between target and draft computation.

## Model contracts

The runtime checks the resolved draft architecture and its matching target,
including hidden width, vocabulary size and the number of checkpoint MTP modules.
It accepts one MTP module, or the Exaone module chain described
below. All supported paths use scalar token positions.

| Resolved draft architecture | Matching target | Draft cache | K restriction |
| --- | --- | --- | --- |
| `MiMoMTPModel` | `MiMoForCausalLM` | Full attention | 1 |
| `ErnieMTPModel` | `Ernie4_5_MoeForCausalLM` | Full attention | 1 |
| `Qwen3NextMTP` | `Qwen3NextForCausalLM` | Full attention | Native sequential rollout |
| `Glm4MoeMTPModel` | `Glm4MoeForCausalLM` | Full attention | Native sequential rollout |
| `Glm4MoeLiteMTPModel` | `Glm4MoeLiteForCausalLM` | Dense MLA | Native sequential rollout |
| `ExaoneMoeMTP` | `ExaoneMoeForCausalLM` / `ExaoneMoEForCausalLM` | Full attention or sliding window, including mixed module groups | One module: native sequential rollout; multiple modules: configured K equals checkpoint module count |
| `NemotronHMTPModel` | `NemotronHForCausalLM` / `NemotronHPuzzleForCausalLM` | Attention/MoE-only MTP block | Native sequential rollout |
| `HYV3MTPModel` | `HYV3ForCausalLM` | Full attention | Native sequential rollout |

These are explicit implementation contracts, not validation of every checkpoint
or topology in each family. A target with hybrid attention or recurrent layers
can have a conventional attention-only MTP head. The target's architecture name
alone does not determine the draft's cache layout.

Exaone MTP modules use the draft checkpoint's module count, attention types and
sliding windows, including when the draft checkpoint is published separately
from the target.

HYV3 and ERNIE MTP heads use the selected draft checkpoint's normalization,
rotary position settings and module layout, including when the draft checkpoint
is published separately from the target.

Native unquantized HYV3 loading checks every MTP-owned parameter, including each
packed projection and local expert shard. Draft-only checkpoints may omit the
target-owned embedding and LM head; the native proposer installs them after
loading. Included tied heads load from the checkpoint embedding, and included
untied heads load from its LM head. An explicit native loader
`enable_weights_track=False` retains the existing unchecked-loading opt-out.

The standalone allocator accepts unquantized full-attention, dense MLA, or
sliding-window KV. Exaone module chains can mix full and sliding-window groups
with a common block size and dtype; other own-KV models require one homogeneous
group. The runtime uses the shared vLLM KV allocator and retains a null block.
Sliding-window attention conservatively reserves the full
context; the attention backend applies the window, and this runtime does not
recycle evicted window pages. Budget the draft KV accordingly.

Sparse/indexer MLA (including any `index_topk` configuration), compressed or
packed KV, mixed MLA/dense groups, other multi-module model families, multimodal
features, mRoPE and mHC are rejected. Nemotron MTP must contain only attention
and MoE layers.
Gemma4's Q-only assistant reads target-owned KV and is not supported by this
own-KV adapter. Its native local path is unaffected.

### Exaone module chains

A checkpoint with `num_nextn_predict_layers=N`, where N is at least two, can use
`num_speculative_tokens=N` with target TP1, draft TP1 and one same-host endpoint.
Choose `transport="inline"` or explicitly choose `transport="cuda_ipc"` on
NVIDIA CUDA. `transport="auto"` continues to select inline.
Declare one `full_attention` or `sliding_attention` entry per module in
`mtp_layer_types`. Sliding modules require positive windows, from
`mtp_sliding_windows` or the checkpoint's existing window settings. The runtime
loads all N checkpoint modules and calls the existing native multi-module
speculator. Every nonfinal prefill chunk sends N known lookahead tokens.

Later modules' trailing KV rows can depend on rejected proposals. The runtime
retains at most N−1 independently owned target input/hidden-state rows per live
sequence. It replays the affected suffix according to the actual accepted count,
including after effective K=0 or scheduler truncation of submitted drafts. Short
prompts retain only their real rows. History is committed only after execution
completes and is released with its sequence generation. Inline history uses CPU
memory; IPC history uses independently owned GPU storage. The receiver completes
its copy before acknowledging the exporter, so retained final-prefill features
and replay history survive exporter arena reuse. Old and staged new history can
coexist during an update.

The startup log reports separate floating-feature payload bounds. For `S` request
slots, `T` feature tokens, model length `L`, module depth `N` and `R` bytes per
floating feature row, committed history uses at most `S*(N-1)*R` bytes. Retained
final-prefill copies can use up to `S*min(T,L)*R` bytes. Including staged history,
incoming, prepared and packed features, a conservative single-verifier payload
bound is `R * [S*min(T,L) + 3*T + 3*S*(N-1)]`. This is separate from native model
buffers, weights, KV backing storage, IPC arenas and allocator overhead. The
draft KV budget covers KV storage; it does not reserve this additional memory.
Warmup executes bounded device-feature prefill before listening and exercises
replay when the token and context budgets permit. It does not validate the peak
of every admitted batch shape.

Effective K can vary from 0 through N. The native forward still executes all N
modules; effective K limits returned drafts and does not currently reduce module
computation. Configured K values different from N and pooled placement remain
unsupported for this path. The single-module path retains its native
sequential rollout behavior.

## Shared weights and execution

The standalone process loads the target embedding and LM head without loading
its transformer layers. Automatic checkpoint downloads use the resolved target
revision. Nemotron embedding names follow its native loader: `model` or
`backbone` prefixes with `embeddings` or `embed_tokens`. Exactly one matching
tensor must be present. Ambiguous aliases, incompatible shapes and quantized
embedding/head tensors are rejected instead of being interpreted as ordinary
floating-point weights.

Remote checkpoint identity uses a resolved immutable Hugging Face commit, or
SHA-256 digests of local checkpoint artifacts and shard indexes. Local directories
are identified by their contents rather than their deployment path. These files
are streamed once per contract construction at startup; hashing adds startup I/O,
not a per-round cost. Checkpoints must remain unchanged between contract creation
and model loading, and throughout serving; hot reload is unsupported. Local
tokenizer artifacts are likewise hashed independently of their directory path.

An alternate `--shared-weights-source` must be a local directory. Its selected
embedding and head tensors must match the target checkpoint in dtype, shape and
contents before GPU materialization. Unpinned alternate repositories and shape-
compatible replacement weights are rejected; this option stages identical shared
weights rather than changing the model being advertised by the handshake.

The runtime remains greedy, eager, same-host and draft TP1. Existing remote
restrictions on target topology, prefix caching, asynchronous scheduling,
resumable input, multimodal inputs and LoRA still apply. Native local MTP does
not inherit these placement restrictions. Set attention backends supported by
the actual draft layout: dense MLA requires an MLA backend; a full-attention
backend used by another checkpoint is not interchangeable.

Remote drafting also rejects KV consumer connectors and KV offloading, including
consumer children of a `MultiConnector`. External KV hits skip target prefill
even when prefix caching is disabled, leaving the remote drafter without the
context it requires. Producer-only connectors retain their existing validation.

## Validation boundaries

CPU tests cover real Transformers configuration normalization, architecture and
feature rejection, shared-weight selection, and the real KV allocator. The KV
tests check independent layer/block writes, dense MLA latent storage and kernel
block splitting, sliding-window rejection/cleanup, and exact budget boundaries.
These tests do not establish attention numerics or model output correctness.

Model validation must record the exact source and checkpoint revisions, compare
native and remote output, require nonzero proposed and accepted drafts, and
check cleanup. Same-GPU tests exercise process separation but do not measure
cross-GPU transport or memory savings. A checkpoint without recorded model
validation remains experimental even when its configuration passes the contract.
