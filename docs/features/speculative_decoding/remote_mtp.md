# Standalone own-KV MTP

The experimental standalone runtime can place a target-conditioned MTP head in
another process. It uses the native MRV2 MTP forward path and transports target
hidden states. The target still verifies every proposed token. This placement
can reduce target GPU memory use; it does not imply faster decoding or overlap
between target and draft computation.

## Model contracts

The runtime checks the resolved draft architecture and its matching target,
including hidden width, vocabulary size and the native MTP module count.
It accepts one MTP module, the Exaone module chain, or the Step3.5 first-head
configuration described below. All supported paths use scalar token positions.

| Resolved draft architecture | Matching target | Draft cache | K restriction |
| --- | --- | --- | --- |
| `MiMoMTPModel` | `MiMoForCausalLM` | Full attention | 1 |
| `MiMoV2MTPModel` | `MiMoV2FlashForCausalLM` | Sliding-window DiffKV, QK192/V128 | First native head, sequential rollout |
| `ErnieMTPModel` | `Ernie4_5_MoeForCausalLM` | Full attention | 1 |
| `Qwen3NextMTP` | `Qwen3NextForCausalLM` | Full attention | Native sequential rollout |
| `Qwen3_5MTP` / `Qwen3_5MoeMTP` | Flat-text `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` | Full attention | One checkpoint head, native sequential rollout |
| `Glm4MoeMTPModel` | `Glm4MoeForCausalLM` | Full attention | Native sequential rollout |
| `Glm4MoeLiteMTPModel` | `Glm4MoeLiteForCausalLM` | Dense MLA | Native sequential rollout |
| `DeepSeekMTPModel` | `DeepseekV3ForCausalLM` | Dense MLA | One checkpoint head, native sequential rollout |
| `ExaoneMoeMTP` | `ExaoneMoeForCausalLM` / `ExaoneMoEForCausalLM` | Full attention or sliding window, including mixed module groups | One module: native sequential rollout; multiple modules: configured K equals checkpoint module count |
| `NemotronHMTPModel` | `NemotronHForCausalLM` / `NemotronHPuzzleForCausalLM` | Attention/MoE-only MTP block | Native sequential rollout |
| `HYV3MTPModel` | `HYV3ForCausalLM` | Full attention | Native sequential rollout |
| `Step3p5MTP` | `Step3p5ForCausalLM` | Homogeneous full attention or sliding window | One module: native sequential rollout; multiple SWA modules: configured K=1 |

These are explicit implementation contracts, not validation of every checkpoint
or topology in each family. A target with hybrid attention or recurrent layers
can have a conventional attention-only MTP head. The target's architecture name
alone does not determine the draft's cache layout.

The draft's effective `max_model_len`, derived from its own checkpoint position
settings and any speculative context limit, must cover the target's configured
`max_model_len`. Startup rejects shorter drafts before native execution. Reduce
the target context limit to use such a checkpoint. Setting a round's K to zero
does not bypass this requirement: MTP still updates draft context in that round.

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
features, arbitrary mRoPE coordinates and mHC are rejected. Nemotron MTP must contain only attention
and MoE layers.
Gemma4's Q-only assistant reads target-owned KV and is not supported by this
own-KV adapter. Its native local path is unaffected.

### DeepSeek-V3

The DeepSeek-V3 path requires native, unquantized BF16 target and draft weights,
BF16 KV, target TP1, draft TP1 and one inline endpoint. `transport="auto"`
continues to select inline. Both dense feed-forward and routed-expert blocks use
the existing native DeepSeek MTP implementation; "dense MLA" describes its KV
layout, not the absence of MoE layers. The native configuration override resolves
`DeepseekV3ForCausalLM` to `DeepSeekMTPModel` with one checkpoint head. K controls
sequential rollout of that head rather than the number of checkpoint heads.
Target and draft checkpoint configurations must agree on the fields the native
head reads from the target: layer indices, head count and expert geometry.

Any `index_topk` attribute, including a zero value, is rejected on either model.
Sparse/indexed DeepSeek variants, mHC, multiple MTP heads, FP8 checkpoint
metadata, CUDA IPC and pooled placement are outside this contract. Published
FP8 checkpoints require a separately validated loading and execution contract;
passing a synthetic BF16 configuration does not establish their support.
Select an MLA attention backend supported by the device for both native and
remote comparison, and keep the MLA prefill backend consistent.

### DeepSeek-V3.2 sparse MLA (validation pending)

The experimental V3.2 contract is separate from dense DeepSeek-V3. It pairs one
native BF16 MLA cache (576 elements per token) with its own indexer cache
(132 bytes per token: FP8 values plus FP32 scale). Both use 64-token logical
blocks, with distinct physical storage and separate native metadata builders.
This does not enable arbitrary byte caches, compressed MLA, or FP8 main KV.

The candidate accepts a flat-text `DeepseekV32ForCausalLM` target and one native
`DeepseekV32MTPModel` head, BF16 weights/features/main KV, target/draft TP1 and
one inline endpoint. Top-k is 512, 1024 or 2048; indexer heads are 64 by 128.
The draft retains its constructor-owned top-k buffers, native streams and
completion events. Iteration index sharing, target index reuse, HiSparse,
quantized model weights, multiple heads, GLM DSA, IPC and pooling are excluded.

Actual NVIDIA model loading, sparse/indexer kernels, proposal correctness and
acceptance still require validation on Hopper with the native sparse backend
and DeepGEMM build. A100 cannot run this native path. CPU allocation/binding
checks and checkpoint serialization do not establish model-serving support.
Do not treat this candidate as a validated replacement for native V3.2 serving.

### Qwen3.5 text

Dense and MoE Qwen3.5 use their native flat-text target architectures and one
full-attention MTP head (`mtp_num_hidden_layers=1`). The target retains its native
Gated DeltaNet state and speculative rollback; the standalone head owns only
attention KV and shares the target checkpoint's embedding and LM head.

Use native BF16 activations and KV, target TP1, draft TP1, sequential greedy
drafting and one inline endpoint. Dense and MoE checkpoints can use unquantized
BF16 weights. Dense Qwen3.5 additionally accepts the serialized per-tensor FP8
profile below. K controls sequential
rollout of the single head. `transport="auto"` continues to select inline.
Native weight-completeness checks include every split Q/K/V projection in
dense and MoE BF16 MTP checkpoints, plus gate/up projections in dense checkpoints.
Complete fused projections also load.
Leaving `enable_weights_track` unset or setting it to `null` keeps these native
BF16 checks enabled; an explicit `false` disables them.

Saved text configurations may inherit three-axis `mrope_section` and
`mrope_interleaved` metadata. The existing native flat-text configuration path
removes those fields from the target's runtime RoPE configuration, and both the
native and standalone MTP proposers use scalar logical positions. The checkpoint
files are unchanged. The draft class advertises multimodal capabilities, but
this contract accepts only the matching flat-text target; it does not transport
image/video embeddings or arbitrary position axes.

Conditional-generation wrappers, multimodal inputs, additional MTP heads,
other quantized checkpoints, CUDA IPC and pooled placement remain outside this
contract. Small seeded model tests establish loading, state and placement
correctness rather than trained quality or performance.

#### Dense Qwen3.5 FP8 weights

Use an actual serialized E4M3 checkpoint with native `quant_method="fp8"`,
`activation_scheme="dynamic"`, per-tensor weight scales, and BF16 activations
and KV. Target and draft must use matching native text/quantization configs.
The runtime preserves native loading and kernel selection; the shared embedding
and LM head must remain unquantized. On A100, native vLLM uses the Marlin W8A16
path: FP8 stored weights with BF16 inputs. This is distinct from hardware FP8
matrix multiplication. Keep `VLLM_BATCH_INVARIANT=0` for the Marlin path.

The standalone loader enables native parameter-completeness tracking for this
profile, including its weight scales. Use the native default checkpoint loader;
explicitly disabling `enable_weights_track` is rejected. Tracking verifies every
split Q/K/V and gate/up projection and its scale; complete fused projections
remain supported. Before allocating draft KV or accepting requests, the runtime
checks the loaded native projection kernels, weight storage, scale metadata and
shared embedding/head dtype. It
accepts native Marlin's packed storage and transformed scales rather than
requiring the checkpoint's on-disk layout after loading. The startup log reports
the resolved weight profile and kernel/storage types. These checks add no
per-step tensor copies or GPU synchronization.

This profile applies only to dense `Qwen3_5ForCausalLM` / `Qwen3_5MTP`, with the
same single-head, TP1 and inline limits. MoE FP8, block-scaled or compressed
weight formats, static activation scales, online quantization and FP8 KV remain
excluded. Merely adding quantization metadata to BF16 tensors does not create
an FP8 checkpoint. Seeded serialization and serving tests are separate from
published quantized-checkpoint quality and performance evaluation.

### MiMo-V2 Flash

MiMo-V2 Flash uses its native first MTP head, with sequential rollout for K>1.
Additional heads in the checkpoint are not executed. The target uses its own
global/sliding-window attention and MoE layers; the MTP block has its own
sliding-window KV and a dense MLP. This is separate from the MiMo-7B K=1 path.

The contract requires native unquantized BF16 target/draft weights and KV,
target TP1, draft TP1 and one inline endpoint. Global and sliding attention must
both use QK width 192 and V width 128. Global and sliding head counts can differ;
each positive Q head count must be divisible by its positive KV head count.
The target must declare one global/sliding entry
per layer in `hybrid_layer_pattern`, and `sliding_window_size` must be positive.
Target and draft configurations must agree on the native head's attention
window, RoPE and layer settings, which its constructor reads from the target.
Use a native DiffKV backend supported by the device and keep it consistent
between native and remote execution. Different V widths within one process
are outside this contract.

The standalone KV allocator retains the full context even though the native
attention kernel applies the sliding window. BF16 synthetic checkpoint tests
do not establish support for published FP8 weights. Pro's fused checkpoint
format, Omni, quantization, additional MTP heads, pooled placement and CUDA IPC
require separate validation and remain excluded.

### Step3.5 checkpoint-owned heads

Step3.5 shares the target embedding and loads each MTP module's own LM head
from its checkpoint. The standalone process does not load or allocate the
unused target LM head. Missing MTP head weights fail native checkpoint
completeness validation; the target head is not a replacement.

This contract requires native unquantized BF16 target/draft weights and KV,
target TP1, draft TP1, one inline endpoint, GQA and dense MTP feed-forward
layers. A single full-attention or sliding-window module can use native
sequential rollout. Multiple modules require homogeneous sliding-window
attention and configured K=1. That configuration uses the native single-head
proposer: all checkpoint modules and their KV are allocated, but only the first
module executes. It does not implement multi-module rollout or replay.

Declare one MTP attention and MLP type per checkpoint module, with positive
sliding-window and head settings where applicable. Full-attention MTP heads
need native configuration geometry covering their absolute layer indices;
homogeneous configurations may explicitly set `per_layer_config` to null.
The public Step3.5 BF16 metadata has three sliding-window MTP modules.
Small BF16 fixtures validate this contract, not the full published checkpoint's
quality or resource requirements. Step3.7 multimodal wrappers, FP8 checkpoints,
mixed MTP attention groups, CUDA IPC and pooling require separate contracts.

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

The standalone process loads only the target embedding and LM head consumed
by the native draft loader, without loading target transformer layers.
Checkpoint-owned Step3.5 heads omit the target LM head. Automatic checkpoint
downloads use the resolved target
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
consumed embedding and head tensors must match the target checkpoint in dtype, shape and
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
