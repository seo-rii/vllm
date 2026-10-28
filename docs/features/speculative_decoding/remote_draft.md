# Remote draft placement (experimental)

Remote placement separates a target-conditioned speculator from the target
process. The target remains responsible for verification and sampling. Placement
is configured independently of the speculative method; native local execution
remains the default.

The standalone GPU runtime provides same-host target-conditioned execution with
independent draft KV. It supports greedy, eager MRV2 execution, target TP1 or TP2
and draft TP1. The protocol test adapter is for development only.

Independent `method="draft_model"` placement uses the
[token-only runtime](../../design/unified_speculation_runtime.md#token-only-state-contract).
Its support includes attention-only models, selected dense MLA drafts and a
bounded NemotronH recurrent draft with no MTP or expert blocks. Recurrent drafts
use a dedicated ordinary model state and replay the full computed prefix on each
forward; this trades extra computation for independent state placement. The
linked contract describes tokenizer compatibility, state/token-page memory
accounting and the single-endpoint inline limits.
The recurrent executor preserves the runtime's effective `draft_load_config`,
including its checkpoint format and loader options, when constructing the
ordinary draft model.

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

Feature schema and frame byte budgets are checked before tensor serialization.
Connections, partial frames, pending bytes and adapter waiters have server-side
bounds. A shutdown grace timeout does not authorize freeing GPU storage whose
completion is unknown; the process must be restarted in that state.

Same-host [shared draft workers](draft_pool.md) provide sticky N:M routing
within their supported model and transport contracts. Predictive caching and
cross-node execution remain unsupported. Moving a speculator to another GPU
does not by itself overlap its computation with target verification or guarantee
higher throughput.

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
suitable for long prefills. EAGLE3 accepts GPT-OSS, Qwen3, Llama, Qwen2 and Mistral targets;
see [own-KV MTP contracts](remote_mtp.md) for the experimental MTP families.
Prefix caching, input streaming, async scheduling, multimodal input, LoRA and
`draft_lm_head_quantization` are rejected for this backend. Native local speculation retains its own support.

### Llama EAGLE3 draft feature contract

Llama, Qwen2 and Mistral remote placement requires the matching native target
(`LlamaForCausalLM`, `Qwen2ForCausalLM` or `MistralForCausalLM`) and an
`Eagle3LlamaForCausalLM` draft (including its `LlamaForCausalLMEagle3` alias),
matching hidden widths and vocabularies, unquantized weights, target TP1 and
draft TP1. Use one inline endpoint; `auto` selects inline. CUDA IPC and pooled
execution for these target/draft combinations are not enabled by this contract.

Qwen2 and Mistral additionally require full attention in both models, with no
resolved target, draft or global cache sliding window. Qwen2's
`use_sliding_window=False` resolves to full attention; Mistral checkpoints must
declare `sliding_window=None`. Dual-chunk attention and conditional adaptive
normalization (`ada_rms_norm_t_cond`) are unsupported. Use Hugging Face
checkpoint tensor names for Mistral, including `model.embed_tokens.weight` and
`lm_head.weight` when untied; the remote shared-weight loader does not remap
original Mistral `tok_embeddings`/`output` keys.

The runtime follows native auxiliary-layer selection and capture order. Layer
position `0` is the target embedding input; positions `1..N` capture the residual
stream after a target layer, before the final model normalization. Explicit
checkpoint selections are sorted and deduplicated as in the native target.
Without a selection, the native defaults are `(2, N // 2, N - 3)`. The resolved
positions must exist and their count and hidden width must match the draft's
auxiliary projection. In particular, small test models may need explicit layer
positions because their defaults can repeat or fall outside the model.

For a Llama EAGLE3 draft, `num_aux_hidden_states` determines the projection count;
`num_aux_layers` belongs to the Qwen3 EAGLE3 draft contract. With
`eagle_config.use_aux_hidden_state=False`, the draft instead consumes the target's
final normalized hidden state without an auxiliary projection. Both peers
fingerprint the resolved feature contract, and the server checks its actual
loaded projection before allocating draft KV and warming the model.

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
