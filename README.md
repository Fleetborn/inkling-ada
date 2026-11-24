![Inkling Ada banner](assets/inkling-ada-banner.jpg)

# Inkling Ada

Inkling Ada is an open-source experiment in serving Thinking Machines’ [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on an RTX 4090. It builds on [NInfer-all](https://github.com/iamwavecut/ninfer-all) for CUDA execution, model loading, and an OpenAI-compatible API. It pairs an SM89 tensor-core attention patch with reproducible serving benchmarks and raw results.

[Quick start](#get-started) · [Benchmarks](#rtx-4090-benchmarks) · [Contributing](#contributing)

## Get started

### One-shot agent deployment

The [Inkling Ada deploy skill](.agents/skills/inkling-ada-deploy/SKILL.md) prepares the pinned SM89 build and model, starts the loopback server, and verifies a completed reply.

Codex discovers the skill from this checkout. To install a personal copy for use from another workspace, run these commands in the environment where your agent runs.

```bash
git clone https://github.com/Fleetborn/inkling-ada.git
cd inkling-ada
mkdir -p "$HOME/.agents/skills"
cp -R .agents/skills/inkling-ada-deploy "$HOME/.agents/skills/"
```

If you already cloned the repo, run the last two commands from its root. For another agent, copy the [entire skill folder](.agents/skills/inkling-ada-deploy) into that agent's skills directory. Include `SKILL.md`, `agents/`, `references/`, and `scripts/`.

Then give your agent one request.

> Use `$inkling-ada-deploy` to start Inkling 2 Mini on my Linux RTX 4090 and verify a completed reply.

### Manual setup

Use a supported NVIDIA driver and an Inkling-capable CUDA Linux archive from [NInfer-all releases](https://github.com/iamwavecut/ninfer-all/releases). Verify the numbered archive parts against its SHA-256 manifest and check the distribution requirements for that release. On older Linux hosts, use its Dockerfile and NVIDIA Container Toolkit as described in the [NInfer Linux release guide](https://github.com/iamwavecut/ninfer-all/blob/master/docs/release-archive-linux.md).

Download the [UD-IQ4_XS NInfer artifact](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3), then start the stock upstream server in one terminal.

```bash
bash serve.sh /path/to/ninfer-serve /path/to/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
```

Run the six completed-answer fixtures in a second terminal. Use the same run number for each build or drafting mode. The suite adds a verbatim handoff and a one-operator Python edit to the four fixtures in the benchmark table. Both new cases require the full expected answer and a normal stop.

```bash
python3 benchmark_completed.py 8080 1
```

The `serve.sh` launcher binds the server to loopback, keeps experts on the GPU, and uses a 16,384-token context, FP16 KV, and one active sequence. Append `8080 dspark` to the `serve.sh` command to enable speculative decoding. The commands above run the stock release. For the patched binary and matched scalar baseline, use the [pinned SM89 build instructions](docs/sm89-tensor-qk.md#apply-and-reproduce).

### Compare engine builds and drafting modes

Build A from NInfer `18029c8e0ce2555bb499e04be24d7a4e9c12a5bc` with our [SM89 tensor-QK patch](docs/sm89-tensor-qk.md#apply-and-reproduce). Build B from clean NInfer [`7894c119f638eb6cae52e4b3e8efc53ebaad216b`](https://github.com/iamwavecut/ninfer-all/tree/7894c119f638eb6cae52e4b3e8efc53ebaad216b). Target SM89 and record both binary hashes and CUDA compiler versions. Use `bash serve.sh "$engine" "$model" 8080 none` for both. Use the older UD-IQ4_XS artifact for this two-build, plain-decoding comparison. Verify its SHA-256 is `a4a1bbfbdc1150cecc9df2bc12ae87c076330bff81579e243e7875115329bb54`. [Pinned artifact and build instructions](.agents/skills/inkling-ada-deploy/references/prepare.md)

Run `python3 benchmark_completed.py 8080 "$run" > "$label-$run.jsonl"` with a distinct label per build or mode. Restart the benchmark server at the start of each of four blocks, including between the two B blocks. Run A with IDs 1 and 2, B with 1 and 2, B with 3 and 4, then A with 3 and 4. Use the same smoke warmup and fixture order throughout, with persistent prefix caching off. Pair by run and prompt hash. Check all six normal stops and fixture validations, then compare per-fixture median time to first visible answer and completion. Report answer hashes, output and cached token counts, and any differences in cache hits or output lengths alongside latency.

For a separate, build-B-only drafting comparison, switch to the newer UD-IQ4_XS artifact containing the **mainline DSpark drafter**. Verify SHA-256 `d96986947048ed3ac643da0ed6c389e8ab821f66d10e2abf1f4347224b4180db` and use that file for every mode, including plain decoding. Build B can use the older pinned artifact for plain decoding. DSpark requires the newer layout. [Matched UD-IQ4_XS artifact](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3/tree/0baae7fdb07506ec285bd334f2671a889aa890c0) · [Artifact requirements](https://github.com/iamwavecut/ninfer-all/blob/7894c119f638eb6cae52e4b3e8efc53ebaad216b/docs/inkling2.md#L189-L202)

```bash
common=(--host 127.0.0.1 --port 8080 --model-id inkling-2-mini --max-context 16384
        --max-concurrency 1 --expert-residency device --kv-dtype fp16)
# Start one server at a time. Repeat the paired comparison against plain for each mode.
"$engine_b" "$model_mainline" "${common[@]}" --spec none
"$engine_b" "$model_mainline" "${common[@]}" --spec none --lookup-ngram 3 --draft-tokens 3
"$engine_b" "$model_mainline" "${common[@]}" --spec dspark --draft-tokens 3
"$engine_b" "$model_mainline" "${common[@]}" --spec dspark --draft-tokens 3 --lookup-ngram 3
```

Check the server log for the selected mode and record proposed and accepted draft counts where available. The lookup-only run proposes continuations already present in the sequence. DSpark uses the artifact's drafter. In the combined mode, the scheduler tries lookup first and can call DSpark when lookup misses. [Inkling draft selection](https://github.com/iamwavecut/ninfer-all/blob/7894c119f638eb6cae52e4b3e8efc53ebaad216b/src/runtime/engine/inkling2_core.cpp#L2439-L2466)

## RTX 4090 benchmarks

### November 25, 2026 · Trappist

Clean NInfer `7894c11` completed the 150K retrieval in **35.08 s**, versus **174.01 s** for our patched `18029c8` and **306.29 s** for production Strata 0.2.21. Both NInfer builds used CUDA 12.9.86 for SM89, the same UD-IQ4_XS file, GPU-resident experts, MXFP8 KV and a 163,840-token context. Strata ran its production SQ-20 weights, FP8 KV, 393,216-token cap and 64-token chunks with vision loaded. All three served one benchmark request at a time with drafting off. [Configurations and raw requests](results/rtx4090-strata-0221-ninfer-context-drafting.json)

| Prompt tokens (NInfer / Strata) | Patched Ada (s) | Clean NInfer (s) | Strata (s) |
| --- | ---: | ---: | ---: |
| 2,161 / 2,163 | 1.38 | 1.17 | 1.76 |
| 8,461 / 8,463 | 2.95 | 1.98 | 6.68 |
| 15,661 / 15,663 | 5.60 | 3.15 | 13.07 |
| 31,662 / 31,664 | 13.75 | 5.83 | 30.49 |
| 64,062 / 64,064 | 40.13 | 12.31 | 79.43 |
| 150,062 / 150,064 | 174.01 | 35.08 | 306.29 |

Each cell is the completion time for one request. All 18 requests returned the expected answer and reported zero cached prompt tokens. At 150K, time to first streamed token was 172.23 s for patched Ada, 33.93 s for clean NInfer and 305.99 s for Strata. Their sampled peak GPU memory was 20,222, 20,246 and 21,796 MiB respectively, sampled every 500 ms.

On the short fixtures, Strata completed arithmetic in 0.071 s and prose in 0.766 s in one request each. The corresponding plain NInfer medians across four requests were 0.419 s and 2.300 s for patched Ada, and 0.388 s and 2.257 s for clean NInfer. In a separate drafting comparison, clean NInfer used the newer UD-IQ4_XS artifact with the mainline drafter, 16K context and FP16 KV in every mode. Across four requests per fixture and mode, median completion for verbatim copying was **1.237 s with DSpark** and **1.220 s with lookup plus DSpark**. The code edit took **0.992 s** and **0.961 s** respectively. All 96 drafting-comparison requests passed. [Per-request answers and timings](results/rtx4090-strata-0221-ninfer-context-drafting.json)

### Earlier Strata 0.2.19 comparison

Each benchmark ran one stream on the same RTX 4090. Ada used UD-IQ4_XS and the tensor QK patch. Strata v0.2.19 used its Inkling engine 0.1.65 and SQ-20 weights. Both used FP16 KV and a 16,384-token context for these four fixtures. Each number is the median completion time of four requests, all of which passed their fixture checks. Output lengths and prose wording can differ.

| Fixture | Prompt tokens | Ada | Strata | Strata + DSpark |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic | 30 | 0.419 s | 0.595 s | 0.332 s |
| Prose | 52 | 2.298 s | 3.376 s | 2.072 s |
| Inventory retrieval | 2,159 | 1.381 s | 1.981 s | 1.197 s |
| Inventory retrieval | 8,459 | 2.791 s | 3.880 s | 3.142 s |

Strata with DSpark led on arithmetic, prose, and the 2,159-token lookup. Ada led at 8,459 tokens. The stacks use different weight packs and can generate different numbers of tokens. [Completed-answer data](results/rtx4090-strata-completed.json)

### Longer prompts

Run the six-point retrieval sweep on each engine using its served model ID and the same run number. For both Ada builds use `--max-context 163840` in the direct server command above. Record the effective server cap for each engine. The largest input needs about 150,060 tokens plus room for 1,024 output tokens.

```bash
python3 benchmark_completed.py 8080 1 --model-id inkling-2-mini --context-sweep > context-run-1.jsonl
```

The client warms the server with a READY request, then sends inventories of 105, 420, 780, 1,580, 3,200 and 7,500 rows. These prompts differ from the historical table below because each size has a distinct opening to limit prefix reuse. Check cached tokens when comparing cold latency. Each JSON line gives the target length, validation result and any request error. Completed requests also include the server-reported prompt length, answer, cache usage and timings when supplied. Validation requires the correct two integers, a normal stop and a reported length within 32 tokens of the estimate calibrated on the historical prompts (20 tokens per row plus 59). A failed warmup emits six failed records without sending retrieval requests. A failed retrieval still allows later points to run. Either failure makes the command exit nonzero. The socket timeout is 600 seconds.

![Paired RTX 4090 latency bars. Ada and Strata complete the 15.7K prompt in 4.71 and 15.91 seconds, and the 64K prompt in 26.02 and 86.23 seconds.](assets/rtx4090-context-benchmark.png)

The longer context sweep ran one correct completed retrieval request per point. Ada used UD-IQ4_XS, FP16 KV, and a configured 65,536-token context. Strata used SQ-20, FP8 KV, 64-token chunks, and no drafting after DSpark exhausted VRAM in the large configuration. Its CUDA configuration was automatically capped at 286,912 tokens.

| Prompt tokens | Ada completion | Strata completion |
| ---: | ---: | ---: |
| 15,659 | 4.706 s | 15.911 s |
| 31,659 | 10.218 s | 35.011 s |
| 64,059 | 26.024 s | 86.232 s |

At a verified 307,459-token prompt, Ada allocated a 327,680-token MXFP8 KV cache and prefilled about 141,000 tokens before the 152-second cutoff. The recorded first attempt exited after 59 seconds because of a harness error. Strata was limited by its 286,912-token KV cap. The 307K experiment recorded partial prefill and a capacity limit. [Context sweep data](results/rtx4090-context-sweep.json)

In a separate matched-weights experiment, the tensor QK patch reduced median completion time from 3.010 to 2.796 seconds on the 8,459-token fixture. [Experiment and numerical checks](docs/sm89-tensor-qk.md)

The [original llama.cpp baseline](docs/llama-cpp-baseline.md) measures a synthetic 128-token output cap. Its measured requests reached that cap during reasoning.

### 500K llama.cpp context trial

In a separate November 21, 2026 run on an RTX 4090, [Unsloth llama.cpp b12588](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8) loaded a 524,288-token context and accepted one synthetic 500,014-token retrieval prompt. The run used UD-IQ4_XS weights, a Q4_0 KV cache, a 32-token microbatch, all model layers on the GPU, and speculative decoding disabled.

| Prompt tokens processed | Prefill elapsed | Cumulative prefill |
| ---: | ---: | ---: |
| 100,364 | 191.43 s | 524.30 tok/s |
| 200,716 | 558.34 s | 359.49 tok/s |
| 301,068 | 1,110.32 s | 271.16 tok/s |
| 356,364 | 1,495.86 s | 238.23 tok/s |

The request timed out at 1,500 seconds before its first answer. The table shows partial prefill measurements. Full-prompt latency, retrieval accuracy, and decode throughput remain unmeasured at this length.

## Contributing

Independent benchmark runs, correctness tests, focused SM89 improvements, and build documentation are welcome. Open an [issue](https://github.com/Fleetborn/inkling-ada/issues/new) describing the hardware, model, settings, and observed result before starting a larger change. For a small fix, open a pull request with the relevant check or measurement. Preserve raw results and explain differences in weights, context, cache, and decoding settings when comparing runs.

Inkling Ada uses the [Apache 2.0 license](LICENSE). The [logo](assets/inkling-ada-logo.png) and [share image](assets/inkling-ada-banner.jpg) are in the repository.

## Acknowledgments

NInfer-all provides the foundation for this work. [Basalt](https://github.com/jesdga95/basalt/blob/main/docs/models/inkling2.md) and [Strata](https://github.com/Niko1221/Strata) provide technical inspiration for Inkling attention, cache design, speculative decoding, and expert placement.
