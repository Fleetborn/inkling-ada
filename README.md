# Inkling Ada

Serve Thinking Machines’ [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single RTX 4090 and measure changes against a llama.cpp baseline. The first version uses NInfer-all for CUDA execution, model loading, and the OpenAI-compatible API.

## Run the server

Use a supported NVIDIA driver and the CUDA 12.9 Linux archive from [NInfer-all 2026-11-22.3](https://github.com/iamwavecut/ninfer-all/releases/tag/2026-11-22.3). Verify the numbered archive parts against its SHA-256 manifest. The archive targets Debian 13. On older Linux hosts, use the archive’s Dockerfile and NVIDIA Container Toolkit as described in the [NInfer Linux release guide](https://github.com/iamwavecut/ninfer-all/blob/master/docs/release-archive-linux.md).

Download the [UD-IQ4_XS NInfer artifact](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3). It retains the upstream Unsloth GGUF’s quantized weight blocks and includes the DSpark drafter.

```bash
bash serve.sh /path/to/ninfer-serve /path/to/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
python3 benchmark.py http://127.0.0.1:8080 inkling-2-mini
```

The profile keeps experts on the GPU and uses a 16,384-token context, FP16 KV, and one active sequence. Pass the port followed by `dspark` to enable speculative decoding. For example, use `8080 dspark` as the final two arguments to `serve.sh`. The server binds to loopback.

## Compare engines

Use the [Unsloth mainline UD-IQ4_XS GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/Mainline_Support) and [llama.cpp b12613](https://github.com/ggml-org/llama.cpp/releases/tag/b12613) for the baseline. Save the `.gguf_file` artifact locally with the `.gguf` extension. Run each server in turn on an otherwise idle GPU.

```bash
llama-server -m Inkling-2-Mini-UD-IQ4_XS.gguf -a inkling-2-mini \
  --host 127.0.0.1 --port 8080 -c 16384 -ngl 99 -np 1 -b 2048 -ub 512 \
  -fa auto -ctk f16 -ctv f16 --fit off --jinja
python3 benchmark.py http://127.0.0.1:8080 inkling-2-mini
```

The benchmark sends a READY smoke prompt and runs three single-stream requests at each of two prompt sizes. Each measured request has a 128-token output cap and a unique prefix. Output is JSON Lines with prompt hashes, token counts, cache counts when available, engine timing when available, and client latency. First streamed token includes reasoning. First visible answer can be null if the output cap is reached during reasoning.

## RTX 4090 results

Measured on bm04 on November 22, 2026. Rates and latencies are medians of three requests per engine and prompt length. Peak VRAM is the highest sample across both lengths, taken every 500 ms. [Per-request data](results/bm04-2026-11-22.json)

| Engine | Prompt tokens | Prefill tok/s | Decode tok/s | First token | Total time | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp | 2,143 | 1,976 | 72.1 | 1.15 s | 2.88 s | 17.2 GiB |
| NInfer | 2,141 | 4,284 | 130.3 | 0.52 s | 1.48 s | 17.7 GiB |
| NInfer + DSpark | 2,141 | 4,250 | 112.6 | 0.53 s | 1.63 s | 18.4 GiB |
| llama.cpp | 8,443 | 1,770 | 69.8 | 4.88 s | 6.66 s | 17.2 GiB |
| NInfer | 8,441 | 4,067 | 125.3 | 2.09 s | 3.09 s | 17.7 GiB |
| NInfer + DSpark | 8,441 | 4,001 | 105.7 | 2.13 s | 3.31 s | 18.4 GiB |

Plain NInfer decoded 1.8× as fast and processed prompts 2.2–2.3× as fast as llama.cpp. DSpark accepted 19% of drafted tokens and reduced decode throughput by 14–16% versus plain NInfer on this workload. That result sets the default to `none`.

All three configurations returned READY on the smoke prompt. All 18 measured requests reached the 128-token cap during reasoning. The results establish throughput for this synthetic workload. Completed-answer latency, answer quality, long-context retrieval, vision, and concurrency remain unmeasured.

Request text and UD-IQ4_XS weight blocks matched across engines. Their template handling differed by two prompt tokens, and llama.cpp reused 12 shared prefix tokens per request while NInfer reported zero reuse. NInfer ran in a Debian 13 container using the CUDA 12.9 release. llama.cpp ran natively with CUDA 12.8. The table uses each engine’s reported prefill and decode rates, and the same client’s streamed first-token and total time.

## Design references

[Ada prefill experiment](docs/sm89-prefill-attention.md) profiles Inkling on bm04 and compares a wider full-attention tile on completed answers. The results favor the existing 8-row route and identify the tensor-core score product as the next candidate.

Inkling evaluates 24 weight layers in 48 visits. The 12 middle layers run three times and share weights, with separate KV and short-convolution state per visit. Keeping the quantized experts on the GPU removes expert transfers from the short-context benchmark. Quantized KV and expert paging are candidates for longer contexts once measurements show which resource is limiting. [Inkling architecture in vLLM](https://vllm-project.github.io/2026/11/20/inkling2-mini.html)

[Strata](https://github.com/Niko1221/Strata) provides a reference for host and GPU expert placement. [Basalt](https://github.com/jesdga95/basalt/blob/main/docs/models/inkling2.md) provides a reference for attention with learned relative bias, per-visit cache sizing, and adaptive DSpark windows. Basalt’s implementation targets Blackwell. Any Ada kernel work should first be compared with the existing NInfer path on the same weights and prompts.
