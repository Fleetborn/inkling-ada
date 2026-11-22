# Original llama.cpp and NInfer benchmark

This experiment compared stock NInfer-all with llama.cpp on an RTX 4090 using capped output. See the [project README](../README.md) for setup and completed-answer measurements.


Use the [Unsloth mainline UD-IQ4_XS GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/Mainline_Support) and [llama.cpp b12613](https://github.com/ggml-org/llama.cpp/releases/tag/b12613) for the baseline. Save the `.gguf_file` artifact locally with the `.gguf` extension. Run each server in turn on an otherwise idle GPU.

```bash
llama-server -m Inkling-2-Mini-UD-IQ4_XS.gguf -a inkling-2-mini \
  --host 127.0.0.1 --port 8080 -c 16384 -ngl 99 -np 1 -b 2048 -ub 512 \
  -fa auto -ctk f16 -ctv f16 --fit off --jinja
python3 benchmark.py http://127.0.0.1:8080 inkling-2-mini
```

The benchmark sends a READY smoke prompt and runs three single-stream requests at each of two prompt sizes. Each measured request has a 128-token output cap and a unique prefix. Output is JSON Lines with prompt hashes, token counts, cache counts when available, engine timing when available, and client latency. First streamed token includes reasoning. First visible answer can be null if the output cap is reached during reasoning.

## RTX 4090 results

Measured on the benchmark host on November 22, 2026. Rates and latencies are medians of three requests per engine and prompt length. Peak VRAM is the highest sample across both lengths, taken every 500 ms. [Per-request data](../results/rtx4090-2026-11-22.json)

| Engine | Prompt tokens | Prefill tok/s | Decode tok/s | First token | Total time | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp | 2,143 | 1,976 | 72.1 | 1.15 s | 2.88 s | 17.2 GiB |
| NInfer | 2,141 | 4,284 | 130.3 | 0.52 s | 1.48 s | 17.7 GiB |
| NInfer + DSpark | 2,141 | 4,250 | 112.6 | 0.53 s | 1.63 s | 18.4 GiB |
| llama.cpp | 8,443 | 1,770 | 69.8 | 4.88 s | 6.66 s | 17.2 GiB |
| NInfer | 8,441 | 4,067 | 125.3 | 2.09 s | 3.09 s | 17.7 GiB |
| NInfer + DSpark | 8,441 | 4,001 | 105.7 | 2.13 s | 3.31 s | 18.4 GiB |

Plain NInfer decoded 1.8× as fast and processed prompts 2.2–2.3× as fast as llama.cpp. DSpark accepted 19% of drafted tokens and reduced decode throughput by 14–16% versus plain NInfer on this workload. That result sets the default to `none`.

All three configurations returned READY on the smoke prompt. All 18 measured requests reached the 128-token cap during reasoning. The results establish throughput for this synthetic workload. See the [completed-answer experiments](../README.md#rtx-4090-benchmarks) for retrieval correctness and full answer latency.

Request text and UD-IQ4_XS weight blocks matched across engines. Their template handling differed by two prompt tokens, and llama.cpp reused 12 shared prefix tokens per request while NInfer reported zero reuse. NInfer ran in a Debian 13 container using the CUDA 12.9 release. llama.cpp ran natively with CUDA 12.8. The table uses each engine’s reported prefill and decode rates, and the same client’s streamed first-token and total time.
