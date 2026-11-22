# Ada full-attention prefill experiment

On bm04, Nsight Systems measured 977 ms in Inkling’s prompt attention across 8,448 input tokens, or 50% of summed GPU kernel time. The 40 full-context attention calls accounted for 436 ms, or 22.3% of summed GPU kernel time. The profile uses NInfer-all `18029c8e0ce2555bb499e04be24d7a4e9c12a5bc`, FP16 KV, GPU-resident experts, and one sequence on the RTX 4090. [Baseline trace summary](../results/bm04-sm89-baseline-profile.json)

Basalt stages each K/V tile once for a wider group of query rows. The first Ada experiment applies that reuse pattern to NInfer’s existing scalar attention math. It increases the prompt tile from 8 to 16 query rows per block on SM89 for a contiguous, single-sequence full-attention chunk with FP16 K/V. [Basalt prompt attention](https://github.com/jesdga95/basalt/blob/0dc981c8c9593e0d7cd0b9059aa74cfa4545814d/src/models/inkling2/kernels/attn_prompt.cu)

1. Compare output bytes against the existing 8-row route across full-attention tile boundaries, nonzero prefixes, and partial final tiles. Check both BF16 and packed Q8 outputs used by the engine.
2. Capture matching prefill traces and compare the full-attention kernel duration and total prefill time. Inspect compile reports if the 512-thread tile lowers occupancy or spills.
3. Compare identical completed prompts on baseline and candidate builds from the same source revision and compiler. Require normal stops and matching answer text and token counts before comparing first visible answer and completion latency.

## Result

Keep the 8-row route as the default. Across 40 matching full-attention calls, the wider tile took 471.2 ms versus 449.2 ms for the baseline, a 4.9% regression. It used 512 threads and 90 registers per thread, versus 256 threads and 96 registers for the baseline. Neither kernel spilled registers. The 8,459-token answer took 0.54% to 1.33% longer in every paired run. [Per-request results and trace summaries](../results/bm04-sm89-full-prefill.json)

| Fixture | Input tokens | Baseline median | 16-row median |
| --- | ---: | ---: | ---: |
| Arithmetic | 30 | 0.419 s | 0.419 s |
| Prose | 52 | 2.256 s | 2.256 s |
| Inventory retrieval | 2,159 | 1.390 s | 1.393 s |
| Inventory retrieval | 8,459 | 3.005 s | 3.034 s |

All 32 requests stopped normally and passed their fixture checks. All 16 pairs matched answer text and token counts. The CUDA kernel test passed the independent oracle and eight byte comparisons against the 8-row route for both BF16 and packed Q8 output. The Nsight trace identified 40 candidate calls with 512-thread blocks. The comparison used four repetitions per fixture in baseline–candidate–candidate–baseline order, single-stream FP16 KV, the same quant and toolchain, and no speculation. The recorded hashes identify the patch and benchmark client used for that measurement.

The next candidate is Basalt’s tensor-core query–key product adapted to Ada’s `mma.sync`. The trace locates enough time in prompt attention to justify testing that arithmetic directly. Start with FP16-KV full attention, test FP16 query rounding against the existing oracle, and keep learned relative bias and softmax accumulation in FP32. [Basalt kernel source](https://github.com/jesdga95/basalt/blob/0dc981c8c9593e0d7cd0b9059aa74cfa4545814d/src/models/inkling2/kernels/attn_prompt.cu)

## Reproduce the comparison

Build the pinned NInfer source twice with the same compiler. Use the clean pinned revision for the baseline and apply the [16-row patch](../patches/ninfer-sm89-full-prefill.patch) directly to the same clean revision for its candidate. Each experiment starts from clean upstream. Use the NInfer release guide to install the source build dependencies and fetch its pinned media source archives. [NInfer build documentation](https://github.com/iamwavecut/ninfer-all/blob/18029c8e0ce2555bb499e04be24d7a4e9c12a5bc/CONTRIBUTING.md)

```bash
cmake -S /path/to/ninfer -B /path/to/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
cmake --build /path/to/build --target ninfer-serve ninfer_bench ninfer_tests
ctest --test-dir /path/to/build -R '^ninfer_inkling2_kernels_test$' --output-on-failure
```

Run the GPU test on SM89 to exercise the alternate route. The paired BF16 and Q8 comparisons run only on an eligible SM89 device.

Use `serve.sh` with each build and the same UD-IQ4_XS artifact. Run `python3 benchmark_completed.py 8080 RUN` with the same run number for each pair. The suite uses arithmetic, prose, and two inventory retrieval prompts at roughly 2K and 8K input tokens. It records actual output length, validation results, answer hashes, first visible answer time, and total client latency. Every fixture must stop normally and meet its check for the command to exit successfully.

To attribute time to kernels, capture a warm prefill repetition with Nsight Systems. Run in the NInfer source directory with the appropriate artifact and executable paths.

```bash
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop -o pp8448 \
  /path/to/ninfer_bench --weights /path/to/model.ninfer \
  --corpus bench/fixtures/bench_corpus.ids --kv-dtype fp16 --expert-residency device \
  --max-ctx 16384 --warmup 1 -r 1 --profile-measured -p 8448
```
