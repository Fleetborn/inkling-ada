# Ada tensor-core query–key scores

Prompt attention used 50% of measured GPU kernel time in the 8,448-token Inkling trace on bm04. This experiment replaces the scalar query–key product inside the existing eight-query block for SM89 full-attention FP16-KV prefill. Each of two warps computes up to 16 flattened query-head rows against 16 keys using CUDA WMMA, whose half-precision inputs accumulate into FP32 scores. [Baseline profile](../results/bm04-sm89-baseline-profile.json) and [Basalt prompt attention](https://github.com/jesdga95/basalt/blob/0dc981c8c9593e0d7cd0b9059aa74cfa4545814d/src/models/inkling2/kernels/attn_prompt.cu)

Round scaled queries to FP16 once per block. A query outside finite FP16 range selects the scalar score path. The existing code applies the learned relative bias, causal mask, and online softmax in FP32. The existing FP32 probability–value accumulation, sink, and gate consume the result. Query rounding and a different reduction order can change a close greedy decision. [CUDA WMMA](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-c-programming-guide/index.html#warp-matrix-functions)

1. Apply the patch directly to NInfer `18029c8e0ce2555bb499e04be24d7a4e9c12a5bc`. Verify CUDA compilation and the existing attention oracle for BF16 and packed Q8 output, including biased full attention and incomplete tiles.
2. Capture matching 8,448-token warm prefill traces and confirm that the tensor-core route ran for full-attention visits.
3. Run the existing four completed-answer fixtures four times per build in baseline–candidate–candidate–baseline order on bm04. Compare normal stops, fixture checks, answer hashes, output lengths, first visible answer, and completion latency.

## Measured result

On bm04’s RTX 4090, 40 full-attention calls took **232.4 ms versus 447.5 ms** on the 8-row scalar build, a 48.1% reduction. The trace selected the tensor-core specialization for all 40 calls. The SM89 binary contains `HMMA.16816.F32` for that kernel. At 8,459 input tokens, median prompt throughput rose from 4,094 to 4,611 tok/s and time to the first visible answer fell from 2.882 to 2.669 s. [Per-request results, numerical checks, and matched traces](../results/bm04-sm89-tensor-qk.json)

| Fixture | Input tokens | Scalar QK median | Tensor-core QK median |
| --- | ---: | ---: | ---: |
| Arithmetic | 30 | 0.419 s | 0.420 s |
| Prose | 52 | 2.257 s | 2.298 s |
| Inventory retrieval | 2,159 | 1.391 s | 1.381 s |
| Inventory retrieval | 8,459 | 3.010 s | 2.796 s |

All 32 requests stopped normally and satisfied their fixture checks. Arithmetic and inventory answer text matched in all 12 pairs. The four prose answers differed in wording and output length. The existing GPU oracle passed for BF16 and packed Q8 output, including the new tests for an incomplete tile, a nonzero cache prefix, a call without relative bias, and the scalar fallback for a query outside FP16 range. The comparison covers single-stream UD-IQ4_XS, FP16 KV, four synthetic fixtures, and one GPU.

## Apply and reproduce

Build the scalar baseline from the pinned NInfer revision. Apply the [tensor QK patch](../patches/ninfer-sm89-tensor-qk.patch) to that clean revision for the candidate build, then follow the [same SM89 build and profiling commands](sm89-prefill-attention.md#reproduce-the-comparison). Run `benchmark_completed.py` with run numbers 1 and 2 on the baseline and then the candidate. Run numbers 3 and 4 on the candidate and then the baseline. Both builds use identical model weights and `serve.sh` settings with speculation disabled.
