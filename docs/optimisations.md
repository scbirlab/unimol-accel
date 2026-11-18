# How it works

## What the profile showed

Uni-Mol is a pair-bias transformer: the pre-softmax attention logits `(B·H, L, L)` are carried from layer to layer as
the next layer's bias, with `head_dim = 8`. Three things stood out:

1. **Padding.** Batches are padded to the longest molecule, and the pair tensors grow with `L²`.
2. **An unused tail.** The encoder computes `delta_pair_repr` and norm losses that `UniMolModel.forward` discards.
3. **Extra passes over the logits** (`bmm` then in-place add, plus a defensive clone before softmax).

## The three patches

| Patch | What changes | Exact? |
| --- | --- | --- |
| `UniMolModel.forward` | molecules are grouped by true token length with an optimal DP partition, `cost(group) = n(L + L²/16) + 48·L` (minimum group 4, only if the plan saves ≥ 10 % of padded cost); each group runs the original forward; outputs return in caller order | yes (per-sample maths is independent: LayerNorm only) |
| `TransformerEncoderWithPair.forward` | stops after the final layer norm while a context flag is set by the wrapper above | yes |
| `SelfMultiheadAttention.forward` | `baddbmm(bias, q, kᵀ)` replaces `bmm` + `+=`; no clone before softmax | bit-identical in fp32 |

## Results (2-vCPU Intel Xeon @ 2.3 GHz VM, PyTorch 2.15, random weights; medians of interleaved runs)

**Same batch size (32), original vs accel:** inference 1778 → 865 ms (**2.06×**); training step 6790 → 3980 ms (**1.71×**).

**But the original is already fast at small batches** (ms per molecule, inference, 48 molecules):

| batch size | 4 | 8 | 16 | 32 | 48 |
|---|---:|---:|---:|---:|---:|
| original | 36 | 37 | 41 | 52 | 75 |
| unimol-accel | 36 | 36 | 30 | 33 | 28 |

Best-versus-best is **≈1.3×** (36 → 28 ms/molecule); users already running small batches gain ≈1.0–1.1×. Across random
batch compositions (B = 8–48) the planner never ran slower than the original (min 0.98×, where it chose not to split)
and gained 1.3–2.8× on ragged batches. Training bucketing is on by default (1.57–1.71× at batch 32); gradients match
(tested with dropout off) but dropout masks are drawn per sub-batch.

Lean attention plus tail skip alone are within noise end to end (0.98–1.04×); the gain is bucketing.

## Negative results (kept as evidence)

- **Fused C++ pair-bias attention** (forward, backward, hash-based dropout): **slower than eager PyTorch** (0.2–0.7× for
  inference) even after restructuring into 4-row blocks with broadcast FMA and templated head sizes. `head_dim = 8` is
  too small to beat MKL batched GEMM; GCC `target_clones` dispatch was much worse than `-march=native`. Source kept in
  `experiments/native_pair_attention/`; not shipped.
- **Conformer generation** (RDKit ETKDG + MMFF) is a large fixed per-molecule cost that this package does not touch.

## Not measured

Many-core CPUs, CUDA, and real pretrained weights (Hugging Face was not reachable from the development sandbox).
