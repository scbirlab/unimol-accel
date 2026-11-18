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
| `UniMolModel.forward` | molecules are grouped by true token length with an optimal DP partition, `cost(group) = n(L + L²/16) + 48·L` (minimum group 4, only if the plan saves ≥ 10 % of padded cost); each group runs the original forward; outputs return in caller order. **Inference only by default**: training is bucketed only with `UNIMOL_ACCEL_BUCKET_TRAIN=1` (dropout masks would otherwise be drawn per sub-batch) | yes (per-sample maths is independent: LayerNorm only) |
| `TransformerEncoderWithPair.forward` | stops after the final layer norm while a context flag is set by the wrapper above | yes |
| `SelfMultiheadAttention.forward` | `baddbmm(bias, q, kᵀ)` replaces `bmm` + `+=`; no clone before softmax | bit-identical in fp32 |

### The cost model is a heuristic

`plan_buckets` is exactly optimal (checked by exhaustive search in the tests) for `n·(L + pair_weight·L²) + group_overhead·L`,
not for measured time. A split is then accepted only if that same full model (every group paying its overhead) predicts a
saving of at least `min_saving` (10 %) against running the batch as one padded group. An earlier version compared the
plan *without* the per-group overhead, which made weak splits look better than the planner's own objective said they were
(a 16-molecule case was seen as a 27 % saving; the full model gives 6 %, so it now runs unsplit; this is a unit test). All four constants are tunable by environment variable (`UNIMOL_ACCEL_COST_PAIR`, `_GROUP`,
`_MIN_GROUP`, `_MIN_SAVING`), `status()["bucketing"]` shows what is in force, and `benchmarks/calibrate_cost_model.py`
fits them on the current machine. On the 2-vCPU test VM: `t ≈ 1.8e-4·n·L + 9.4e-6·n·L²` (R² 0.99), i.e. pair weight
≈0.053 against the default 0.0625; per-group overhead measured small (a split of 32 molecules into 4 groups cost +7, +4 and
−13 units of `a·L` at L = 24, 48, 80: noise and cache effects dominate). `benchmarks/cost_model_sensitivity.py`
(batch 32, medians of interleaved runs) shows how much that matters:

Two runs each (first / second), speed-up over the original with the number of planned groups:

| length distribution | defaults (1/16, 48) | fit-like (0.056, 0) | group overhead 150 |
|---|---:|---:|---:|
| uniform 12–80 | 1.64× / 1.52× (2) | 1.58× / 1.74× (8) | 1.22× / 1.21× (1) |
| bimodal 14–24 / 60–80 | 2.03× / 2.06× (2) | 2.04× / 1.73× (8) | 2.16× / 2.07× (2) |
| long tail 12–118 | 3.99× / 4.35× (3) | 4.66× / 4.93× (7) | 2.39× / 2.74× (2) |
| narrow 28–36 | 1.16× / 0.88× (1) | 1.06× / 0.74× (7) | 1.16× / 0.88× (1) |

The second run used the corrected acceptance rule. Run-to-run differences of ±20 % (see the narrow row, where the
default runs the original path in both) are as large as most differences between settings, so only the large effects are
real here: over-splitting narrow batches (zero overhead), and under-splitting long-tailed ones (overhead 150: roughly
4× → 2.5×). The defaults are never far from the best, which is all that can be claimed from one noisy 2-vCPU machine. The earlier
statement that a random-composition sweep never ran slower than the original (min 0.98×) was measured before the
acceptance-rule correction and has not been re-run. Calibrate on yours.

## Results (2-vCPU Intel Xeon @ 2.3 GHz VM, PyTorch 2.15, random weights; medians of interleaved runs)

**Same batch size (32), original vs accel:** inference 1778 → 865 ms (**2.06×**); training step 6790 → 3980 ms (**1.71×**) with opt-in training bucketing (≈1.0–1.1× by default, where training is not bucketed).

**But the original is already fast at small batches** (ms per molecule, inference, 48 molecules):

| batch size | 4 | 8 | 16 | 32 | 48 |
|---|---:|---:|---:|---:|---:|
| original | 36 | 37 | 41 | 52 | 75 |
| unimol-accel | 36 | 36 | 30 | 33 | 28 |

Best-versus-best is **≈1.3×** (36 → 28 ms/molecule); users already running small batches gain ≈1.0–1.1×. Across random
batch compositions (B = 8–48) the planner never ran slower than the original (min 0.98×, where it chose not to split)
and gained 1.3–2.8× on ragged batches. Training bucketing is **opt-in** (`UNIMOL_ACCEL_BUCKET_TRAIN=1`; 1.57–1.79× at batch 32): gradients match
(tested with dropout off, and that the batch really is split) but dropout masks are drawn per sub-batch, so trajectories
differ from the original; by default training is not bucketed (≈1.0–1.1×).

Lean attention plus tail skip alone are within noise end to end (0.98–1.04×); the gain is bucketing.

## Negative results (kept as evidence)

- **Fused C++ pair-bias attention** (forward, backward, hash-based dropout): **slower than eager PyTorch** (0.2–0.7× for
  inference) even after restructuring into 4-row blocks with broadcast FMA and templated head sizes. `head_dim = 8` is
  too small to beat MKL batched GEMM; GCC `target_clones` dispatch was much worse than `-march=native`. Source kept in
  `experiments/native_pair_attention/`; not shipped.
- **Conformer generation** (RDKit ETKDG + MMFF) is a large fixed per-molecule cost that this package does not touch.

## Not measured

Many-core CPUs, CUDA, and real pretrained weights (Hugging Face was not reachable from the development sandbox).
