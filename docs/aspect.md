# Use with Aspect

[scbirlab/aspect](https://github.com/scbirlab/aspect) featurises tables through registered transforms; its `unimol`
transform (`aspect-data[unimol]`) loads Uni-Mol weights from the Hugging Face Hub and embeds batches in-process.
unimol_accel patches the upstream classes Aspect uses, so **no change to Aspect is needed**.

## Activating it

```bash
pip install unimol-accel "aspect-data[unimol]"
python -m unimol_accel -m aspect.cli_module.cli featurize compounds.csv \
    --features "smiles:conformers:unimol@um" --output features.parquet
python -m unimol_accel --also atom_jepa_accel script.py            # both accel packages
python -m unimol_accel -m pytest path/to/aspect/test        # Aspect's own tests with accel on
```

or, from Python, `import unimol_accel` before the featuriser runs. Patches act on classes, so the import may come before or after `unimol_tools`.

## Input modes

Aspect's `unimol` transform has two input modes and both end in the patched `UniMolModel.forward`:

| Aspect setting | Input to `UniMolRepr.get_repr` | Conformers come from |
| --- | --- | --- |
| `conformers="aspect"` (default), chain `("smiles", ["conformers", "unimol"])` | `{"atoms": [...], "coordinates": [...]}` | Aspect's `conformers` step (ETKDG + MMFF, cacheable and shareable) |
| `conformers="unimol"`, `("smiles", "unimol")` | list of SMILES | Uni-Mol's own RDKit step (one conformer, seed 42, 2D fallback) |

Aspect flushes `batch_size` conformers at a time, so bucketing acts **within** each batch of `batch_size` conformers.

## Choosing `batch_size`

The original code is already fast at small batches, and accel mainly removes the padding penalty of big ones
(ms per molecule through Aspect, pre-generated conformers, 48 molecules, 2-vCPU VM):

| `batch_size` | 4 | 8 | 32 | 48 |
|---|---:|---:|---:|---:|
| original | 43 | 42 | 54 | 76 |
| with unimol-accel | 42 | 39 | 31 | 29 |

So the default `batch_size=32` gains ≈1.75×, but best-versus-best is ≈1.45× (42 → 29 ms/molecule); at batch 4–8 expect
almost nothing. Conformer generation (≈27 ms/molecule here) is not accelerated and is comparable to the embedding
cost; `conformers(n_threads=…)` or reusing cached conformers is the next lever. Measure on your machine with
`python benchmarks/bench_aspect.py`.

## What was and was not verified

- All integration tests (`tests/integration/`) run the real Aspect code with **random weights** saved in the real
  on-disk format, with only the Hugging Face download redirected to a temporary directory. Pretrained weights could
  not be downloaded in the environment where this was developed; the arithmetic is the same, but please also run
  Aspect's own tests on a machine where the weights are cached.
- An end-to-end `aspect featurize` CLI run (CSV in, Parquet out) with and without the runner agreed: Uni-Mol features identical (rel. error 0.0) on a 12-molecule table.
- Aspect keys its feature cache on its own settings; accelerated outputs agree with the original to fp32 rounding (identical in the CLI test above), so cached Uni-Mol features stay valid.
