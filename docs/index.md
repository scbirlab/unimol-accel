---
html_theme.sidebar_secondary.remove: true
---

# unimol-accel

**Drop-in CPU acceleration for [Uni-Mol](https://github.com/deepmodeling/Uni-Mol).**

```python
import unimol_accel  # that's it
import unimol_tools
```

Pure-PyTorch patches give length-bucketed sub-batching inside `UniMolModel.forward`, skip the encoder tail that is computed and discarded, and use a leaner pair-bias attention. No change to weights, checkpoints or `state_dict` keys; it also runs under
[scbirlab/aspect](https://github.com/scbirlab/aspect) without editing Aspect (`python -m unimol_accel -m aspect.cli_module.cli ...`).

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Install
:link: installation
:link-type: doc
pip, versions, and what is verified.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc
One import, `status()`, `disable()`, and an equivalence check.
:::

:::{grid-item-card} Use with Aspect
:link: aspect
:link-type: doc
SMILES vs pre-generated conformers, the CLI runner, caching notes.
:::

:::{grid-item-card} How it works
:link: optimisations
:link-type: doc
What was profiled, what changed, what did *not* help, with numbers.
:::

:::{grid-item-card} Benchmarks
:link: benchmarks/index
:link-type: doc
Run it yourself, safely, offline. Report from a 2-vCPU sandbox.
:::

:::{grid-item-card} Compatibility
:link: compatibility
:link-type: doc
Verified versions, hash guard, how to add a release.
:::
::::

```{admonition} All timings come from one small machine
:class: warning
Published numbers are from a shared 2-vCPU VM. They are not a forecast for your hardware: measure with
`python -m unimol_accel.bench` before relying on them.
```

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
aspect
optimisations
benchmarks/index
compatibility
api
changelog
contributing
```
