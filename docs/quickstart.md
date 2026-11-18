# Quickstart

```python
import unimol_accel  # activates automatically (UNIMOL_ACCEL_DISABLE=1 to prevent)
import unimol_tools

print(unimol_accel.status())  # enabled?, versions, which patches are active and why not
unimol_accel.disable()  # restores the original functions (parameters untouched)
unimol_accel.enable()  # idempotent
```

## Check equivalence yourself

```python
import torch, unimol_accel as ua

# model: any UniMolModel / UniMolRepr().model; batch from model.batch_collate_fn
ua.disable()
ref = model(**batch, return_repr=True)
ua.enable()
new = model(**batch, return_repr=True)
```
`pytest tests` does this for logits, `cls_repr`, atomic representations and gradients, and `tests/integration` through Aspect.
