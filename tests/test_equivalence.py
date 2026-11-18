import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import pytest
import torch

pytest.importorskip("unimol_tools")
import unimol_accel as ua
from unimol_accel import _patches as P
from _um_common import make_model, make_samples


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    m = make_model().eval()
    batch, _ = m.batch_collate_fn(make_samples(16))
    return m, batch


def cls_of(out):
    """cls_repr is a bare tensor in unimol_tools>=0.1.6 and a dict entry in older versions."""
    return out["cls_repr"] if isinstance(out, dict) else out


def run(m, batch, **kw):
    with torch.no_grad():
        return m(**batch, **kw)


def test_status_and_idempotent(setup):
    ua.enable()
    s1 = ua.status()
    ua.enable()
    s2 = ua.status()
    assert s1 == s2 and all(v == "active" for v in s1["patches"].values())


def test_disable_restores_original(setup):
    ua.disable()
    assert all(v.startswith("inactive") for v in ua.status()["patches"].values())
    from unimol_tools.models.transformers import SelfMultiheadAttention

    assert not hasattr(SelfMultiheadAttention.forward, "_unimol_accel")
    ua.enable()


def test_inference_equal(setup):
    m, b = setup
    ua.disable()
    ref = run(m, b)
    ref_r = run(m, b, return_repr=True)
    ua.enable()
    new = run(m, b)
    new_r = run(m, b, return_repr=True)
    torch.testing.assert_close(new, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cls_of(new_r), cls_of(ref_r), atol=1e-5, rtol=1e-5)


def test_atomic_reprs_order_and_values(setup):
    m, b = setup
    ua.disable()
    ref = run(m, b, return_repr=True, return_atomic_reprs=True)
    ua.enable()
    new = run(m, b, return_repr=True, return_atomic_reprs=True)
    assert all(a == c for a, c in zip(new["atomic_symbol"], ref["atomic_symbol"]))
    for a, c in zip(new["atomic_reprs"], ref["atomic_reprs"]):
        torch.testing.assert_close(a, c, atol=1e-5, rtol=1e-5)


def test_gradients_equal_no_dropout(setup):
    m, b = setup
    m.train()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    for mod in m.modules():
        if hasattr(mod, "dropout") and isinstance(mod.dropout, float):
            mod.dropout = 0.0
        if hasattr(mod, "emb_dropout"):
            mod.emb_dropout = 0.0
    grads = []
    for flag in (False, True):
        (ua.enable if flag else ua.disable)()
        m.zero_grad()
        out = m(**b)
        out.float().pow(2).sum().backward()
        grads.append([p.grad.clone() for p in m.parameters() if p.grad is not None])
    ua.enable()
    m.eval()
    assert len(grads[0]) == len(grads[1])
    for a, c in zip(*grads):
        torch.testing.assert_close(a, c, atol=1e-4, rtol=1e-3)


def test_gradients_equal_with_bucketing_in_training(setup, monkeypatch):
    # bucketing is active in train mode by default; with dropout off gradients must match the original
    test_gradients_equal_no_dropout(setup)
    m, b = setup
    ua.enable()
    m.train()
    m.zero_grad()
    m(**b).float().pow(2).sum().backward()
    m.eval()


def test_state_dict_keys_unchanged(setup):
    m, _ = setup
    ua.disable()
    k0 = list(m.state_dict())
    ua.enable()
    k1 = list(m.state_dict())
    assert k0 == k1


def test_bucketing_disabled_env(setup, monkeypatch):
    m, b = setup
    monkeypatch.setenv("UNIMOL_ACCEL_BUCKET", "0")
    ua.disable()
    ref = run(m, b)
    ua.enable()
    torch.testing.assert_close(run(m, b), ref, atol=1e-5, rtol=1e-5)


def test_plan_buckets_partition_and_cost():
    lengths = [10, 12, 80, 11, 79, 40, 41, 13, 12, 78, 11, 10]
    groups = P.plan_buckets(lengths)
    flat = sorted(i for g in groups for i in g)
    assert flat == list(range(len(lengths)))
    assert len(groups) >= 2


def test_unknown_source_hash_keeps_original(monkeypatch):
    monkeypatch.setitem(P.KNOWN_HASHES, "UniMolModel.forward", ["deadbeef"])
    ua.disable()
    for p in ua._active:
        if p.name == "UniMolModel.forward":
            assert p.apply() is False and "unverified" in p.reason
    monkeypatch.undo()
    ua.enable()


def test_export_and_tracing_use_original_forward(monkeypatch):
    """Bucketing reads tensor values, which a tracer would freeze; exporters must get upstream's code."""
    from unimol_accel import _patches

    assert not _patches._exporting()
    monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
    assert _patches._exporting()
    monkeypatch.undo()
    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
    assert _patches._exporting()
