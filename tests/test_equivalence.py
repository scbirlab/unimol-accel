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


def _no_dropout(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "dropout") and isinstance(mod.dropout, float):
            mod.dropout = 0.0
        if hasattr(mod, "emb_dropout"):
            mod.emb_dropout = 0.0


def _grads(m, b, enabled):
    (ua.enable if enabled else ua.disable)()
    m.zero_grad()
    calls = []
    h = m.encoder.register_forward_hook(lambda mod, inp, out: calls.append(inp[0].shape[0]))
    try:
        m(**b).float().pow(2).sum().backward()
    finally:
        h.remove()
    return [p.grad.clone() for p in m.parameters() if p.grad is not None], calls


def _assert_grads_close(ref, got):
    assert len(ref) == len(got)
    for a, c in zip(ref, got):
        torch.testing.assert_close(c, a, atol=1e-4, rtol=1e-3)


def test_gradients_equal_no_dropout(setup):
    m, b = setup
    m.train()
    _no_dropout(m)
    ref, _ = _grads(m, b, False)
    got, _ = _grads(m, b, True)
    ua.enable()
    m.eval()
    _assert_grads_close(ref, got)


def test_training_is_not_bucketed_by_default(setup, monkeypatch):
    """Default: one encoder call per training batch, so dropout masks and trajectories match the original."""
    m, b = setup
    monkeypatch.delenv("UNIMOL_ACCEL_BUCKET_TRAIN", raising=False)
    m.train()
    _no_dropout(m)
    ref, _ = _grads(m, b, False)
    got, calls = _grads(m, b, True)
    ua.enable()
    m.eval()
    assert calls == [b["src_tokens"].shape[0]]  # a single, unsplit forward
    _assert_grads_close(ref, got)
    assert ua.status()["bucketing"]["training"] is False


def test_training_bucketing_opt_in_splits_and_matches_gradients(setup, monkeypatch):
    """Opt-in (UNIMOL_ACCEL_BUCKET_TRAIN=1): the batch really is split into sub-batches, gradients still match."""
    m, _ = setup
    # a deliberately ragged batch: the 8 shortest and 8 longest of 48 molecules
    samples = sorted(make_samples(48), key=lambda smp: len(smp[0]["src_tokens"]))
    b, _ = m.batch_collate_fn(samples[:8] + samples[-8:])
    lengths = (b["src_tokens"] != m.padding_idx).sum(1).tolist()
    cm = P.CostModel()
    assert P.worth_splitting(
        P.plan_buckets(lengths, cost=cm), lengths, len(lengths), max(lengths), cm
    ), "test batch must be ragged enough to be bucketed"
    monkeypatch.setenv("UNIMOL_ACCEL_BUCKET_TRAIN", "1")
    m.train()
    _no_dropout(m)
    ref, _ = _grads(m, b, False)
    got, calls = _grads(m, b, True)
    ua.enable()
    m.eval()
    assert len(calls) >= 2 and sum(calls) == b["src_tokens"].shape[0], calls
    _assert_grads_close(ref, got)
    assert ua.status()["bucketing"]["training"] is True


def test_seeded_dropout_training_matches_original_by_default(setup, monkeypatch):
    """Same seed + same weights: the default path draws the same dropout masks as the original."""
    m, b = setup
    monkeypatch.delenv("UNIMOL_ACCEL_BUCKET_TRAIN", raising=False)
    m.train()
    outs = []
    for enabled in (False, True):
        (ua.enable if enabled else ua.disable)()
        torch.manual_seed(1234)
        outs.append(m(**b).detach().float())
    ua.enable()
    m.eval()
    torch.testing.assert_close(outs[1], outs[0], atol=1e-4, rtol=1e-3)


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


HARD_CASE = [21, 35, 16, 16, 16, 56, 49, 15, 39, 58, 28, 42, 16, 48, 29, 43]


def _random_batch(m, lengths):
    n, L = len(lengths), max(lengths)
    vocab = m.embed_tokens.num_embeddings
    tok = torch.randint(4, vocab - 1, (n, L))
    for i, ln in enumerate(lengths):
        tok[i, ln:] = m.padding_idx
    return (
        tok,
        torch.rand(n, L, L),
        torch.randn(n, L, 3),
        torch.randint(0, vocab * vocab, (n, L, L)),
    )


def test_acceptance_uses_the_full_cost_model_including_group_overhead(setup):
    """Ignoring per-group overhead made this split look like a 27% saving; the full model says ~6%, below the 10% bar."""
    cm = P.CostModel()
    groups = P.plan_buckets(HARD_CASE, cost=cm)
    assert len(groups) >= 2  # the planner does split (it is optimal for the full model) ...
    B, L = len(HARD_CASE), max(HARD_CASE)
    saving = 1 - cm.plan_cost(groups, HARD_CASE) / cm.group_cost(B, L)
    assert 0 < saving < cm.min_saving
    assert not P.worth_splitting(
        groups, HARD_CASE, B, L, cm
    )  # ... but the wrapper must not accept it
    # a clearly ragged batch is still accepted
    ragged = [12] * 8 + [80] * 8
    g = P.plan_buckets(ragged, cost=cm)
    assert P.worth_splitting(g, ragged, len(ragged), max(ragged), cm)
    # and the real forward agrees: one encoder call for the hard case, several for the ragged one
    m, _ = setup
    ua.enable()
    for lengths, expect_split in ((HARD_CASE, False), (ragged, True)):
        calls = []
        h = m.encoder.register_forward_hook(
            lambda mod, inp, out, sink=calls: sink.append(inp[0].shape[0])
        )
        try:
            with torch.no_grad():
                m(*_random_batch(m, lengths), return_repr=True)
        finally:
            h.remove()
        assert (len(calls) > 1) == expect_split, calls


@pytest.mark.parametrize(
    "name,value",
    [
        ("UNIMOL_ACCEL_COST_PAIR", "nan"),
        ("UNIMOL_ACCEL_COST_PAIR", "-0.1"),
        ("UNIMOL_ACCEL_COST_PAIR", "inf"),
        ("UNIMOL_ACCEL_COST_GROUP", "-5"),
        ("UNIMOL_ACCEL_COST_GROUP", "nan"),
        ("UNIMOL_ACCEL_COST_MIN_GROUP", "0"),
        ("UNIMOL_ACCEL_COST_MIN_SAVING", "1.5"),
        ("UNIMOL_ACCEL_COST_MIN_SAVING", "-0.2"),
        ("UNIMOL_ACCEL_COST_MIN_SAVING", "nan"),
        ("UNIMOL_ACCEL_COST_PAIR", "junk"),
    ],
)
def test_invalid_cost_model_values_are_rejected_with_a_warning(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.warns(UserWarning, match=name):
        cm = P.CostModel.from_env()
    assert cm == P.CostModel()  # the default replaced the invalid value


def test_valid_boundary_values_are_accepted(monkeypatch):
    monkeypatch.setenv("UNIMOL_ACCEL_COST_PAIR", "0")
    monkeypatch.setenv("UNIMOL_ACCEL_COST_GROUP", "0")
    monkeypatch.setenv("UNIMOL_ACCEL_COST_MIN_SAVING", "0")
    cm = P.CostModel.from_env()
    assert (cm.pair_weight, cm.group_overhead, cm.min_saving) == (0.0, 0.0, 0.0)


def test_cost_model_is_configurable_and_default_is_unchanged(monkeypatch):
    cm = P.CostModel()
    assert (cm.pair_weight, cm.group_overhead, cm.min_group, cm.min_saving) == (
        1 / 16,
        48.0,
        4,
        0.10,
    )
    lengths = [10, 12, 80, 11, 79, 40, 41, 13, 12, 78, 11, 10, 60, 61, 62, 63]
    base = P.plan_buckets(lengths)
    assert base == P.plan_buckets(lengths, cost=cm)
    monkeypatch.setenv(
        "UNIMOL_ACCEL_COST_GROUP", "1e9"
    )  # splitting is prohibitively expensive: one group
    assert len(P.plan_buckets(lengths)) == 1
    monkeypatch.setenv("UNIMOL_ACCEL_COST_GROUP", "0")
    monkeypatch.setenv("UNIMOL_ACCEL_COST_MIN_GROUP", "1")
    assert len(P.plan_buckets(lengths)) > len(base)
    assert ua.status()["bucketing"]["cost_model"]["group_overhead"] == 0.0


def test_plan_is_optimal_for_its_cost_model():
    """Exhaustive check on small inputs: no contiguous partition of the sorted lengths is cheaper."""
    import itertools

    cm = P.CostModel(min_group=1)
    lengths = [9, 30, 31, 32, 70, 71]
    ls = sorted(lengths)

    def total(cuts):
        edges = [0, *cuts, len(ls)]
        return sum(cm.group_cost(b - a, ls[b - 1]) for a, b in zip(edges, edges[1:]))

    best = min(
        total(c) for r in range(len(ls)) for c in itertools.combinations(range(1, len(ls)), r)
    )
    groups = P.plan_buckets(lengths, cost=cm)
    got = sum(cm.group_cost(len(g), max(lengths[i] for i in g)) for g in groups)
    assert got == pytest.approx(best)
