"""unimol_accel under scbirlab/aspect's `unimol` featuriser (real Aspect + real UniMolRepr, random weights)."""

import numpy as np
import pytest

pytest.importorskip("aspect")
pytest.importorskip("schemist")
import unimol_accel as ua
from mols import SMILES

MOLS = SMILES[:14] + [None, "not a smiles"]


def conformer_rows(n_conformers=1, mols=MOLS):
    from aspect.transform.conformers import Conformers

    return Conformers(n_conformers=n_conformers)({"smiles": mols}, "smiles")


def embed(au, rows, conformers, batch_size=8, accel=True):
    (ua.enable if accel else ua.disable)()
    try:
        fn = au.UniMolEmbedding(conformers=conformers, batch_size=batch_size, device="cpu")
        return (
            fn(rows if conformers == "aspect" else {"s": rows}, "s")
            if conformers == "unimol"
            else fn({"c": rows}, "c")
        )
    finally:
        ua.enable()


@pytest.mark.parametrize("batch_size", [4, 8, 32])
def test_pregenerated_conformers_match(aspect_unimol, batch_size):
    rows = conformer_rows()
    ref = embed(aspect_unimol, rows, "aspect", batch_size, accel=False)
    new = embed(aspect_unimol, rows, "aspect", batch_size, accel=True)
    assert new.shape == (len(MOLS), 512)
    np.testing.assert_allclose(new, ref, atol=1e-5, rtol=1e-5)
    assert not new[-2:].any()  # missing / invalid rows stay zero vectors


def test_multiple_conformers_per_molecule_span_batches(aspect_unimol):
    rows = conformer_rows(3)
    ref = embed(aspect_unimol, rows, "aspect", 5, accel=False)
    new = embed(aspect_unimol, rows, "aspect", 5, accel=True)
    np.testing.assert_allclose(new, ref, atol=1e-5, rtol=1e-5)


def test_smiles_mode_matches(aspect_unimol):
    smiles = [s for s in MOLS if isinstance(s, str)]
    ref = embed(aspect_unimol, smiles, "unimol", 8, accel=False)
    new = embed(aspect_unimol, smiles, "unimol", 8, accel=True)
    np.testing.assert_allclose(new, ref, atol=1e-5, rtol=1e-5)


def test_patches_active_inside_aspect_call(aspect_unimol, monkeypatch):
    seen = []
    from unimol_accel import _patches as P

    orig = P._merge_outputs
    monkeypatch.setattr(P, "_merge_outputs", lambda *a, **k: (seen.append(1), orig(*a, **k))[1])
    ua.enable()
    wide = SMILES[:32]  # ragged batch: planner must split it
    rows = conformer_rows(mols=wide)
    new = embed(aspect_unimol, rows, "aspect", 32, accel=True)
    ref = embed(aspect_unimol, rows, "aspect", 32, accel=False)
    np.testing.assert_allclose(new, ref, atol=1e-5, rtol=1e-5)
    assert seen, "length bucketing never ran through UniMolRepr.get_repr"
