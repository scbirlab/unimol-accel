"""Fake Hugging Face hub for offline, deterministic tests of scbirlab/aspect with random weights."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch


@pytest.fixture(scope="session")
def fake_unimol_hub(tmp_path_factory):
    pytest.importorskip("aspect")
    pytest.importorskip("unimol_tools")
    from _um_common import make_model, make_dict

    root = tmp_path_factory.mktemp("unimol_hub")
    d = make_dict()
    with open(root / "mol.dict.txt", "w") as f:  # unimol_tools adds [MASK] itself
        for sym, cnt in zip(d.symbols, d.count):
            if sym != "[MASK]":
                f.write(f"{sym} {cnt}\n")
    m = make_model(seed=3)
    torch.save({"model": m.state_dict()}, root / "mol_pre_all_h_220816.pt")
    torch.save({"model": m.state_dict()}, root / "mol_pre_no_h_220816.pt")
    return root


@pytest.fixture()
def aspect_unimol(fake_unimol_hub, monkeypatch, tmp_path):
    import huggingface_hub
    import aspect.transform.unimol as au

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo, name, **kw: str(fake_unimol_hub / name)
    )
    au._load_unimol_cached.cache_clear()
    yield au
    au._load_unimol_cached.cache_clear()
