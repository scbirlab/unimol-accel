import sys
import os
import time

if os.environ.get("UNIMOL_TOOLS_PATH"):
    sys.path.insert(0, os.environ["UNIMOL_TOOLS_PATH"])
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
from mols import SMILES

# avoid importing the weight-download path
from unimol_tools.data import Dictionary
from unimol_tools.data.conformer import inner_smi2coords, coords2unimol
from unimol_tools.models import unimol as um
from unimol_tools.models.transformers import TransformerEncoderWithPair

ATOMS = "C N O S H Cl F Br I Si P B Na K Al Ca Sn As Hg Fe Zn Cr Se Gd Au Li".split()


def make_dict():
    d = Dictionary()
    for s in ["[CLS]", "[PAD]", "[SEP]", "[UNK]"]:
        d.add_symbol(s, is_special=True)
    for a in ATOMS:
        d.add_symbol(a)
    d.add_symbol("[MASK]", is_special=True)
    return d


def make_model(seed=0, output_dim=1):
    torch.manual_seed(seed)
    dic = make_dict()
    m = um.UniMolModel.__new__(um.UniMolModel)
    nn.Module.__init__(m)
    a = um.molecule_architecture()
    m.args = a
    m.output_dim = output_dim
    m.data_type = "molecule"
    m.remove_hs = False
    m.dictionary = dic
    m.mask_idx = dic.index("[MASK]")
    m.padding_idx = dic.pad()
    m.embed_tokens = nn.Embedding(len(dic), a.encoder_embed_dim, m.padding_idx)
    m.encoder = TransformerEncoderWithPair(
        encoder_layers=a.encoder_layers,
        embed_dim=a.encoder_embed_dim,
        ffn_embed_dim=a.encoder_ffn_embed_dim,
        attention_heads=a.encoder_attention_heads,
        emb_dropout=a.emb_dropout,
        dropout=a.dropout,
        attention_dropout=a.attention_dropout,
        activation_dropout=a.activation_dropout,
        max_seq_len=a.max_seq_len,
        activation_fn=a.activation_fn,
        no_final_head_layer_norm=a.delta_pair_repr_norm_loss < 0,
    )
    K = 128
    n_edge = len(dic) * len(dic)
    m.gbf_proj = um.NonLinearHead(K, a.encoder_attention_heads, a.activation_fn)
    m.gbf = um.GaussianLayer(K, n_edge)
    m.classification_head = um.LinearHead(a.encoder_embed_dim, output_dim, a.pooler_dropout)
    # make the Gaussian kernel parameters non-trivial (as in a trained model)
    with torch.no_grad():
        m.gbf.means.weight.uniform_(0, 3)
        m.gbf.stds.weight.uniform_(0.3, 3)
        m.gbf.mul.weight.uniform_(0.8, 1.2)
        m.gbf.bias.weight.uniform_(-0.1, 0.1)
    return m


def make_samples(n, remove_hs=False, seed=42):
    dic = make_dict()
    out = []
    for s in (SMILES * ((n // len(SMILES)) + 1))[:n]:
        atoms, coords, _ = inner_smi2coords(s, seed=seed, mode="fast", remove_hs=remove_hs)
        out.append((coords2unimol(atoms, coords, dic, 256, remove_hs=remove_hs), 0.0))
    return out


def t(f, n=3):
    f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)
