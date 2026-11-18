"""Patch implementations for ``unimol_tools``.

Every patch is

* **guarded** -- it is only applied if the SHA-256 of the upstream function's source matches a
  version this package was verified against (unknown upstream => original code is kept);
* **reversible** -- the original callable is stored and restored by :func:`disable`;
* **idempotent** -- applying twice is a no-op;
* **numerically conservative** -- no approximate arithmetic: results equal the original up to
  floating-point rounding (see ``tests/``).
"""

from __future__ import annotations

import contextvars
import dataclasses
import hashlib
import inspect
import math
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------
# thread/context-local state
# ---------------------------------------------------------------------------------------------
_SKIP_TAIL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unimol_accel_skip_tail", default=False
)


def _src_hash(fn: Callable) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


# SHA-256 prefixes of upstream sources verified against (deepmodeling/Uni-Mol @ 90f52c4, unimol_tools 0.1.3.post1)
KNOWN_HASHES: Dict[str, List[str]] = {
    "SelfMultiheadAttention.forward": ["42af08daa5b00868"],
    "TransformerEncoderWithPair.forward": ["13cb8f5af903dd7d"],
    "UniMolModel.forward": ["caf270b93386efae", "7b47c399bb22d300"],  # repo 90f52c4 / PyPI 0.1.6
}


@dataclass
class Patch:
    name: str
    owner: object  # class carrying the attribute
    attr: str
    make: Callable[[Callable], Callable]
    applied: bool = False
    original: Optional[Callable] = None
    reason: str = ""  # why it was not applied (empty if applied / not tried)

    def apply(self, force: bool = False) -> bool:
        if self.applied:
            return True
        current = getattr(self.owner, self.attr)
        digest = _src_hash(current)
        if not force and digest not in KNOWN_HASHES.get(self.name, []):
            self.reason = f"unverified upstream source (hash {digest}); original kept"
            return False
        self.original = current
        patched = self.make(current)
        patched.__wrapped__ = current  # type: ignore[attr-defined]
        patched._unimol_accel = True  # type: ignore[attr-defined]
        setattr(self.owner, self.attr, patched)
        self.applied, self.reason = True, ""
        return True

    def revert(self) -> None:
        if self.applied and self.original is not None:
            setattr(self.owner, self.attr, self.original)
        self.applied = False


# ---------------------------------------------------------------------------------------------
# 1. lean pair-bias attention
# ---------------------------------------------------------------------------------------------
def _make_attention_forward(original):
    from unimol_tools.models.transformers import softmax_dropout  # noqa: F401  (kept for parity)

    def forward(self, query, key_padding_mask=None, attn_bias=None, return_attn=False):
        """Same maths as upstream; fewer passes over the (B*H, L, L) logits tensor.

        * ``baddbmm(bias, q, k^T)`` replaces ``bmm`` followed by an in-place ``+=``;
        * the defensive ``clone`` in ``softmax_dropout(inplace=False)`` is dropped (softmax is not in place).
        Results are bit-identical to upstream in float32 on CPU (see tests).
        """
        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        if attn_bias is None or attn_bias.dtype != query.dtype:
            return original(
                self,
                query,
                key_padding_mask=key_padding_mask,
                attn_bias=attn_bias,
                return_attn=return_attn,
            )
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        k = (
            k.view(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
        )
        v = (
            v.view(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
        )
        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz and key_padding_mask.size(1) == src_len
        attn_weights = torch.baddbmm(attn_bias, q, k.transpose(1, 2))  # = bias + q k^T
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn = F.dropout(F.softmax(attn_weights, dim=-1), p=self.dropout, training=self.training)
        o = torch.bmm(attn, v)
        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        if not return_attn:
            return o
        return o, attn_weights, attn

    return forward


# ---------------------------------------------------------------------------------------------
# 2. encoder: skip the pair-representation tail when the caller discards it
# ---------------------------------------------------------------------------------------------
def _make_encoder_forward(original):
    def forward(self, emb, attn_mask=None, padding_mask=None):
        if not _SKIP_TAIL.get():
            return original(self, emb, attn_mask=attn_mask, padding_mask=padding_mask)
        # Identical to upstream up to the point where the discarded pair-representation outputs
        # (delta_pair_repr, norm losses) would be computed.
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        assert attn_mask is not None
        if attn_mask is not None and padding_mask is not None:
            attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
            attn_mask.masked_fill_(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
            padding_mask = None
        for layer in self.layers:
            x, attn_mask, _ = layer(
                x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        return x, None, None, None, None

    return forward


# ---------------------------------------------------------------------------------------------
# 3. UniMolModel.forward: length-bucketed sub-batching + skip tail
# ---------------------------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclasses.dataclass(frozen=True)
class CostModel:
    """Heuristic cost of one padded sub-batch: ``n * (L + pair_weight * L^2) + group_overhead * L``.

    ``L`` is the longest member, ``n`` the group size. The partition chosen by :func:`plan_buckets` is optimal
    for *this* function, not necessarily for measured run time, so the constants are tunable (environment
    variables below) and ``benchmarks/calibrate_cost_model.py`` fits them to the machine it runs on.
    """

    pair_weight: float = (
        1.0 / 16.0
    )  # UNIMOL_ACCEL_COST_PAIR: weight of the L^2 (pair) term relative to L
    group_overhead: float = (
        48.0  # UNIMOL_ACCEL_COST_GROUP: fixed cost of an extra group, in units of L
    )
    min_group: int = 4  # UNIMOL_ACCEL_COST_MIN_GROUP: smallest group (except the longest one)
    min_saving: float = (
        0.10  # UNIMOL_ACCEL_COST_MIN_SAVING: only bucket if the plan saves at least this fraction
    )

    @classmethod
    def from_env(cls) -> CostModel:
        """Read the constants from the environment; invalid values (NaN, inf, negative, out of range) are
        rejected with a warning and replaced by the default rather than silently accepted."""
        d = cls()

        def pick(name, default, ok, cast):
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                v = cast(raw)
                if ok(v):
                    return v
            except ValueError:
                pass
            warnings.warn(f"{name}={raw!r} is not valid; using the default {default}", stacklevel=3)
            return default

        finite_nonneg = lambda v: math.isfinite(v) and v >= 0.0  # noqa: E731
        return cls(
            pair_weight=pick("UNIMOL_ACCEL_COST_PAIR", d.pair_weight, finite_nonneg, float),
            group_overhead=pick("UNIMOL_ACCEL_COST_GROUP", d.group_overhead, finite_nonneg, float),
            min_group=pick("UNIMOL_ACCEL_COST_MIN_GROUP", d.min_group, lambda v: v >= 1, int),
            min_saving=pick(
                "UNIMOL_ACCEL_COST_MIN_SAVING",
                d.min_saving,
                lambda v: math.isfinite(v) and 0.0 <= v < 1.0,
                float,
            ),
        )

    def group_cost(self, n: int, L: int) -> float:
        return n * (L + self.pair_weight * L * L) + self.group_overhead * L

    def plan_cost(self, groups, lengths) -> float:
        """Total cost of a plan (including each group's overhead), in the same units as the planner optimises."""
        return sum(self.group_cost(len(g), max(lengths[i] for i in g)) for g in groups)


def plan_buckets(
    lengths: List[int],
    min_group: Optional[int] = None,
    group_overhead: Optional[float] = None,
    pair_weight: Optional[float] = None,
    cost: Optional[CostModel] = None,
) -> List[List[int]]:
    """Partition molecule indices (sorted by length) into groups minimising padded cost.

    ``cost`` (default: :meth:`CostModel.from_env`) defines ``cost(group) = n*(L + pair_weight*L^2) +
    group_overhead*L`` with L the longest member; the individual keyword arguments override single fields.
    Returns groups of *original indices*, each sorted by length. Optimal for that cost by dynamic programming.
    """
    cm = cost or CostModel.from_env()
    cm = dataclasses.replace(
        cm,
        **{
            k: v
            for k, v in (
                ("min_group", min_group),
                ("group_overhead", group_overhead),
                ("pair_weight", pair_weight),
            )
            if v is not None
        },
    )
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    ls = [lengths[i] for i in order]
    n = len(ls)
    INF = float("inf")
    best = [INF] * (n + 1)
    cut = [0] * (n + 1)
    best[0] = 0.0
    for end in range(1, n + 1):
        L = ls[end - 1]
        for start in range(max(0, end - 256), end):
            if end - start < cm.min_group and not (start == 0 and end == n):
                if end != n:  # only the final (longest) group may be smaller than min_group
                    continue
            c = best[start] + cm.group_cost(end - start, L)
            if c < best[end]:
                best[end], cut[end] = c, start
    groups, end = [], n
    while end > 0:
        start = cut[end]
        groups.append([order[i] for i in range(start, end)])
        end = start
    groups.reverse()
    return groups


def worth_splitting(groups, lengths, B: int, L: int, cm: CostModel) -> bool:
    """Accept a split only if the *full* cost model predicts a saving of at least ``min_saving``.

    Same objective the planner optimises (every group pays its overhead), compared with running the whole batch
    as one group padded to ``L``.
    """
    if len(groups) <= 1:
        return False
    return cm.plan_cost(groups, lengths) <= (1.0 - cm.min_saving) * cm.group_cost(B, L)


_UNMERGEABLE = object()


def _merge_outputs(parts, index, B, device):
    """Concatenate per-group outputs and restore the caller's molecule order.

    Handles the output shapes of both known ``UniMolModel.forward`` versions: a tensor (logits, or
    ``cls_repr`` in unimol_tools>=0.1.6) or a dict whose values are tensors (per-molecule rows) or lists
    (per-molecule items such as atomic symbols / representations).
    """
    inv = torch.empty(B, dtype=torch.long, device=device)
    inv[torch.tensor(index, device=device)] = torch.arange(B, device=device)

    def one(values):
        if all(torch.is_tensor(v) for v in values):
            cat = torch.cat(values, dim=0)
            return cat.index_select(0, inv.to(cat.device)) if cat.shape[0] == B else _UNMERGEABLE
        if all(isinstance(v, list) for v in values):
            flat = [item for v in values for item in v]
            if len(flat) != B:
                return _UNMERGEABLE
            ordered = [None] * B
            for pos, orig_i in enumerate(index):
                ordered[orig_i] = flat[pos]
            return ordered
        return _UNMERGEABLE

    if all(torch.is_tensor(p) or isinstance(p, list) for p in parts):
        return one(parts)
    if all(isinstance(p, dict) for p in parts) and len({tuple(p) for p in parts}) == 1:
        out = {}
        for key in parts[0]:
            out[key] = one([p[key] for p in parts])
            if out[key] is _UNMERGEABLE:
                return _UNMERGEABLE
        return out
    return _UNMERGEABLE


def _exporting() -> bool:
    """True while tracing/exporting (ONNX, ``torch.jit.trace``, ``torch.export``).

    Length bucketing reads tensor values (``.tolist()``) and branches on them, which a tracer would freeze
    into the graph for one batch, so exporters get upstream's own code. ``torch.compile`` is unaffected.
    """
    try:
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            return True
    except Exception:  # noqa: BLE001
        pass
    f = getattr(torch.compiler, "is_exporting", None)
    return bool(f()) if f is not None else False


def _make_unimol_forward(original):
    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        return_repr=False,
        return_atomic_reprs=False,
        **kwargs,
    ):
        if _exporting():
            return original(
                self,
                src_tokens,
                src_distance,
                src_coord,
                src_edge_type,
                return_repr=return_repr,
                return_atomic_reprs=return_atomic_reprs,
                **kwargs,
            )
        min_batch = _env_int("UNIMOL_ACCEL_BUCKET_MIN_BATCH", 8)
        bucket = os.environ.get("UNIMOL_ACCEL_BUCKET", "1") != "0"
        train_ok = os.environ.get("UNIMOL_ACCEL_BUCKET_TRAIN", "0") == "1"  # opt-in: see README
        token = _SKIP_TAIL.set(True)
        try:
            B = src_tokens.size(0)
            if (
                (not bucket)
                or B < min_batch
                or (self.training and not train_ok)
                or src_tokens.dim() != 2
            ):
                return original(
                    self,
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    return_repr=return_repr,
                    return_atomic_reprs=return_atomic_reprs,
                    **kwargs,
                )
            L = src_tokens.size(1)
            lengths = (src_tokens != self.padding_idx).sum(1).tolist()
            cm = CostModel.from_env()
            groups = plan_buckets(lengths, cost=cm)
            # accept a split only if the *full* cost model (overhead of every group included, as in the planner)
            # predicts the saving, relative to running everything as one padded group
            if not worth_splitting(groups, lengths, B, L, cm):
                return original(
                    self,
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    return_repr=return_repr,
                    return_atomic_reprs=return_atomic_reprs,
                    **kwargs,
                )
            parts, index = [], []
            for g in groups:
                Lg = max(lengths[i] for i in g)
                idx = torch.tensor(g, device=src_tokens.device)
                sub = original(
                    self,
                    src_tokens.index_select(0, idx)[:, :Lg],
                    src_distance.index_select(0, idx)[:, :Lg, :Lg],
                    src_coord.index_select(0, idx)[:, :Lg],
                    src_edge_type.index_select(0, idx)[:, :Lg, :Lg],
                    return_repr=return_repr,
                    return_atomic_reprs=return_atomic_reprs,
                    **kwargs,
                )
                parts.append(sub)
                index.extend(g)
            merged = _merge_outputs(parts, index, B, src_tokens.device)
            if merged is _UNMERGEABLE:  # unknown output structure: recompute the safe way
                return original(
                    self,
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    return_repr=return_repr,
                    return_atomic_reprs=return_atomic_reprs,
                    **kwargs,
                )
            return merged
        finally:
            _SKIP_TAIL.reset(token)

    return forward


def build_patches() -> List[Patch]:
    from unimol_tools.models import transformers as tr
    from unimol_tools.models import unimol as um

    return [
        Patch(
            "SelfMultiheadAttention.forward",
            tr.SelfMultiheadAttention,
            "forward",
            _make_attention_forward,
        ),
        Patch(
            "TransformerEncoderWithPair.forward",
            tr.TransformerEncoderWithPair,
            "forward",
            _make_encoder_forward,
        ),
        Patch("UniMolModel.forward", um.UniMolModel, "forward", _make_unimol_forward),
    ]
