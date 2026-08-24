"""Activation injection — overwrite the placeholder embedding with a vector.

Recreated from the reference repo:
  - `inject_at_marked_positions` (nla/injection.py) — simplified for single-GPU
    (no sequence-parallel slice, no distributed abort path).
  - `normalize_activation` (nla/schema.py) — scale a vector to a target L2-norm.

The placeholder token's lexical identity is irrelevant: the model's layers never
read it, only the activation vector we swap into its embedding slot.
"""

from __future__ import annotations

import torch

_EPS = 1e-8


def normalize_activation(v: torch.Tensor, target_scale: float | None) -> torch.Tensor:
    """Scale `v` (..., d) to L2-norm `target_scale`, or pass through if None.

    Zero vectors stay zero. Norm computed in fp32 for precision, then cast back.
    """
    if target_scale is None:
        return v
    norm = v.float().norm(dim=-1, keepdim=True)
    factor = (norm / target_scale).clamp_min(_EPS)
    scaled = v.float() / factor
    # Keep exact zeros zero (norm 0 -> factor _EPS -> 0/_EPS == 0 already, but be explicit).
    scaled = torch.where(norm > _EPS, scaled, v.float())
    return scaled.to(v.dtype)


def inject_at_placeholder(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    vectors: torch.Tensor,
    placeholder_id: int,
    left_id: int | None = None,
    right_id: int | None = None,
) -> torch.Tensor:
    """Overwrite embedding rows at placeholder positions with activation vectors.

    input_ids:   [B, S]            — the token stream.
    embeddings:  [B, S, d]         — embedding-layer output. Cloned; original untouched.
    vectors:     [N, d]            — one row per injection site, in row-major order.
    placeholder_id:                 token id marking an injection site.
    left_id/right_id:               if given, a match is only valid when its
                                    immediate neighbors equal these ids (rejects
                                    false positives). Mirrors the repo's neighbor
                                    check; pass None to skip.

    Raises RuntimeError if the number of valid sites != vectors.shape[0].
    """
    assert vectors.ndim == 2 and vectors.shape[1] == embeddings.shape[-1], (
        f"vectors must be [N, d_model], got {tuple(vectors.shape)}, d_model={embeddings.shape[-1]}"
    )
    seq_len = input_ids.shape[-1]
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)

    matches = (input_ids == placeholder_id).nonzero()  # [M, 2] (batch, seq), sorted
    vec_idx = 0
    for b, p in matches.tolist():
        if left_id is not None and (p == 0 or input_ids[b, p - 1] != left_id):
            continue
        if right_id is not None and (p == seq_len - 1 or input_ids[b, p + 1] != right_id):
            continue
        out[b, p] = vectors[vec_idx]
        vec_idx += 1

    if vec_idx != vectors.shape[0]:
        raise RuntimeError(
            f"found {vec_idx} valid injection sites, expected {vectors.shape[0]}. "
            f"Check the prompt template, placeholder token id, or neighbor ids."
        )
    return out
