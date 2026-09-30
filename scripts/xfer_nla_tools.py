#!/usr/bin/env python3
"""Batched frozen-NLA helpers for the 8B interface experiments.

`NLA.verbalize` handles one vector at a time. Every AV prompt is identical except
for the injected placeholder row, so B vectors share one prompt length and batch
without padding. Greedy decoding, max 300 new tokens, as in derive_pool.py.

The AR prompts differ in length: they are right-padded, and the state is read at
each row's LAST REAL token. Attention is causal, so trailing pads cannot affect it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reasoning_attention.nla.injection import inject_at_placeholder, normalize_activation  # noqa: E402
from reasoning_attention.nla.prompts import build_ar_prompt, build_av_messages, extract_explanation  # noqa: E402

AV_PATH = ROOT / "checkpoints/av_rl_ep1"
AR_PATH = ROOT / "checkpoints/ar_rl_ep1"


class FrozenNLA:
    def __init__(self, device: str = "cuda") -> None:
        from reasoning_attention.nla.arch import inner_transformer
        from reasoning_attention.nla.model import NLA
        from steer_demo import load_ar

        self.nla = NLA.av_only(str(AV_PATH))
        self.nla.av.eval()
        self.cfg = self.nla.config
        self.tok = self.nla.tokenizer
        self.scale = self.cfg.resolve_injection_scale(self.nla.d_model)
        assert abs(self.scale - 1000.0) < 1e-6, self.scale
        prompt = self.tok.apply_chat_template(
            build_av_messages(self.cfg.placeholder_token), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        self.av_ids = self.tok(prompt, return_tensors="pt")["input_ids"].to(device)
        assert int((self.av_ids == self.cfg.placeholder_token_id).sum()) == 1, "placeholder count"
        self.ar_tok, self.ar_backbone, self.affine = load_ar(AR_PATH)
        self.ar_trunk = inner_transformer(self.ar_backbone)
        for p in list(self.nla.av.parameters()) + list(self.ar_backbone.parameters()) + list(
                self.affine.parameters()):
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def verbalize(self, vecs: torch.Tensor, max_new_tokens: int = 300, batch: int = 64,
                  temperature: float = 0.0, seed: int | None = None) -> list[str]:
        """vecs [N, 2048] in 1.7B layer-20 space (any norm; rescaled to 1000).

        temperature 0 = greedy (default, as in Finding 15); > 0 samples (T1 uses T=1,
        the AV's RL sampling temperature), seeded per call for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
        out: list[str] = []
        emb_layer = self.nla.av.get_input_embeddings()
        for i in range(0, len(vecs), batch):
            v = normalize_activation(vecs[i : i + batch].to(self.device).float(), self.scale)
            ids = self.av_ids.expand(len(v), -1)
            emb = inject_at_placeholder(ids, emb_layer(ids), v, self.cfg.placeholder_token_id)
            gen = self.nla.av.generate(inputs_embeds=emb, attention_mask=torch.ones_like(ids),
                                       max_new_tokens=max_new_tokens,
                                       do_sample=temperature > 0,
                                       **({"temperature": temperature, "top_p": 1.0, "top_k": 0}
                                          if temperature > 0 else {}),
                                       pad_token_id=self.tok.pad_token_id)
            for g in gen:
                out.append(extract_explanation(self.tok.decode(g, skip_special_tokens=True)) or "")
        return out

    @torch.no_grad()
    def reconstruct(self, texts: list[str], batch: int = 64) -> torch.Tensor:
        """AR(z): [N, 2048] predicted layer-20 activation (fp32)."""
        res = []
        for i in range(0, len(texts), batch):
            enc = self.ar_tok([build_ar_prompt(t) for t in texts[i : i + batch]],
                              return_tensors="pt", padding=True, padding_side="right").to(self.device)
            h = self.ar_trunk(**enc).last_hidden_state
            last = enc["attention_mask"].sum(1) - 1
            res.append(self.affine(h[torch.arange(len(last)), last].float()))
        return torch.cat(res)


if __name__ == "__main__":
    # Parity check against the reference single-vector implementations.
    from steer_demo import reconstruct as ref_reconstruct

    P = torch.load(ROOT / "data/xfer8b/pairs.pt", weights_only=False)
    f = FrozenNLA()
    hs = P["h_s"][:6].float()
    zb = f.verbalize(hs)
    zs = [f.nla.verbalize(h, max_new_tokens=300, return_explanation=True).strip() for h in hs]
    print("verbalize batched == single:", [a.strip() == b for a, b in zip(zb, zs)])
    rb = f.reconstruct(zs)
    rs = torch.stack([ref_reconstruct(f.ar_tok, f.ar_backbone, f.affine, z) for z in zs])
    print("reconstruct max rel diff:", float(((rb - rs).norm(dim=-1) / rs.norm(dim=-1)).max()))
    cos = torch.nn.functional.cosine_similarity(rs, hs.cuda(), dim=-1)
    print("cos(AR(AV(h)), h):", [round(float(c), 3) for c in cos])
    print("example:\n", zs[0][:600])
