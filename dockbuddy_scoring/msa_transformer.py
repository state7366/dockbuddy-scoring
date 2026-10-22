# -*- coding: utf-8 -*-
"""libscore_msa_transformer.py —— MSA Transformer(ESM-MSA-1b)打分适配器(2026-10-11)
===================================================================================
ESM-MSA-1b(Rao et al. 2021)用多序列比对(而非单序列)预测每个位置氨基酸 logits。
打分语义: 突变残基 vs WT 残基的伪似然差(正=突变残基更被 MSA 支持)。

依赖: fair-esm + 权重 esm_msa1b_t12_100M_UR50S(~500MB, torch hub)。需 MSA + GPU。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple


def _fields(v) -> Tuple[str, str, str, int]:
    if isinstance(v, dict):
        site = int(v["site"]); wt = str(v["from"]).upper(); mt = str(v["to"]).upper()
        vid = str(v.get("id") or f"{wt}{site}{mt}")
        return vid, wt, mt, site
    return (str(getattr(v, "id", "")), str(getattr(v, "from_aa")).upper(),
            str(getattr(v, "to_aa")).upper(), int(getattr(v, "site")))


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    """MSA Transformer 伪似然打分, 返回 {variant_id: logit(mt)-logit(wt)}。

    参数: msa(a3m/fasta, 必填), device(默认 cpu)。
    """
    import torch
    import esm

    msa_path = str(cfg.get("msa", "") or "").strip()
    if not msa_path:
        raise RuntimeError("MSA Transformer 需要 msa(a3m/fasta)")
    device = str(cfg.get("device", "cpu") or "cpu")

    models_dir = str(cfg.get("models_dir", "") or "").strip()
    local = os.path.join(models_dir, "esm_msa1b_t12_100M_UR50S.pt") if models_dir else ""
    if local and os.path.isfile(local):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(local)
    else:
        model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model = model.eval().to(device)

    # 读 MSA → 转为 one-hot batch
    from esm.data import MSABatchConverter
    seqs = []
    with open(msa_path, encoding="utf-8") as f:
        cur = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append(cur)
                cur = ""
            else:
                cur += line
        if cur:
            seqs.append(cur)
    # a3m: 去小写插入, 保持大写 + gap(对齐)
    seqs = ["".join(c for c in s if c.isupper() or c == "-") for s in seqs if s]
    if not seqs:
        raise RuntimeError("MSA 为空")
    # 截断到合理深度(MSA Transformer 显存随深度线性)
    max_depth = int(cfg.get("max_depth", 64) or 64)
    seqs = seqs[:max_depth]

    batch_converter = MSABatchConverter(alphabet)
    msa_pairs = [(f"seq{i}", s) for i, s in enumerate(seqs)]
    batch_tokens = batch_converter(msa_pairs)[2].to(device)
    with torch.no_grad():
        logits = model(batch_tokens, repr_layers=[], return_contacts=False)["logits"]

    out: Dict[str, float] = {}
    for v in variants:
        vid, wt, mt, site = _fields(v)
        # logits [batch, depth, seqlen, vocab]; 取 depth=0(query/WT 序列)。
        # MSABatchConverter prepend_bos=True: 残基 site(1-based) 位于 token 位置
        # site(BOS 占 0), 与官方 examples/variant-prediction/predict.py 的
        # label_row "score = token_probs[0, 1 + idx]" 一致(1 + (site-1) == site)。
        # 故索引应为 site, 而非 site-1(旧版 off-by-one 取错残基)。
        if not (1 <= site < logits.shape[2]):
            continue
        tok = [alphabet.get_idx(a) for a in (wt, mt)]
        out[vid] = round(float(logits[0, 0, site, tok[1]] - logits[0, 0, site, tok[0]]), 6)
    return out


__all__ = ["score_variants"]
