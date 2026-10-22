# -*- coding: utf-8 -*-
"""libscore_esm_if.py —— ESM-IF1 反向折叠打分适配器(2026-10-11 新增)
===================================================================================
ESM-IF1(Hsu et al. 2022)反向折叠: 给定骨架坐标, 预测每个位置的氨基酸 logits。
打分语义: 突变残基 vs WT 残基在该位置的 logit 差(正=结构更偏好突变残基)。

依赖: fair-esm(esm.inverse_folding) + 权重 esm_if1_gvp4_t16_142M_UR50(~500MB,
torch hub 自动下载)。需 GPU/CPU。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple


def _fields(v) -> Tuple[str, str, str, int]:
    if isinstance(v, dict):
        site = int(v["site"]); wt = str(v["from"]).upper(); mt = str(v["to"]).upper()
        vid = str(v.get("id") or v.get("variant_id") or f"{wt}{site}{mt}")
        return vid, wt, mt, site
    return (str(getattr(v, "id", "")), str(getattr(v, "from_aa")).upper(),
            str(getattr(v, "to_aa")).upper(), int(getattr(v, "site")))


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    """ESM-IF1 反向折叠打分, 返回 {variant_id: logit(mt)-logit(wt)}。

    参数: structure_pdb(必填, 结构), chain(默认 A), device(默认 cpu)。
    """
    pdb = str(cfg.get("structure_pdb", "") or "").strip()
    if not pdb:
        raise RuntimeError("ESM-IF1 需要 structure_pdb(骨架结构 PDB)")
    chain = str(cfg.get("chain", "A") or "A")
    device = str(cfg.get("device", "cpu") or "cpu")

    try:
        # biotite>=1.0 移除 filter_backbone → filter_peptide_backbone(esm.inverse_folding 旧 API)
        import biotite.structure as _bs
        if not hasattr(_bs, "filter_backbone") and hasattr(_bs, "filter_peptide_backbone"):
            _bs.filter_backbone = _bs.filter_peptide_backbone
        import torch
        import esm
        import esm.inverse_folding
        from esm.inverse_folding.util import (CoordBatchConverter,
                                              extract_coords_from_structure,
                                              load_structure)
    except ImportError as e:
        raise RuntimeError(
            "ESM-IF1 需 fair-esm + torch_geometric + torch_scatter + biotite; "
            "pip install torch_geometric torch-scatter") from e

    models_dir = str(cfg.get("models_dir", "") or "").strip()
    local = os.path.join(models_dir, "esm_if1_gvp4_t16_142M_UR50.pt") if models_dir else ""
    if local and os.path.isfile(local):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(local)
    else:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)

    structure = load_structure(pdb, chain)
    coords, native_seq = extract_coords_from_structure(structure)
    seq = "".join(str(wt_seq).split()).upper()

    # CoordBatchConverter 构造正确输入, model.forward 得 logits(prev=tokens[:-1])
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords_b, confidence, _, tokens, padding_mask = batch_converter(batch)
    coords_b = coords_b.to(device)
    confidence = confidence.to(device)
    tokens = tokens.to(device)
    padding_mask = padding_mask.to(device)
    prev = tokens[:, :-1]
    with torch.no_grad():
        logits, _ = model.forward(coords_b, padding_mask, confidence, prev)
    logits = logits.cpu()  # [batch, vocab, seqlen]

    out: Dict[str, float] = {}
    for v in variants:
        vid, wt, mt, site = _fields(v)
        if not (1 <= site <= logits.shape[2]):
            continue
        ti, tj = alphabet.get_idx(wt), alphabet.get_idx(mt)
        out[vid] = round(float(logits[0, tj, site - 1] - logits[0, ti, site - 1]), 6)
    return out


__all__ = ["score_variants"]
