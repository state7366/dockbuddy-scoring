# -*- coding: utf-8 -*-
"""libscore_thermompnn.py —— ThermoMPNN 热稳定重设计打分(2026-10-11 新增)
===================================================================================
ThermoMPNN(Dieckhaus et al. 2024)是 ProteinMPNN 在热稳定数据上微调的变体, 调用
方式与 ProteinMPNN 相同, 仅权重不同(thermoMPNN_default.pt ~40MB)。
打分语义: 反向折叠采样中某替换出现的频率(热稳定偏好)。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable


def _fields(v):
    if isinstance(v, dict):
        return (str(v.get("id") or f"{v['from']}{v['site']}{v['to']}"),
                int(v["site"]), str(v["to"]).upper())
    return (str(getattr(v, "id", "")), int(getattr(v, "site")),
            str(getattr(v, "to_aa")).upper())


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    """ThermoMPNN 重设计打分, 返回 {variant_id: 采样替换频率}。

    参数: receptor(结构 PDB), repo_dir(ProteinMPNN 仓库), checkpoint(权重路径),
    num_seqs, temperature, seed。
    """
    from .mpnn import redesign_sequences

    receptor = str(cfg.get("receptor", "") or "").strip()
    repo_dir = str(cfg.get("repo_dir", "") or "").strip()
    ckpt = str(cfg.get("checkpoint", "") or "").strip()
    if not receptor:
        raise RuntimeError("ThermoMPNN 需要 receptor(结构 PDB)")
    if not ckpt:
        raise RuntimeError(
            "ThermoMPNN 需要 checkpoint 权重路径(见 github.com/Kuhlman-Lab/ThermoMPNN)")

    seq = "".join(str(wt_seq).split()).upper()
    sites = sorted({_fields(v)[1] for v in variants})
    fixed = [i for i in range(1, len(seq) + 1) if i not in sites]

    res = redesign_sequences(
        seq, receptor, fixed, sites, repo_dir=repo_dir, checkpoint=ckpt,
        num_seqs=int(cfg.get("num_seqs", 8) or 8),
        temperature=float(cfg.get("temperature", 0.1) or 0.1),
        seed=int(cfg.get("seed", 0) or 0))
    seqs = res.get("sequences", [])
    out: Dict[str, float] = {}
    if seqs:
        for v in variants:
            vid, site, mt = _fields(v)
            if 1 <= site <= len(seqs[0]):
                out[vid] = sum(1 for s in seqs if s[site - 1] == mt) / len(seqs)
    return out


__all__ = ["score_variants"]
