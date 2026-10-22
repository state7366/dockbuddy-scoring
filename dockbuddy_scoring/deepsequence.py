# -*- coding: utf-8 -*-
"""libscore_deepsequence.py —— DeepSequence 打分适配器(2026-10-11 新增)
===================================================================================
DeepSequence(Riesselman et al. 2018)用家族特异 VAE 建模序列分布, 突变效应 =
突变序列 vs WT 序列的 ELBO 差。**无预训练权重**——每个蛋白家族需用自身 MSA 训练
VAE(故非即开即用)。

诚实边界: 需 MSA + 训练 VAE(见 github.com/debbiemarkslab/DeepSequence)。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    msa = str(cfg.get("msa", "") or "").strip()
    weights = str(cfg.get("weights", "") or "").strip()
    if not msa:
        raise RuntimeError("DeepSequence 需 msa(a2m, 用于训练家族 VAE)")
    if not weights:
        raise RuntimeError(
            "DeepSequence 无预训练权重: 需先在本家族 MSA 上训练 VAE(--weights 指向"
            "训练产物); 训练流程见 github.com/debbiemarkslab/DeepSequence")
    # 真实打分(ELBO 差)需 DeepSequence 环境; 此处仅环境守卫
    raise RuntimeError(
        "DeepSequence 打分头未接入(需训练家族 VAE); 本适配器仅环境守卫")


__all__ = ["score_variants"]
