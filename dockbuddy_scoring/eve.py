# -*- coding: utf-8 -*-
"""libscore_eve.py —— EVE 打分适配器(2026-10-11 新增)
===================================================================================
EVE(Frazer et al. 2021)用贝叶斯 VAE 建模家族序列分布, 突变效应 = 突变序列的
evidence lower bound vs WT。**无预训练权重**——需本家族深 MSA 训练。

诚实边界: 需深 MSA + 训练(见 github.com/OATML-Markslab/EVE)。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    msa = str(cfg.get("msa", "") or "").strip()
    weights = str(cfg.get("weights", "") or "").strip()
    if not msa:
        raise RuntimeError("EVE 需 msa(a2m, 用于训练贝叶斯 VAE)")
    if not weights:
        raise RuntimeError(
            "EVE 无预训练权重: 需先在本家族深 MSA 上训练 VAE(--weights 指向训练产物); "
            "训练流程见 github.com/OATML-Markslab/EVE")
    raise RuntimeError("EVE 打分头未接入(需训练); 本适配器仅环境守卫")


__all__ = ["score_variants"]
