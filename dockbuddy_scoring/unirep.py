# -*- coding: utf-8 -*-
"""libscore_unirep.py —— UniRep 打分适配器(2026-10-11 新增)
===================================================================================
UniRep(Alley et al. 2019)用 mLSTM 学习蛋白序列表示, 突变效应打分需**evotuning**
(在目标家族上微调一个线性打分头)。打分语义: 突变后表示经 evotuned 头输出的变化。

依赖: jax + unirep(GitHub ElArkk/uniRep) + mLSTM 权重(~1GB) + evotuning 训练。
诚实边界: evotuning 需同源序列训练打分头, 非即开即用; 本适配器为薄骨架。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple


def _fields(v) -> Tuple[str, str, str, int]:
    if isinstance(v, dict):
        site = int(v["site"]); wt = str(v["from"]).upper(); mt = str(v["to"]).upper()
        vid = str(v.get("id") or f"{wt}{site}{mt}")
        return vid, wt, mt, site
    return (str(getattr(v, "id", "")), str(getattr(v, "from_aa")).upper(),
            str(getattr(v, "to_aa")).upper(), int(getattr(v, "site")))


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    """UniRep evotuning 打分(薄骨架), 返回 {variant_id: score}。

    需 evotuned 权重头路径 cfg['weights'](必填); 未提供则报错。
    """
    weights = str(cfg.get("weights", "") or "").strip()
    if not weights:
        raise RuntimeError(
            "UniRep 打分需 evotuned 权重头(weights 参数); evotuning 需同源序列训练, "
            "非即开即用, 详见 GitHub ElArkk/uniRep README")

    try:
        import jax  # noqa: F401
    except ImportError as e:
        raise RuntimeError("UniRep 需 jax(见 ElArkk/uniRep 环境)") from e

    seq = "".join(str(wt_seq).split()).upper()
    out: Dict[str, float] = {}
    for v in variants:
        vid, wt, mt, site = _fields(v)
        if 1 <= site <= len(seq) and seq[site - 1] == wt:
            # 薄骨架: 真实 evotuned 头打分未实现(需训练), 占位返回 None 语义
            out[vid] = float("nan")
    if out:
        raise RuntimeError(
            "UniRep evotuned 打分头未接入(需训练 evotuning 权重); 本适配器仅环境守卫")
    return out


__all__ = ["score_variants"]
