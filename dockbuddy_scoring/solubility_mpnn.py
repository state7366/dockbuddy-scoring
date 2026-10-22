# -*- coding: utf-8 -*-
"""libscore_solubility_mpnn.py —— SolubilityMPNN 可溶性重设计打分(2026-10-11)
===================================================================================
SolubilityMPNN 是 ProteinMPNN 在可溶性数据上微调的变体, 调用同 ProteinMPNN。
打分语义: 反向折叠采样中某替换出现的频率(可溶性偏好)。
依赖: ProteinMPNN 仓库 + SolubilityMPNN 权重。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    from .thermompnn import score_variants as _tv
    return _tv(wt_seq, variants, **cfg)


__all__ = ["score_variants"]
