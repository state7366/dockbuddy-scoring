# -*- coding: utf-8 -*-
"""msa_stats.py —— MSA 位点频率统计(从 DockBuddy prescreen/sequence_scan 内联的解耦版本)。"""
from typing import Dict, List, Sequence

_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def position_freqs(seqs: Sequence[str], L: int) -> List[Dict[str, float]]:
    """每位点氨基酸频率(忽略 gap), 返回 [{aa: freq}, ...](freq 归一化)。"""
    out: List[Dict[str, float]] = []
    for i in range(L):
        counts: Dict[str, int] = {}
        for s in seqs:
            c = s[i].upper() if i < len(s) else "-"
            if c in _AA_SET:
                counts[c] = counts.get(c, 0) + 1
        total = sum(counts.values()) or 1
        out.append({aa: n / total for aa, n in counts.items()})
    return out
