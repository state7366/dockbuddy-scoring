# -*- coding: utf-8 -*-
"""
scoring/lfb.py —— LFB(似然-适应度桥接)打分校正层(2026-10-12 新增)
===================================================================================
定位: 接入 Pugh et al. 2025 (bioRxiv 10.1101/2025.05.20.655154) 提出的
Likelihood-Fitness Bridging(LFB) —— 一个**无需重训**的通用后处理校正层, 用于修正
语言模型「序列似然 = 适应度」假设的系统性偏差(系统发育结构 + 采样偏差; 模型容量
越大偏差越严重, 套用后可逆转「大模型性能平台期」)。

校正机制: 把「单条 WT 序列上的伪对数似然比」推广为「在同源家族(承受相似选择压力)
序列集合上的**平均**伪对数似然比」, 从而边缘化掉遗传漂变(可视为 Ornstein-Uhlenbeck
进化模型下的去漂变)。即, 对每个突变 v:

    LFB(v) = (1/|F|) Σ_{s∈F} [ log P(mt | s_mask) − log P(wt | s_mask) ]

其中 F = 与 WT 等长对齐的同源序列集合(家族 MSA), 突变位点 index 与 WT 一致。

与 esm2 的关系: 无家族序列(F 为空或仅含 WT)时退化为普通 ESM-2 zero-shot 打分。

设计决策(记录备查):
  - 复用 scoring.esm.esm1v_single_score 作为单条序列的打分核心(对任何 fair-esm mask
    模型通用), 本模块只做「家族平均」的薄封装, 不复制打分核心。
  - lfb_from_scorer 是纯函数(注入 single_score), 供 mock 单测无需 GPU/权重。
  - 零依赖 import: torch / fair-esm 延迟到函数内 import(同 esm.py / esm2.py 纪律)。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Sequence

LOG = logging.getLogger("dockpipeline")

ESM2_DEFAULT = "esm2_t33_650M_UR50D"


def _variant_fields(v) -> tuple:
    """从 Variant 或 dict 提取 (site 1-based, from_aa, to_aa, variant_id)。"""
    if isinstance(v, dict):
        site = int(v["site"])
        wt, mt = str(v["from"]).upper(), str(v["to"]).upper()
        vid = str(v.get("id") or v.get("variant_id") or f"{wt}{site}{mt}")
    else:
        site = int(getattr(v, "site"))
        wt = str(getattr(v, "from_aa")).upper()
        mt = str(getattr(v, "to_aa")).upper()
        vid = str(getattr(v, "id", None) or getattr(v, "label", None) or f"{wt}{site}{mt}")
    return site, wt, mt, vid


def lfb_from_scorer(wt_seq: str, variants: Iterable[Any],
                    family_seqs: Sequence[str],
                    single_score: Callable[[str, int, str, str], float],
                    ) -> Dict[str, float]:
    """LFB 核心(纯函数): 在家族序列集合上对单条打分取平均, 实现去漂变。

    参数:
      wt_seq: WT 序列(1-letter)。
      variants: 可迭代, 每项为 Variant 或 dict{'site','from','to'}。
      family_seqs: 与 WT 等长对齐的同源序列列表(家族 MSA 的行); 空则退化为 [wt_seq]。
      single_score(seq, site, from_aa, to_aa) -> float: 单条序列上的掩码伪对数似然比。

    返回 {variant_id: lfb_score}。纯逻辑, 无外部依赖, 可直接单测。
    """
    fields = [_variant_fields(v) for v in variants]
    fam = ["".join(str(s).split()) for s in family_seqs] if family_seqs else []
    if not fam:
        fam = ["".join(str(wt_seq).split())]
    out: Dict[str, float] = {}
    for site, wt, mt, vid in fields:
        acc = sum(single_score(s, site, wt, mt) for s in fam)
        out[vid] = round(acc / len(fam), 6)
    return out


def _load_family(family_seqs: Sequence[str], family_file: str) -> List[str]:
    """统一家族序列来源: 优先 family_seqs, 否则从 family_file 读(FASTA 或每行一条)。

    2026-10-15 基线扫描坑固化: a3m 输入含小写插入段(5972/6417 行), 不剥会导致
    esm alphabet 编码 KeyError 'p'; 同时 gap 列对 LFB 无意义。统一口径:
    去小写插入 → 去 gap → 只保留与首条(视为 query/WT)等长的序列。"""
    if family_seqs:
        return ["".join(str(s).split()) for s in family_seqs]
    if family_file:
        seqs: List[str] = []
        with open(family_file, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        # FASTA: 跳过 ">..." 头; 否则每行视为一条序列
        if any(ln.startswith(">") for ln in lines):
            buf = []
            for ln in lines:
                if ln.startswith(">"):
                    if buf:
                        seqs.append("".join(buf))
                        buf = []
                else:
                    buf.append(ln)
            if buf:
                seqs.append("".join(buf))
        else:
            seqs = ["".join(ln.split()) for ln in lines]
        # a3m 清洗(幂等: 纯 FASTA 无小写时原样保留)
        def _clean(s: str) -> str:
            return "".join(c for c in s if not c.islower()).replace("-", "").replace(".", "")
        cleaned = [_clean(s) for s in seqs]
        if cleaned and any(len(c) != len(s) for c, s in zip(cleaned, seqs)):
            # 检测到 a3m 插入段 → 以首条长度为准过滤
            ref_len = len(cleaned[0]) if cleaned else 0
            kept = [c for c in cleaned if c and len(c) == ref_len]
            return kept if kept else cleaned
        return [c for c in cleaned if c]
    return []


def score_variants(wt_seq: str, variants: Iterable[Any], *,
                   family_seqs: Sequence[str] = None,
                   family_file: str = "",
                   model_name: str = ESM2_DEFAULT,
                   device: str = "auto",
                   models_dir: str = "") -> Dict[str, float]:
    """LFB 打分入口(接入 registry 的 score 型 adapter `lfb`)。

    参数:
      family_seqs: 与 WT 等长对齐的同源序列列表(直接内联)。
      family_file: 同源序列文件(FASTA 或每行一条序列); 与 family_seqs 二选一。
      二者皆缺省时退化为普通 ESM-2 零样本(仅用 WT 一条)。
      model_name: ESM-2 模型名(缺省 esm2_t33_650M_UR50D)。
      device: 'cuda' / 'cpu' / 'auto'。
      models_dir: esm2 本地权重目录(空则走 fair-esm 自动下载)。
    """
    from dockbuddy_common import resolve_device
    from dockbuddy_scoring.esm import esm1v_single_score
    from dockbuddy_scoring.esm2 import load_esm2

    device = resolve_device(device)
    seq = "".join(str(wt_seq).split())
    fam = _load_family(family_seqs, family_file)
    if not fam:
        fam = [seq]
    model, alphabet = load_esm2(model_name, device, models_dir)

    def _single(s: str, site: int, wt: str, mt: str) -> float:
        return float(esm1v_single_score(model, alphabet, s, site, wt, mt))

    return lfb_from_scorer(seq, variants, fam, _single)


__all__ = ["lfb_from_scorer", "score_variants", "ESM2_DEFAULT"]
