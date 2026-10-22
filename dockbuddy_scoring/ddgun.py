# -*- coding: utf-8 -*-
"""
libscore_ddgun.py —— DDGun3D 稳定性 ΔΔG 打分适配器(2026-10-11 新增)
===================================================================================
定位: 打分矩阵的稳定性维度补充 —— 用 DDGun(Montanucci et al. 2019, NAR)预测单点
突变的 ΔΔG(kcal/mol)。DDGun 是**基于进化 profile 的统计势**方法(与 FoldX/Rosetta
的物理力场互补): 用 MSA 的每位点氨基酸频率 + 疏水性/BLOSUM/序列邻位势打分。

接入方式(pip 包, 纯 Python, 本机可跑):
  - pip install ddgun(0.0.2); 需 biopython + pandas。
  - **坑**: ddgun 依赖 Biopython 旧 API `Bio.PDB.Polypeptide.three_to_one`
    (1.80+ 改名 protein_letters_3to1), 需 monkeypatch。
  - ddgun_seq(substitution, profile) 需要「进化 profile」(每列 20 AA 频率 + SEQ),
    由 MSA 构建(本适配器用 prescreen.sequence_scan.position_freqs 算频率)。

诚实边界:
  - DDGun 是**进化统计势 ΔΔG**(基于家族频率 + BLOSUM + 疏水 + 序列邻位势),
    与 FoldX/Rosetta 的**物理力场**口径不同, 两数不可直接相减(遵循项目纪律
    「两种球心/口径不可混用」的同一精神)。
  - **符号语义(2026-10-16 实证定标, 推翻此前文档里的说法)**:
      本适配器输出的**负值 = 该突变更不利; 正值 = 相对更可接受**,
      即与 evmutation / saprot / venusplm / venusrem / msa_transformer **同向**
      (同为"负 = 差"), 而与 esm1v/esm2 **反向**(那两个是"正 = 更被 PLM 接受")。
    定标证据(PBATase 基线 39 变体, 交叉核对):
      · G→P 系列公认强去稳定 —— FoldX ΔΔG 全为**正**且很大
        (G17P +22.56 / G34P +13.73 / G5P +9.49 / G32P +7.24),
        对应 DDGun 全为**负**(−2.42 / −0.81 / −2.60 / −1.57);
      · 催化三联体/氧阴离子穴突变(S126A/S126A/H204F/H204L/F56T/T57A/T57G/A58Q)
        saprot 全为负(−5.91~−9.60), DDGun 亦全为负(−0.86~−3.19);
      · 相关系数: 修复索引后 r(ddgun, evmutation)=+0.59、r(ddgun, venusplm)=+0.49、
        r(ddgun, msa_transformer)=+0.35 —— 同为"负 = 差"故呈正相关;
        r(ddgun, foldx)=−0.31 —— foldx 是"正 = 去稳定", 故与之反号。三者自洽。
    ⚠️ 本文件此前把符号写反了("正值≈不被接受"), 且与「profile 索引差一位」叠加,
      曾导出 "F205G DDGun −2.399, 进化上可接受" 的结论 —— 该结论**双重错误**
      (位点错 + 符号反), 已作废。修正后 F205G = **−3.718, 即 DDGun 判定为
      家族中较不利的突变**(位点 205 家族为 F 36.2% / C 14.2% / L 9.7%, 无 G),
      与 FoldX +1.45(物理去稳定)**同向**, 并不存在原先描述的"进化 vs 物理张力"。
  - 打分只是稳定性/进化代理, 不预测活性; MSA 深度不足时 profile 频率不可靠。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

LOG = logging.getLogger("dockpipeline")

_AA_CODES = list("ACDEFGHIKLMNPQRSTVWY")


def _fields(v) -> Tuple[str, str, str, int]:
    if isinstance(v, dict):
        site = int(v["site"]); wt = str(v["from"]).upper(); mt = str(v["to"]).upper()
        vid = str(v.get("id") or v.get("variant_id") or f"{wt}{site}{mt}")
        return vid, wt, mt, site
    vid = str(getattr(v, "id", None) or getattr(v, "label", None) or "")
    return (vid, str(getattr(v, "from_aa")).upper(), str(getattr(v, "to_aa")).upper(),
            int(getattr(v, "site")))


def _load_ddgun():
    """monkeypatch Biopython 旧 API 后 import ddgun, 返回 ddgun_seq。"""
    from Bio.PDB import Polypeptide as _P
    if not hasattr(_P, "three_to_one"):
        _P.three_to_one = _P.protein_letters_3to1
    from ddgun import ddgun_seq
    return ddgun_seq


class _Substitution:
    """ddgun_seq 期望的 substitution 对象(funcs_seq 访问 .aa_from/.aa_to/.aa_pos)。"""

    def __init__(self, aa_from: str, aa_pos: int, aa_to: str):
        self.aa_from = aa_from
        self.aa_pos = aa_pos   # 0-based 序列索引
        self.aa_to = aa_to


def build_profile(seqs: List[str], reference: str):
    """从 MSA 构建 DDGun 兼容的 profile 对象(列=20AA+SEQ, index=POS **0-based**)。

    ⚠️⚠️ 索引基准是本项目踩过的**最严重的一处静默错误**(2026-10-16 发现并修复):
    index 必须是 **0-based (0..L-1)**, 不是 1-based。

    原因(ddgun 0.0.2 上游内部口径不一致, 我们只能适配 funcs_seq 那一侧):
      - ddgun/funcs_seq.py 的 evolScore1/hydroScore1/potential1 全部用
        ``prof.at[p, ...]`` 取值, 而 ``p = sub.aa_pos`` 是 **0-based**(ddgun 自己的
        parse_aa_change 里就是 ``aa_pos = int(...) - 1``)。
      - ddgun/profile.py 的 ``Profile`` 类却把 index 设成 1-based
        (``counts.index = range(1, len+1)``)。两侧对不上 → 用官方 Profile 跑
        ddgun_seq 时, **所有位点都取到前一位的 profile 行**, 静默差一位。

    实测后果(PBATase 基线, 修复前 1-based):
      F205G 实际算的是 prof.at[204] = **H204 催化组氨酸**的进化 profile(而非 F205);
      T57G 算的是 **F56 氧阴离子穴**; S126A 算的是 H125。当时据此写出的
      "F205G ΔΔG=−2.399, 进化上可接受" 结论是**错的**, 已作废。
      同时 site 1/2/3 因 ``prof.at[0]`` 越界 KeyError 直接丢值(基线 39 变体丢 3 个)。

    自检: score_variants 里对每个变体断言 ``profile.at[site-1, 'SEQ'] == wt``,
    这条断言把「索引基准错」变成硬失败, 杜绝再次静默。
    """
    import pandas as pd
    from dockbuddy_scoring.msa_stats import position_freqs

    L = len(reference)
    freqs = position_freqs(seqs, L)
    df = pd.DataFrame(freqs).fillna(0.0)
    for aa in _AA_CODES:
        if aa not in df.columns:
            df[aa] = 0.0
    df["SEQ"] = list(reference)
    # 0-based: 与 funcs_seq 的 prof.at[aa_pos] 口径一致(见上方长注释)
    df.index = pd.Index(range(0, L), name="POS")

    # 构造一个带 .profile 属性的轻量对象(ddgun_seq 只访问 profile.profile)
    class _Profile:
        pass
    p = _Profile()
    p.profile = df[_AA_CODES + ["SEQ"]]
    return p


def read_msa_seqs(path: str) -> Tuple[List[str], str]:
    """读 a3m/a2m/fasta, 返回 (aligned_seqs, reference)。小写插入丢弃。"""
    headers: List[str] = []
    raw: List[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                headers.append(line)
                raw.append("")
            else:
                if not raw:
                    headers.append(">seq0")
                    raw.append("")
                raw[-1] += line.strip()
    has_lower = any(c.islower() for s in raw for c in s)
    seqs = []
    for s in raw:
        if has_lower:
            s = "".join(c if c.isupper() else ("-" if c in "-." else "") for c in s)
        else:
            s = s.replace(".", "-")
        seqs.append(s)
    return seqs, seqs[0] if seqs else ""


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    """DDGun ΔΔG 打分, 返回 {variant_id: ddgun_ddg}。

    参数:
      wt_seq: WT 序列(1-letter)。
      variants: 可迭代 Variant / dict。
      msa: MSA 文件路径(a3m/a2m/fasta, 必填; DDGun 需进化 profile)。
    """
    msa_path = str(cfg.get("msa", "") or "").strip()
    if not msa_path:
        raise RuntimeError("DDGun 需要 msa(a3m/fasta; 用于构建进化 profile)")
    seqs, ref = read_msa_seqs(msa_path)
    if not seqs:
        raise RuntimeError("MSA 为空")

    ddgun_seq = _load_ddgun()
    seq = "".join(str(wt_seq).split())
    profile = build_profile(seqs, ref or seq)
    prof = profile.profile

    out: Dict[str, float] = {}
    for v in variants:
        vid, wt, mt, site = _fields(v)
        if not (1 <= site <= len(seq)):
            continue
        if seq[site - 1].upper() != wt.upper():
            LOG.warning("DDGun 跳过 %s: 位点 %d 与 WT 不符", vid, site)
            continue
        # 索引基准自检(2026-10-16): ddgun 内部取 prof.at[site-1]; 若 build_profile
        # 的 index 基准被改回 1-based, 这里会立刻 KeyError/断言失败, 而不是静默
        # 用前一位的进化 profile 出数(那正是 F205G 算出 H204 profile 的原因)。
        try:
            row_aa = str(prof.at[site - 1, "SEQ"]).upper()
        except KeyError as e:  # noqa: BLE001
            raise RuntimeError(
                f"DDGun profile 索引越界(site={site}): {e} —— build_profile 的 index "
                "必须是 0-based(range(0, L)), 与 funcs_seq 的 prof.at[aa_pos] 对齐"
            ) from e
        if row_aa != wt.upper():
            raise RuntimeError(
                f"DDGun profile 索引基准错: 变体 {vid}(site={site}, from={wt}) "
                f"取到 profile 第 {site - 1} 行的 SEQ={row_aa}, 与 WT 该位残基不符。"
                "根因是 build_profile 的 index 基准与 ddgun funcs_seq 的 "
                "prof.at[aa_pos](0-based) 不一致 —— 必须是 0-based。"
            )
        try:
            out[vid] = float(ddgun_seq(_Substitution(wt, site - 1, mt), profile))
        except Exception as e:  # noqa: BLE001
            LOG.warning("DDGun 打分失败 %s: %s", vid, e)
    return out


__all__ = ["score_variants", "build_profile", "read_msa_seqs"]
