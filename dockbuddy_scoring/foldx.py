# -*- coding: utf-8 -*-
"""
libscore_foldx.py —— FoldX 5.1 突变 ΔΔG 打分适配器(预筛管线「打分」步骤, 2026-10-08)
===================================================================================
定位: 预筛管线「打分」步骤的**真实可跑**适配器 —— 用本地 FoldX 5.1 二进制
(见 pipeline/libdock/foldx.py) 对每个单点突变做 BuildModel, 产出真实 ΔΔG
(kcal/mol), 输出 {variant_id: {foldx_ddg: float}}。与 esm1v 同挂 orchestrator
的 scoring.adapter。

关键差异(与 esm1v 序列打分不同):
  - FoldX 需要 **WT 结构 PDB**(不是序列), 由 manifest.scoring.receptor_pdb 指定。
    若需从序列得到结构, 可先用 folding.adapter=esmfold 折叠 WT 序列落盘, 再喂 FoldX。
  - 每个突变独立调用 libdock.foldx.foldx_ddg(内含 RepairPDB + BuildModel), 慢
    (单突变约几十秒~几分钟); 大规模预筛用 esm1v/scores_file, FoldX 作精选后精算。

诚实口径(沿用 foldx.py 纪律):
  - ΔΔG 是**稳定性代理**, 不是活性; 单点 ΔΔG 不得直接推活性升降。
  - 写入论文前须用 1~2 个有实验 ΔΔG 的突变体做方法学标定。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

LOG = logging.getLogger("dockpipeline")


def _fields(v) -> tuple:
    """从 Variant(有 .from_aa/.to_aa/.site/.chain/.id) 或 dict 提取 (id, from, to, site, chain)。"""
    if isinstance(v, dict):
        site = int(v["site"]); wt = str(v["from"]).upper(); mt = str(v["to"]).upper()
        vid = str(v.get("id") or v.get("variant_id") or f"{wt}{site}{mt}")
        chain = str(v.get("chain", ""))
        return vid, wt, mt, site, chain
    vid = str(getattr(v, "id", None) or getattr(v, "label", None) or "")
    return (vid, str(getattr(v, "from_aa")).upper(), str(getattr(v, "to_aa")).upper(),
            int(getattr(v, "site")), str(getattr(v, "chain", "")))


def score_variants(receptor_pdb: str, variants: Iterable[Any], *,
                   bin_path: str = "", chain: str = "A",
                   timeout: int = 600, workdir: str = "") -> Dict[str, Dict[str, float]]:
    """对一批单点突变跑 FoldX 5.1 BuildModel, 返回 {variant_id: {foldx_ddg: float}}。

    参数:
      receptor_pdb: WT 结构 PDB 绝对路径。
      variants: 可迭代, 每项为 libmutlib.Variant 或 dict{'site','from','to','chain'?}。
      bin_path: FoldX 二进制; 空则走 find_foldx_bin()(FOLDX_BIN 环境变量/探测)。
      chain: 缺省突变所在链(有 chain 的 variant 优先用自身 chain)。
    返回: 成功者 {variant_id: {foldx_ddg}}; 失败者被跳过(记 WARNING, 不写入)。
    """
    from dockbuddy_scoring.foldx_engine import FoldXError, foldx_ddg

    out: Dict[str, Dict[str, float]] = {}
    for v in variants:
        vid, wt, mt, site, vchain = _fields(v)
        ch = vchain or chain
        mutation = f"{wt}{ch}{site}{mt}"
        try:
            r = foldx_ddg(receptor_pdb, mutation, bin_path=bin_path or None,
                          chain=ch, workdir=workdir, timeout=timeout)
        except FoldXError as e:
            LOG.warning("foldx 打分失败 %s: %s", vid, e)
            continue
        if r.get("ok") and r.get("ddG_kcal") is not None:
            out[vid] = {"foldx_ddg": float(r["ddG_kcal"])}
        else:
            LOG.warning("foldx 打分失败 %s: %s", vid, r.get("note", "") or "无 ΔΔG")
    return out


__all__ = ["score_variants"]
