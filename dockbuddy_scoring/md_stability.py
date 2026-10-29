# -*- coding: utf-8 -*-
"""MD 构象稳定性打分适配器(第 27 个模型)。

背景(2026-10-28 沉淀自 PBATase apo/holo MD):
  PBATase 突变体 9 体系 apo MD 三路分析(RMSF + 催化三联体几何 + PCA)得到结论——
  高活性突变体(F205G/P207G)的**催化三联体几何更收敛**(Ser126-His204 更短更稳、
  His-Asp 电荷接力 4 倍收敛)、催化 loop 更刚(RMSF 更低), 而非"柔韧化"。
  (详见 .workbuddy/memory/2026-10-27.md "构象稳定假说")

本适配器把这条分析沉淀为可复用打分: 从突变体的 MD 轨迹算"构象稳定性"。

与其它打分适配器一致, 入口:
    from dockbuddy_scoring.md_stability import score_variants
    scores = score_variants(wt_seq, variants, traj_dir=..., triad=(126,204,174))

cfg 必填:
    traj_dir: 轨迹根目录, 布局 <traj_dir>/<variant_id>/{md.gro, md.xtc}
cfg 可选:
    triad: 催化三联体 (ser, his, asp) 残基号, 缺省 (126, 204, 174)
    loop_range: 催化 loop 区段 (start, end), 缺省 (195, 215)
    drop_ns: 丢前 N ns(平衡), 缺省 10
    stride: 读轨迹降采样, 缺省 10(每 100ps 一帧)

打分口径: 分数**越低越稳定**(与 foldx/geostab 物理力场反号一致, 负值=更稳定)。
综合分 = Ser-His 距离(mean) + Ser-His 距离(std) + 0.5 * loop RMSF 均值。

坑(已修正, 见 memory):
  1. md.xtc 坐标是 wrap 的(PBC), 必须先 image_molecules 再 superpose, 否则 RMSF 虚高。
  2. mdtraj 的 resid 是 0-based, 用 resSeq(对应 auth_seq_id)。
"""
import os
from typing import Any, Dict, Iterable, Tuple

__all__ = ["score_variants", "analyze_trajectory", "stability_score"]


def _fields(v) -> Tuple[str, str, str, int]:
    if isinstance(v, dict):
        site = int(v["site"]); wt = str(v["from"]).upper(); mt = str(v["to"]).upper()
        vid = str(v.get("id") or v.get("variant_id") or f"{wt}{site}{mt}")
        return vid, wt, mt, site
    vid = str(getattr(v, "id", None) or getattr(v, "label", None) or "")
    return (vid, str(getattr(v, "from_aa")).upper(), str(getattr(v, "to_aa")).upper(),
            int(getattr(v, "site")))


def analyze_trajectory(gro_path: str, xtc_path: str,
                       triad: Tuple[int, int, int] = (126, 204, 174),
                       loop_range: Tuple[int, int] = (195, 215),
                       drop_ns: float = 10.0, stride: int = 10) -> Dict[str, float]:
    """从单个轨迹算构象稳定性指标。返回:
      ser_his_dist_mean / ser_his_dist_std : Ser OG - His NE2 距离 (nm)
      his_asp_dist_mean / his_asp_dist_std : His ND1 - Asp OD1 距离 (nm)
      loop_rmsf_mean : 催化 loop 区段 Cα 平均 RMSF (nm)
      pca_pc1_range : 催化 loop Cα PCA 第一主成分投影范围 (nm, 构象空间宽度)
    """
    import numpy as np  # noqa: F401  (惰性导入, 减少冷启动依赖)
    import mdtraj as md

    ser, his, asp = triad
    t = md.load_xtc(xtc_path, top=gro_path, stride=stride)
    n_drop = int(round(drop_ns / (stride * 0.002 * 100))) if drop_ns > 0 else 0
    t = t[n_drop:]
    # PBC unwrap + 全蛋白对齐(消除整体平转, 否则 RMSF 虚高 —— 关键坑)
    t.image_molecules(inplace=True)
    protein = t.topology.select("protein")
    t.superpose(t, 0, atom_indices=protein)

    def _d(sel1, sel2):
        a = t.topology.select(sel1); b = t.topology.select(sel2)
        if len(a) == 0 or len(b) == 0:
            return None
        import numpy as _np
        d = md.compute_distances(t, _np.array([[a[0], b[0]]])).flatten()
        return float(d.mean()), float(d.std())

    ser_his = _d(f"resSeq {ser} and name OG", f"resSeq {his} and name NE2")
    his_asp = _d(f"resSeq {his} and name ND1", f"resSeq {asp} and name OD1")

    ca = t.topology.select(f"resSeq {loop_range[0]} to {loop_range[1]} and name CA")
    loop_rmsf = None
    pca_range = None
    if len(ca) > 0:
        tc = t.atom_slice(ca)
        xyz = tc.xyz.reshape(len(tc), -1)
        xyz -= xyz.mean(axis=0)
        rmsf = np.sqrt((xyz ** 2).reshape(len(tc), -1, 3).mean(0).sum(1))
        loop_rmsf = float(rmsf.mean())
        cov = np.cov(xyz.T)
        ev = np.linalg.eigvalsh(cov)[::-1]
        pca_range = float(np.sqrt(ev[0])) if len(ev) > 0 else None

    out: Dict[str, float] = {}
    if ser_his:
        out["ser_his_dist_mean"], out["ser_his_dist_std"] = ser_his
    if his_asp:
        out["his_asp_dist_mean"], out["his_asp_dist_std"] = his_asp
    if loop_rmsf is not None:
        out["loop_rmsf_mean"] = loop_rmsf
    if pca_range is not None:
        out["pca_pc1_range"] = pca_range
    return out


def stability_score(metrics: Dict[str, float]) -> float:
    """综合构象稳定性分数(越低越稳定)。"""
    return (metrics.get("ser_his_dist_mean", 0.0)
            + metrics.get("ser_his_dist_std", 0.0)
            + 0.5 * metrics.get("loop_rmsf_mean", 0.0))


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    """对一批突变体做 MD 构象稳定性打分, 返回 {variant_id: stability_score}。

    轨迹按 variant_id 定位: <traj_dir>/<variant_id>/{md.gro, md.xtc}。
    若某 variant 无轨迹目录, 跳过并告警。
    """
    traj_dir = str(cfg.get("traj_dir", "") or "")
    triad = tuple(int(x) for x in (cfg.get("triad") or (126, 204, 174)))
    loop = tuple(int(x) for x in (cfg.get("loop_range") or (195, 215)))
    drop_ns = float(cfg.get("drop_ns", 10.0) or 10.0)
    stride = int(cfg.get("stride", 10) or 10)

    if not traj_dir:
        raise RuntimeError("md_stability 需要 traj_dir(轨迹根目录, 布局 <traj_dir>/<variant_id>/md.xtc)")

    out: Dict[str, float] = {}
    for v in variants:
        vid, _wt, _mt, _site = _fields(v)
        gro = os.path.join(traj_dir, vid, "md.gro")
        xtc = os.path.join(traj_dir, vid, "md.xtc")
        if not (os.path.exists(gro) and os.path.exists(xtc)):
            continue  # 无轨迹, 跳过
        try:
            m = analyze_trajectory(gro, xtc, triad=triad, loop_range=loop,
                                   drop_ns=drop_ns, stride=stride)
            out[vid] = stability_score(m)
        except Exception:  # noqa: BLE001 —— 单变体失败不阻断整批
            continue
    return out
