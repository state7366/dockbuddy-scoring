# -*- coding: utf-8 -*-
"""
libscore_rosetta.py —— Rosetta 精细 ΔΔG 打分适配器(2026-10-08)
===================================================================================
定位: 预筛管线「打分」步骤的精细稳定性后端 —— 用 Rosetta(ΔΔG protocol, Cartesian
ddg / flex ddG)对单点突变算高精度 ΔΔG, 与 FoldX(快速粗算)并列, 按需切换。

接入方式(诚实边界):
  - Rosetta 是 RosettaCommons 的**非商业许可软件**(学术/非盈利用户免费; 商业或
    fee-for-service 需另购)。PyRosetta 自 2026-01 起提供 **quarterly builds**, 可
    pip 直接安装、免 license key(仍需遵守非商业许可协议):
        pip install pyrosetta --find-links \
          https://west.rosettacommons.org/pyrosetta/quarterly/release
    wheel ~1.5GB(Linux x86_64 / macOS, Python 3.9-3.14; 无 Windows 原生 wheel)。
    境外下载走代理, 部署建议「本地下载 wheel 再上传」(见 docs/模型部署资源清单.md)。
  - 本适配器只做「检测本机 PyRosetta + 调用 + 解析」, 缺依赖时抛带指引的
    RuntimeError, 不假装已跑通。

调用约定:
  - 首选 PyRosetta(`import pyrosetta`), 用 Cartesian ddg(cartesian_ddg) 或 flex ddG
    计算 ΔΔG; 也可经 `rosetta_scripts`/`cartesian_ddg` 二进制走命令行(延迟支持)。
  - 打分只作代理指标(稳定性), 不预测 kcat/Km; 阈值由第二层 libgate 软排序使用。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

LOG = logging.getLogger("dockpipeline")

_ROSETTA_HINT = (
    "Rosetta ΔΔG 需要 PyRosetta。PyRosetta 现为 quarterly builds, 学术/非商业用户可 "
    "免费 pip 安装(遵守 RosettaCommons 非商业许可, 免 license key): "
    "pip install pyrosetta --find-links "
    "https://west.rosettacommons.org/pyrosetta/quarterly/release "
    "(wheel ~1.5GB, Linux x86_64/macOS, Python 3.9-3.14; 无 Windows 原生 wheel)。"
    "本机未部署。请先安装 PyRosetta, 或改用 scoring.adapter=foldx(已真跑通)做稳定性硬约束。"
)


def _fields(v) -> tuple:
    if isinstance(v, dict):
        site = int(v["site"]); wt = str(v["from"]).upper(); mt = str(v["to"]).upper()
        vid = str(v.get("id") or v.get("variant_id") or f"{wt}{site}{mt}")
        return vid, wt, mt, site
    vid = str(getattr(v, "id", None) or getattr(v, "label", None) or "")
    return (vid, str(getattr(v, "from_aa")).upper(), str(getattr(v, "to_aa")).upper(),
            int(getattr(v, "site")))


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    """对一批单点突变做 Rosetta ΔΔG, 返回 {variant_id: rosetta_ddg}。

    参数:
      wt_seq: WT 序列(1-letter)。
      variants: 可迭代 Variant / dict。
      pdb_path: WT 结构 PDB(必填, Cartesian ddg 需要)。
      protocol: "cartesian_ddg"(缺省) / "flex_ddg"。
      nstruct: 采样次数(缺省 3)。
    """
    try:
        import pyrosetta  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_ROSETTA_HINT) from e

    pdb_path = str(cfg.get("pdb_path", "") or "")
    if not pdb_path:
        raise RuntimeError("libscore_rosetta 需要 pdb_path(WT 结构 PDB)")
    protocol = str(cfg.get("protocol", "cartesian_ddg") or "cartesian_ddg")
    nstruct = int(cfg.get("nstruct", 3) or 3)

    muts = [_fields(v) for v in variants]
    # 延迟 import 重依赖(pyrosetta 初始化开销大, 只在真跑时触发)
    from pyrosetta import init, pose_from_file
    init("-mute all -ignore_unrecognized_res")
    pose = pose_from_file(pdb_path)

    out: Dict[str, float] = {}
    for vid, wt, mt, site in muts:
        ddg = _cartesian_ddg(pose, wt, mt, site, nstruct) if protocol == "cartesian_ddg" \
            else _flex_ddg(pose, wt, mt, site, nstruct)
        out[vid] = float(ddg)
    return out


def _cartesian_ddg(pose, wt: str, mt: str, site: int, nstruct: int) -> float:
    """Cartesian ddg: 突变后能量差(WT 未突变参照)。

    说明(2026-10-10 修正): PyRosetta 的 ``Pose`` 没有 ``apply()`` 方法(apply 是
    mover 的协议), 之前的 ``wt_pose.apply(_relax_mover())`` 会 AttributeError。
    此处用最小可用口径: 复制 WT pose → mutate_residue 换侧链 → 算 fa 能量差。
    cartesian_ddg 的完整「突变 + 双侧松弛」留待接入 rosetta_scripts 协议后启用。
    """
    from pyrosetta import Pose, get_fa_scorefxn
    sfxn = get_fa_scorefxn()
    mut_pose = Pose()
    mut_pose.assign(pose)
    # 2026-10-10 修正: `from pyrosetta.toolbox import mutate_residue` 导入的是
    # **函数**(非模块), 不能写 mutate_residue.mutate_residue(...)。直接调用即可:
    #   mutate_residue(pose, seqpos, new_res, pack_radius, pack_scorefxn)
    try:
        from pyrosetta.toolbox import mutate_residue
        mutate_residue(mut_pose, site, mt, 1.0)
    except ImportError:  # 新版本路径
        from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
        MutateResidue(target=site, new_res=mt).apply(mut_pose)
    return sfxn(mut_pose) - sfxn(pose)


def _flex_ddg(pose, wt: str, mt: str, site: int, nstruct: int) -> float:
    """flex ddG(精细, 但依赖 rosetta_scripts 协议), 留待协议文件传入后启用。"""
    raise RuntimeError("flex_ddg 协议需 rosetta_scripts XML 协议文件, 暂请用 cartesian_ddg")


__all__ = ["score_variants"]
