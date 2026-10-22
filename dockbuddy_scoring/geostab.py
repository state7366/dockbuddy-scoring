# -*- coding: utf-8 -*-
"""
libscore_geostab.py —— GeoStab-suite(GeoDDG/GeoDTm/GeoFitness) 打分适配器(2026-10-08, 2026-10-09 重写)
============================================================================================================
定位: 预筛管线「打分」步骤的几何学习稳定性/适应度后端 —— 用 GeoStab-suite
(Nat Comput Sci 2024, Gonglab-THU)预测突变的 ΔΔG(GeoDDG)与 ΔTm(GeoDTm)。
同时支持 **Seq**(纯序列) 与 **threeD**(序列+结构) 两版(用户要求全量接入, 不只 -Seq)。

接入方式(无托管 API, 本地仓库 + 重依赖):
  - 仓库 github.com/Gonglab-THU/GeoStab(含 model_ddG_Seq/3D、model_dTm_Seq/3D、
    model_fitness_Seq/3D 等模型目录; 权重随仓, 无需另下)。
  - 依赖: python 3.9/3.10 + torch 1.13 + biopandas + biopython + click + pdb-tools +
    ESM-1v/ESM-2(esm2_t33_650M_UR50D)特征 + FoldX 5.0。**版本差异(命名反直觉, 以
    run.sh 为准)**: ``-v Seq`` 输入只有序列 → 内部需 **hhblits(UniRef30) + AlphaFold2**
    预测结构; ``-v threeD`` 输入为结构 PDB → **不需 AlphaFold/MSA**, 只需 FoldX +
    结构 PDB(用户自备, 如 AF3 结构)。

调用约定(已核对官方 run.sh / generate_example_mut_info_csv.py / model_*_3D/predict.py):
  - run.sh 一次只处理**一个突变**: ``mut_info.csv`` 是官方 11 列、单行(index 固定
    'test')的格式, ``mut_pos`` 为 **0-based 序列索引**。因此本适配器对每个突变生成
    独立 out_dir, 逐个跑 run.sh(与 FoldX 适配器「每个突变独立调用」同款口径)。
  - 输入文件(run.sh 真正读取的)由本适配器生成: ``wt_data/result.fasta``、
    ``mut_data/result.fasta``、``mut_data/individual_list.txt``(FoldX BuildModel
    mutant-file, 残基号 **1-based** 与结构 PDB 一致); threeD 版还需把 WT 结构放到
    ``wt_data/relaxed.pdb``(run.sh 的 FoldX RepairPDB 输入)。
  - 输出: ``result_ddG.txt`` / ``result_dTm.txt``(单 float 文本)、
    ``result_fitness.csv``(20×L 矩阵) —— 都在 ``-o`` 目录根下。

诚实边界: 本适配器是**官方仓库的 subprocess 包装** + 输出解析, 不重实现模型;
真跑需 GPU 实例部署 GeoStab + ESM 权重 + FoldX(threeD) + conda geostab 环境。
打分只作代理指标。threeD 版要求 ``structure_pdb`` 为**仅含目标蛋白链的 ATOM-only
PDB**, 残基号 1-based 连续且与 variant.site 一致、残基序列与 wt_seq 一致。
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, Iterable, Optional, Tuple

LOG = logging.getLogger("dockpipeline")


def _conda_bin_dir() -> str:
    """定位 conda 的 bin 目录(用于给 run.sh 子进程注入 PATH)。

    GeoStab 官方 run.sh 顶部是 ``set -Eeuo pipefail`` 且第一件事就 ``conda shell.bash
    hook`` + ``conda activate geostab``。当流水线经非登录 shell(subprocess)调用时,
    ``conda`` 不在 PATH, ``conda shell.bash hook`` 返回 127 触发 set -e 立即退出
    (stderr 又被 2>/dev/null 吞掉, 表现为「空输出 + rc=127」)。故必须在调用前把
    conda bin 目录放进 PATH。返回空串表示未找到(调用方回退原环境)。
    """
    exe = os.environ.get("CONDA_EXE", "")
    if exe and os.path.isfile(exe):
        d = os.path.dirname(exe)
        if os.path.basename(d) == "bin":
            return d
    p = shutil.which("conda")
    if p:
        return os.path.dirname(p)
    for cand in ("/opt/miniconda3/bin", "/opt/conda/bin", "/usr/local/miniconda3/bin",
                 os.path.expanduser("~/miniconda3/bin"),
                 os.path.expanduser("~/anaconda3/bin")):
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "conda")):
            return cand
    return ""


def _conda_env() -> Dict[str, str]:
    """返回把 conda bin 目录塞进 PATH 的子进程环境(供 run.sh 调用)。"""
    env = dict(os.environ)
    d = _conda_bin_dir()
    if d:
        env["PATH"] = d + os.pathsep + env.get("PATH", "")
    return env

# 官方 mut_info.csv 的 11 列(不含 index 列; 首列空字符串为 index 列名, 行 index='test')
_MUTINFO_COLUMNS = ["metric", "version", "pdb_id", "model_id", "chain_id",
                    "pdb_seq", "seq", "mut_pos", "mut_res", "pH", "temperature"]

_GEOSTAB_HINT = (
    "GeoStab 打分需要其官方仓库(github.com/Gonglab-THU/GeoStab) + torch 1.13 + "
    "biopandas/biopython/click/pdb-tools + ESM-1v/ESM-2 权重 + FoldX 5.0。"
    "版本差异: -v Seq 需 hhblits(UniRef30)+AlphaFold2 预测结构; -v threeD 不需 "
    "AF2/MSA, 只需 FoldX + 结构 PDB(threeD 版另传 structure_pdb)。"
    "本机未部署。请在 GPU 实例按仓库 README 部署 conda geostab 环境后运行, "
    "或改用 scoring.adapter=foldx/rosetta。"
)


def _fields(v) -> Tuple[str, str, str, int]:
    """从 Variant / dict 提取 (id, from_aa, to_aa, site)。site 为 1-based。"""
    if isinstance(v, dict):
        site = int(v["site"]); wt = str(v["from"]).upper(); mt = str(v["to"]).upper()
        vid = str(v.get("id") or v.get("variant_id") or f"{wt}{site}{mt}")
        return vid, wt, mt, site
    vid = str(getattr(v, "id", None) or getattr(v, "label", None) or "")
    return (vid, str(getattr(v, "from_aa")).upper(), str(getattr(v, "to_aa")).upper(),
            int(getattr(v, "site")))


def _norm_version(version: str) -> str:
    """用户口径 'threeD'/'3d'/'Seq'/'seq' → 官方口径 '3D'/'Seq'。"""
    v = str(version or "threeD").strip().lower()
    return "Seq" if v in ("seq", "sequence") else "3D"


def _write_mutinfo(path: str, wt_seq: str, variant: Tuple[str, str, str, int], *,
                   version: str = "3D", mode: str = "ddGdTm",
                   pdb_seq: str = "", chain_id: str = "A",
                   pH: float = 7.0, temperature: float = 25.0) -> None:
    """写 GeoStab 官方 11 列单行 mut_info.csv(index='test')。

    variant = (vid, wt, mt, site), site 为 **1-based**; 内部换算 mut_pos = site-1
    (0-based 序列索引)。version 用官方口径 '3D'/'Seq'。
    """
    vid, wt, mt, site = variant
    mut_pos = int(site) - 1
    ver = _norm_version(version)
    metric = "fitness" if str(mode).lower() == "fitness" else "ddGdTm"
    if ver == "3D":
        pdb_seq = pdb_seq or wt_seq
        seq = ""
    else:
        pdb_seq = ""
        seq = wt_seq
    values = [metric, ver, "", "", chain_id, pdb_seq, seq,
              str(mut_pos), mt, str(int(pH)), str(int(temperature))]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + _MUTINFO_COLUMNS)
        w.writerow(["test"] + values)


def _write_inputs(out_dir: str, wt_seq: str, variant: Tuple[str, str, str, int], *,
                  version: str = "3D", chain_id: str = "A",
                  structure_pdb: str = "") -> str:
    """生成 run.sh 真正读取的输入, 返回 (wt_dir, mut_dir)。

    1. ``wt_data/result.fasta`` / ``mut_data/result.fasta``(WT/突变体序列)。
    2. ``mut_data/individual_list.txt``(FoldX BuildModel mutant-file; 残基号
       **1-based** 与结构 PDB 一致, 链固定 'A' 与官方 run.sh 口径相同)。
    3. threeD 版: 把 ``structure_pdb`` 复制为 ``wt_data/relaxed.pdb``。
    """
    vid, wt, mt, site = variant
    mut_pos = int(site) - 1
    if not (0 <= mut_pos < len(wt_seq)):
        raise ValueError(f"位点 {site} 越界(WT 序列长度 {len(wt_seq)})")
    if wt_seq[mut_pos].upper() != wt.upper():
        raise ValueError(
            f"位点 {site} 与 WT 序列不符: 序列[{mut_pos}]={wt_seq[mut_pos]} != {wt}")
    mut_seq = wt_seq[:mut_pos] + mt + wt_seq[mut_pos + 1:]

    wt_dir = os.path.join(out_dir, "wt_data")
    mut_dir = os.path.join(out_dir, "mut_data")
    os.makedirs(wt_dir, exist_ok=True)
    os.makedirs(mut_dir, exist_ok=True)

    with open(os.path.join(wt_dir, "result.fasta"), "w", encoding="utf-8") as f:
        f.write(f">result\n{wt_seq}")
    with open(os.path.join(mut_dir, "result.fasta"), "w", encoding="utf-8") as f:
        f.write(f">result\n{mut_seq}")
    # FoldX mutant-file: "{wt}A{site}{mt};"  —— 链 'A'、残基号 1-based
    with open(os.path.join(mut_dir, "individual_list.txt"), "w", encoding="utf-8") as f:
        f.write(f"{wt}A{site}{mt};")

    if _norm_version(version) == "3D":
        if not structure_pdb or not os.path.exists(structure_pdb):
            raise RuntimeError(
                "threeD 版需要 structure_pdb(WT 结构 PDB 绝对路径, 仅目标链 ATOM-only, "
                "残基号 1-based 与 variant.site 一致)。")
        shutil.copyfile(structure_pdb, os.path.join(wt_dir, "relaxed.pdb"))
    return wt_dir, mut_dir


def _parse_ddg_dtm(out_dir: str, metric: str) -> Optional[float]:
    """读官方 result_ddG.txt / result_dTm.txt(单 float 文本)。"""
    fname = "result_ddG.txt" if metric == "ddg" else "result_dTm.txt"
    p = os.path.join(out_dir, fname)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            return float(f.read().strip())
    except (ValueError, TypeError):
        return None


def _parse_fitness(out_dir: str, mt: str, mut_pos: int) -> Optional[float]:
    """读官方 result_fitness.csv(20×L, index=20 AA, columns=序列位置)。

    突变体 fitness = 该矩阵 [mt, mut_pos] 处的值(mut_pos 为 0-based 序列索引)。
    """
    p = os.path.join(out_dir, "result_fitness.csv")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            rows = list(csv.reader(f))
    except Exception:  # noqa: BLE001
        return None
    if len(rows) < 2:
        return None
    ncols = len(rows[1])
    if mut_pos + 1 >= ncols:
        return None
    for row in rows[1:]:
        if row and row[0].strip() == mt:
            try:
                return float(row[1 + mut_pos])
            except (ValueError, IndexError):
                return None
    return None


def _parse_output(out_dir: str, variant: Tuple[str, str, str, int],
                  metric: str) -> Optional[float]:
    """从官方输出文件解析单个突变的预测值(ddg/dtm/fitness)。"""
    vid, wt, mt, site = variant
    mut_pos = int(site) - 1
    if metric in ("ddg", "dtm"):
        return _parse_ddg_dtm(out_dir, metric)
    return _parse_fitness(out_dir, mt, mut_pos)


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    """用 GeoStab(GeoDDG/GeoDTm)对一批单点突变打分, 返回 {variant_id: ddg|dtm|fitness}。

    参数(cfg):
      geostab_dir: GeoStab 仓库路径(必填)。
      version: "threeD"(缺省, 需 structure_pdb) / "Seq"(纯序列, 需实例端 hhblits+AF2)。
      mode: "ddGdTm"(缺省) / "fitness"。
      metric: 返回 ddg 或 dtm(缺省 ddg; mode=fitness 时返回 fitness)。
      structure_pdb: threeD 版 WT 结构 PDB 绝对路径(仅目标链 ATOM-only, 残基号
        1-based 与 variant.site 一致、序列与 wt_seq 一致)。
      pdb_seq: threeD 版结构实际序列(缺省 wt_seq; 用于 mut_info.csv 的 pdb_seq 列)。
      chain_id: 缺省 "A"。
      out_dir: 输出根目录(缺省 geostab_dir 下临时目录); 每突变一子目录 <variant_id>/。
      pH / temperature: 缺省 7.0 / 25.0。
    """
    geostab_dir = str(cfg.get("geostab_dir", "") or "")
    run_sh = os.path.join(geostab_dir, "run.sh")
    if not geostab_dir or not os.path.exists(run_sh):
        raise RuntimeError(_GEOSTAB_HINT)

    version = _norm_version(cfg.get("version", "threeD"))
    mode = str(cfg.get("mode", "ddGdTm") or "ddGdTm")
    metric = str(cfg.get("metric", "ddg") or "ddg")
    if mode.lower() == "fitness":
        metric = "fitness"
    pH = float(cfg.get("pH", 7.0) or 7.0)
    temperature = float(cfg.get("temperature", 25.0) or 25.0)
    chain_id = str(cfg.get("chain_id", "A") or "A")
    pdb_seq = str(cfg.get("pdb_seq", "") or "")
    structure_pdb = str(cfg.get("structure_pdb", "")
                        or cfg.get("receptor_pdb", "") or "")
    out_root = str(cfg.get("out_dir", "") or os.path.join(geostab_dir, "_dockbuddy_out"))

    muts = [_fields(v) for v in variants]
    if not muts:
        return {}
    if version == "3D" and not structure_pdb:
        raise RuntimeError(
            "threeD 版需要 structure_pdb(WT 结构 PDB 绝对路径); 或改用 version='Seq'。")

    out: Dict[str, float] = {}
    for variant in muts:
        vid = variant[0]
        out_dir = os.path.join(out_root, vid)
        os.makedirs(out_dir, exist_ok=True)
        try:
            _write_mutinfo(os.path.join(out_dir, "mut_info.csv"), wt_seq, variant,
                           version=version, mode=mode, pdb_seq=pdb_seq,
                           chain_id=chain_id, pH=pH, temperature=temperature)
            _write_inputs(out_dir, wt_seq, variant, version=version,
                          chain_id=chain_id, structure_pdb=structure_pdb)
        except (ValueError, RuntimeError) as e:
            LOG.warning("GeoStab 输入准备失败 %s: %s", vid, e)
            continue

        cmd = ["bash", "run.sh", "-m", mode, "-v", version, "-o", out_dir]
        LOG.info("GeoStab: %s (variant=%s)", " ".join(cmd), vid)
        try:
            # 注入 conda 到 PATH(run.sh 依赖 conda activate geostab, 见 _conda_env)
            # 坑: run.sh 内部用 `&` 后台跑 esm2/esm1v embedding, 后台进程继承 stdout/stderr
            # 管道 → capture_output 会一直等管道关闭(卡到 timeout)。故 stdout/stderr 落盘,
            # 不 capture; start_new_session 让后台进程进独立 session, run.sh 退出即返回。
            logf = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8")
            proc = subprocess.run(cmd, cwd=geostab_dir, stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=3600, env=_conda_env(), start_new_session=True)
            logf.close()
        except subprocess.TimeoutExpired:
            LOG.warning("GeoStab 超时 %s(>3600s)", vid)
            continue
        if proc.returncode != 0:
            try:
                tail = open(os.path.join(out_dir, "run.log"), encoding="utf-8",
                            errors="ignore").read()[-2000:]
            except Exception:  # noqa: BLE001
                tail = ""
            LOG.warning("GeoStab 失败 %s:\n%s", vid, tail)
            continue
        val = _parse_output(out_dir, variant, metric)
        if val is None:
            LOG.warning("GeoStab 输出未解析到 %s(%s); 目录: %s", metric, vid, out_dir)
            continue
        out[vid] = val
    return out


__all__ = ["score_variants", "_write_mutinfo", "_write_inputs", "_parse_output"]
