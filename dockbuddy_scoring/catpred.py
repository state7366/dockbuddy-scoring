# -*- coding: utf-8 -*-
"""
libscore_catpred.py —— CatPred kcat 活性打分适配器(2026-10-11 新增)
===================================================================================
定位: 给打分矩阵补一个**活性维度**(现有 esm1v/foldx/saprot 等都是结合/稳定/进化
代理, 不预测活性)。用 CatPred(Nat. Commun. 2025, maranasgroup/CatPred)对
「酶序列 × 底物 SMILES」预测 kcat(log10), 输出 {variant_id: catpred_kcat}。

接入方式(独立 repo 薄适配器, 与 geostab/evmutation 同款):
  - 仓库 github.com/maranasgroup/CatPred; 权重随仓(需按 README 下载)。
  - 依赖: torch + 其自身的 GNN/特征提取依赖。真跑需 GPU 实例 + repo + 权重。

诚实边界(重要, 必须遵守):
  - **kcat 是绝对值预测, 不是 Δ活性**: CatPred 输出「某序列对某底物的 kcat 预测」,
    不直接给「突变导致的 kcat 变化」。用于突变扫描时, 应对每个突变体序列分别预测
    kcat, 与 WT 预测值做**差值**才能谈「突变效应」。
  - **外推风险**: kcat 预测对训练集外酶(<40% 序列相似)误差大(文献 R²~0.17);
    PBATase 是 AlphaFold 预测的新酶, 覆盖不足, 预测仅作参考维度, 不作硬判据。
  - 已否决 DLKcat(被 Kroll & Lercher 2024 证伪, PMID 39346751 对突变/陌生酶无意义);
    CatPred 带逐样本不确定性, 是当前较稳的选择, 但仍需防外推。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

LOG = logging.getLogger("dockpipeline")

_CATPRED_HINT = (
    "CatPred kcat 打分需要其官方仓库(github.com/maranasgroup/CatPred, 克隆后 "
    "--repo-dir 指向它) + 预训练 checkpoint(capsule_data_update.tar.gz ~10.2GB, "
    "下载 https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz) + "
    "conda env(environment.yml, 含 rdkit 等)。官方推荐 Linux + GPU 环境。"
    "本机(Windows/CPU)不部署; 请在 Linux GPU 实例按 README 部署后运行。"
    "注意: kcat 是绝对值预测(需与 WT 预测值做差才是突变效应), 且对陌生酶外推误差大。"
)


def _fields(v) -> tuple:
    if isinstance(v, dict):
        site = int(v["site"]); wt = str(v["from"]).upper(); mt = str(v["to"]).upper()
        vid = str(v.get("id") or v.get("variant_id") or f"{wt}{site}{mt}")
        return vid, wt, mt, site
    vid = str(getattr(v, "id", None) or getattr(v, "label", None) or "")
    return (vid, str(getattr(v, "from_aa")).upper(), str(getattr(v, "to_aa")).upper(),
            int(getattr(v, "site")))


def _mutated_sequence(wt_seq: str, wt: str, mt: str, site: int) -> str:
    i = site - 1
    if not (0 <= i < len(wt_seq)):
        raise ValueError(f"位点 {site} 越界(WT 序列长度 {len(wt_seq)})")
    if wt_seq[i].upper() != wt.upper():
        raise ValueError(f"位点 {site} 与 WT 不符: 序列[{i}]={wt_seq[i]} != {wt}")
    return wt_seq[:i] + mt + wt_seq[i + 1:]


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    """对一批突变体的 kcat 预测, 返回 {variant_id: catpred_kcat(log10)}。

    参数:
      wt_seq: WT 序列(1-letter)。
      variants: 可迭代 Variant / dict。
      substrate_smiles: 底物 SMILES(必填; kcat 依赖底物, 与结合/稳定打分不同)。
      repo_dir: CatPred 仓库路径(必填, 含 catpred 包与 demo_run.py)。
      checkpoint_dir: 预训练 checkpoint 目录(必填; capsule_data_update 解压目录)。
      structure_dir: 结构 PDB 目录(必填), 须含 WT.pdb; 每个变体可选 <variant_id>.pdb,
        缺失时回退 WT 结构近似(CatPred 的 pdbpath 列需要结构)。
      use_gpu: 是否用 GPU(缺省 False)。
      include_wt: 是否在结果里附 WT 的 log10kcat(键 "WT"; 缺省 True)。
    返回: {variant_id: Δlog10kcat}(突变效应 = log10kcat(variant) − log10kcat(WT))。
    """
    import csv
    import os
    import sys
    import tempfile

    repo_dir = str(cfg.get("repo_dir", "") or "")
    checkpoint_dir = str(cfg.get("checkpoint_dir", "") or "")
    substrate_smiles = str(cfg.get("substrate_smiles", "") or "").strip()
    structure_dir = str(cfg.get("structure_dir", "") or "")
    use_gpu = bool(cfg.get("use_gpu", False))

    missing = [k for k, v in (
        ("repo_dir", repo_dir), ("checkpoint_dir", checkpoint_dir),
        ("substrate_smiles", substrate_smiles), ("structure_dir", structure_dir)) if not v]
    if missing:
        raise RuntimeError(
            "CatPred 需要 repo_dir + checkpoint_dir + substrate_smiles + structure_dir; "
            f"缺失: {', '.join(missing)}")

    # CatPred 仓库启用了 checkpoint 反序列化信任根校验: 仅允许从受信任根目录加载 .pt,
    # 默认信任根只有 repos/(源码目录)。我们的权重经 tarball 解压到 checkpoint_dir,
    # 位于 MODELS_ROOT/catpred/... 之下, 不在默认信任根内 → 需把模型根目录加入
    # CATPRED_TRUSTED_DESERIALIZATION_ROOTS(官方提供的扩展点; 权重由我们自己的 COS 拉取,
    # 加入信任根是合理的)。checkpoint_dir = <catpred根>/data/pretrained/production/kcat,
    # 上溯 4 级即得 catpred 根目录。
    _catpred_root = checkpoint_dir
    for _ in range(4):
        _catpred_root = os.path.dirname(_catpred_root)
    _roots = [r for r in os.environ.get("CATPRED_TRUSTED_DESERIALIZATION_ROOTS", "")
              .split(os.pathsep) if r]
    if _catpred_root and _catpred_root not in _roots:
        _roots.append(_catpred_root)
        os.environ["CATPRED_TRUSTED_DESERIALIZATION_ROOTS"] = os.pathsep.join(_roots)

    # checkpoint 未解压时, 从 tarball(可选)自动解压到 catpred 根目录(解压出 data/ 结构)
    tarball = str(cfg.get("tarball", "") or "")
    if tarball and os.path.isfile(tarball) and not os.path.isdir(checkpoint_dir):
        import tarfile
        # checkpoint_dir = <catpred>/data/pretrained/production/kcat → 上 4 级 = catpred 根
        extract_root = checkpoint_dir
        for _ in range(4):
            extract_root = os.path.dirname(extract_root)
        LOG.info("CatPred: 解压 checkpoint %s -> %s ...", tarball, extract_root)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(extract_root)
    if not os.path.isdir(checkpoint_dir):
        raise RuntimeError(
            f"CatPred checkpoint 缺失: {checkpoint_dir}(需解压 capsule_data_update.tar.gz "
            "到 catpred 目录, 或经 tarball 参数自动解压)")

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        # 官方 API(与 demo_run.py 同款): CatPred 无 `catpred.predict` 顶层模块,
        # 真实入口是 catpred.inference.PredictionRequest + run_prediction_pipeline。
        from catpred.inference import PredictionRequest, run_prediction_pipeline
    except ImportError as e:
        raise RuntimeError(_CATPRED_HINT) from e

    seq = "".join(str(wt_seq).split())
    fields = [_fields(v) for v in variants]

    # CatPred 输入 CSV 需 SMILES/sequence/pdbpath 三列(pdbpath=每突变体结构)。
    # CatPred 要求每 sequence 唯一 pdbpath(service.py conflicting_pdbpaths 检查),
    # 故缺某变体结构时**跳过该变体**(记录 warning), 不回退 WT 结构(回退会致多序列
    # 共享同一 pdbpath 而报错)。
    wt_pdb = os.path.join(structure_dir, "WT.pdb")
    if not os.path.isfile(wt_pdb):
        raise RuntimeError(
            f"CatPred 需要 WT 结构 {wt_pdb}(structure_dir 下须有 WT.pdb)")

    def _pdb_for(vid: str):
        cand = os.path.join(structure_dir, f"{vid}.pdb")
        return cand if os.path.isfile(cand) else None

    rows = [{"Substrate": "WT", "SMILES": substrate_smiles,
             "sequence": seq, "pdbpath": wt_pdb}]
    scored_fields = []
    for vid, wt, mt, site in fields:
        pdb = _pdb_for(vid)
        if not pdb:
            LOG.warning("CatPred 跳过 %s: 缺结构 %s(CatPred 需每序列唯一 pdbpath)",
                        vid, os.path.join(structure_dir, f"{vid}.pdb"))
            continue
        rows.append({"Substrate": vid, "SMILES": substrate_smiles,
                     "sequence": _mutated_sequence(seq, wt, mt, site),
                     "pdbpath": pdb})
        scored_fields.append((vid, wt, mt, site))

    with tempfile.TemporaryDirectory() as td:
        in_csv = os.path.join(td, "input.csv")
        with open(in_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Substrate", "SMILES", "sequence", "pdbpath"])
            w.writeheader()
            w.writerows(rows)
        out_dir = os.path.join(td, "results")
        os.makedirs(out_dir, exist_ok=True)
        request = PredictionRequest(
            parameter="kcat", input_file=in_csv, checkpoint_dir=checkpoint_dir,
            use_gpu=use_gpu, repo_root=repo_dir)
        try:
            final_output = run_prediction_pipeline(request=request, results_dir=out_dir)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"CatPred 预测失败: {e}") from e

        # 解析输出(须在 with 块内: TemporaryDirectory 退出即清理, 块外读会 FileNotFoundError)
        import pandas as pd
        df = pd.read_csv(final_output)
        col = next((c for c in ("log10kcat_max", "Prediction_log10") if c in df.columns), None)
        if col is None:
            raise RuntimeError(f"CatPred 输出缺 log10kcat 列(现有: {list(df.columns)})")
        log10 = {str(r["sequence"]): float(r[col]) for _, r in df.iterrows()}

    wt_log10 = log10.get(seq)
    if wt_log10 is None:
        raise RuntimeError("CatPred 输出缺少 WT 序列行, 无法计算突变效应")

    out: Dict[str, float] = {}
    if cfg.get("include_wt", True):
        out["WT"] = round(wt_log10, 6)
    for vid, wt, mt, site in scored_fields:
        ms = _mutated_sequence(seq, wt, mt, site)
        if ms in log10:
            out[vid] = round(log10[ms] - wt_log10, 6)  # Δlog10kcat = 突变效应
    return out


__all__ = ["score_variants", "_mutated_sequence"]
