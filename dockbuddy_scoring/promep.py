# -*- coding: utf-8 -*-
"""
libscore_promep.py —— ProMEP(多模态零样本) 打分适配器(2026-10-08)
===================================================================================
定位: 预筛管线「打分」步骤的多模态适配器 —— 用 ProMEP(中科大 wenjiegroup, Apache-2.0)
对单点突变做零样本打分, 输出 {variant_id: {promep_score: float}}。ProMEP 融合序列
(ESM 嵌入)+结构(SE(3)-Transformer 几何)表示做突变效应预测, 是 ESM-1v/SaProt 之外的
第三类打分路线。

接入方式(经联网核实, 无托管 API, 无 pip, 需 conda + 重依赖):
  - 仓库 github.com/wenjiegroup/ProMEP(Apache-2.0)
  - 关键依赖: SE(3)-Transformers(NVIDIA 几何深度学习) + ESM 嵌入, 较重。
  - 流程: inference.py(per-residue 表示) → inference_dms.py(log-ratio) →
    predict_mutation_effects.py。

设计决策(诚实, 2026-10-08 更新):
  - **零依赖 import**: torch/se3_transformer 延迟 import; 本机未部署抛 RuntimeError。
  - **真跑状态(已实测)**: 在腾讯云 GPU 实例上, 官方 `examples/1_dms_scanning.py`
    已真跑通(读预计算 `promep-fitness_prediction.h5` + `cas9.fasta` → 生成 25991 个
    饱和突变体并按 fitness 排序)。`score_variants` 现实现**全量推理**(官方三阶段:
    inference.py(drlnet 结构表示) → inference_dms.py(dmsnet log-ratio) → 逐突变打分),
    但权重 `Trained_model/{drlnet,dmsnet}.pth` 不在仓库、无 release/Zenodo、且
    SE(3)-Transformer 栈(python3.8+torch1.13+dgl0.9+e3nn0.3.3+apex)与 torch2.5 冲突,
    故需有完整环境的 GPU 实例 + 权重后方可真跑, 不假装已全量跑通。
  - 打分只是代理指标, 不预测 kcat/Km。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable

LOG = logging.getLogger("dockpipeline")

_PROMEP_HINT = (
    "ProMEP 全量推理需要 torch + se3-transformer + dgl + h5py + ProMEP 仓库 + "
    "Trained_model/{drlnet,dmsnet}.pth 权重(github.com/wenjiegroup/ProMEP)。权重不在"
    "仓库、无 release/Zenodo, 需从作者或 Code Ocean 获取。请在有完整环境的 GPU 实例上"
    "运行, 或改用 scoring.adapter=scores_file/esm1v/foldx/saprot/venusrem。"
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
    """对一批单点突变做 ProMEP **全量推理**打分, 返回 {variant_id: promep_score}。

    全量推理 = 官方三阶段: inference.py(结构感知 per-residue 表示, drlnet) →
    inference_dms.py(log-ratio 序列+结构条件概率, dmsnet) → 逐突变 log-ratio 打分。

    参数:
      wt_seq: WT 序列(1-letter)。
      variants: 可迭代 Variant / dict。
      repo_dir: ProMEP 仓库路径(必填)。
      drlnet_path: Trained_model/drlnet.pth(结构编码器权重, 必填)。
      dmsnet_path: Trained_model/dmsnet.pth(适应度预测器权重, 必填)。
      dataroot: TargetDataset 输入目录(含 WT PDB; 必填)。
      out_dir: 中间产物目录(缺省 repo_dir 下临时目录)。
    """
    try:
        import torch  # noqa: F401
        import se3_transformer  # noqa: F401
        import h5py  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_PROMEP_HINT) from e

    repo_dir = str(cfg.get("repo_dir", "") or "")
    drlnet = str(cfg.get("drlnet_path", "") or "")
    dmsnet = str(cfg.get("dmsnet_path", "") or "")
    dataroot = str(cfg.get("dataroot", "") or "")
    if not repo_dir or not (drlnet and dmsnet) or not dataroot:
        raise RuntimeError("libscore_promep 全量推理需要 repo_dir + drlnet_path + "
                           "dmsnet_path + dataroot(权重 Trained_model/drlnet.pth、"
                           "dmsnet.pth 不在仓库, 需从作者/Code Ocean 获取)")

    import subprocess
    import numpy as np
    out_dir = str(cfg.get("out_dir", "") or os.path.join(repo_dir, "_dockbuddy_promep"))
    os.makedirs(out_dir, exist_ok=True)
    emb_h5 = os.path.join(out_dir, "embeddings.h5")
    fit_h5 = os.path.join(out_dir, "fitness_prediction.h5")

    # 阶段 1: 结构感知 per-residue 表示(drlnet)
    subprocess.run(["python3", "inference.py", "-m", drlnet, "-d", dataroot,
                    "-f", emb_h5, "-t", "ec"], cwd=repo_dir, check=True,
                   capture_output=True, text=True, timeout=7200)
    # 阶段 2: log-ratio 序列+结构条件概率(dmsnet)
    subprocess.run(["python3", "inference_dms.py", "-m", dmsnet, "-d", dataroot,
                    "-f", fit_h5, "-t", "ec"], cwd=repo_dir, check=True,
                   capture_output=True, text=True, timeout=7200)

    return _promep_log_ratio(wt_seq, variants, fit_h5)


def _promep_log_ratio(wt_seq: str, variants: Iterable[Any],
                      fit_h5: str) -> Dict[str, float]:
    """从 fitness_prediction.h5 的 token 概率算逐突变 log-ratio 打分。"""
    import h5py
    import numpy as np
    import torch

    muts = [_fields(v) for v in variants]
    hf = h5py.File(fit_h5, "r")
    try:
        all_data = np.array(hf.get("embeddings"))
        token_probs = torch.log_softmax(torch.from_numpy(all_data[0]), dim=-1)
    finally:
        hf.close()

    alphabet = _default_alphabet()
    out: Dict[str, float] = {}
    for vid, wt, mt, site in muts:
        idx = site - 1
        wt_i, mt_i = alphabet.index(wt), alphabet.index(mt)
        score = (token_probs[idx, mt_i] - token_probs[idx, wt_i]).item()
        out[vid] = float(score)
    return out


def _default_alphabet():
    """ESM 标准 20 氨基酸字母表(与 myesm.Alphabet 对齐)。"""
    return list("LAGVSERTIDPKQNFYMHWC")


__all__ = ["score_variants"]
