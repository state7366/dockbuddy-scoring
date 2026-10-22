# -*- coding: utf-8 -*-
"""
scoring/muformer.py —— µProtein(µFormer+µSearch) 接入框架(2026-10-12 新增)
===================================================================================
定位: 接入 Sun et al. 2025 (Nat Mach Intell 10.1038/s42256-025-01103-w, 微软) 的
µProtein —— µFormer(siamese 突变效应预测) + µSearch(RL 导航适应度地形), 用于「单点
→ 多点组合突变设计」。

**重要口径(与零样本打分器本质不同)**: µFormer 是**监督 siamese 模型**(输入 WT 与
突变序列对, 输出突变效应), 必须先用「单点突变湿实验数据」训练, 才能对多点突变打分。
故本适配器是「训练 + 打分」框架, 不是即插即用零样本打分器。

前置(诚实声明, 缺一不可):
  1. PBATase 单点突变湿实验活性数据(训练 µFormer 的标签; 用户尚未产出)。
  2. GPU + 独立 conda py3.8 + torch 1.12 环境(µFormer 训练; mu-former README 要求
     torch 1.12+cu113; 本地 CPU 不可行)。
  3. figshare 权重: µFormer 预训练 encoder `uformer_l_encoder.pt`
     (10.6084/m9.figshare.26892355); µSearch 用 BLAT µFormer ckpt
     (figshare 30227557)。

接入形态: 训练(µFormer 微调) 与搜索(µSearch) 走远程 GPU 实例(镜像固化 conda py3.8
环境 + figshare 权重进 COS), 参考 tranception 的独立环境先例。本地仅保留薄封装 + 前置校验。

状态(2026-10-12): 代码+权重(650M PMLM, 2.6GB)已本地就绪; 训练/打分待单点数据 + GPU。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

LOG = logging.getLogger("dockpipeline")

# µFormer 仓库本地落位(相对 dockBuddy 根)
_MU_PROTEIN_REPO = "deploy/models/mu_protein"


def _require_data_and_env() -> None:
    """前置校验: 明确报错, 而非静默失败。"""
    raise RuntimeError(
        "µFormer 是监督 siamese 模型, 需「单点突变湿实验数据」训练 + GPU(conda py3.8 "
        "+ torch 1.12)才能打分; 当前两者均未就绪。零样本突变打分请用 esm1v/esm2/saprot/"
        "profam/lfb 等。接入步骤见 docs/papers/模型接入计划.md §3.2 µProtein。"
    )


def score_variants(wt_seq: str, variants: Iterable[Any], *,
                   checkpoint: str = "", device: str = "cpu",
                   **cfg) -> Dict[str, float]:
    """µFormer 打分入口(需已训练 checkpoint; 未就绪时抛前置错误)。

    参数:
      checkpoint: 已微调的 µFormer checkpoint(训练产出)。
      variants: 变体列表(单点/多点)。
    """
    _require_data_and_env()


def train(wt_seq: str, train_tsv: str, valid_tsv: str, test_tsv: str, *,
          fasta: str = "", pretrained_model: str = "", output_dir: str = "",
          device: str = "cuda") -> Dict[str, Any]:
    """µFormer 微调入口(包装 mu-former/main.py; 需单点数据 + GPU)。

    参数同 mu-former/main.py: --decoder-name siamese --pretrained-model <encoder>
    --fasta <fasta> --train/valid/test <tsv> --output-dir <dir>。
    """
    _require_data_and_env()


__all__ = ["score_variants", "train"]
