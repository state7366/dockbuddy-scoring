# -*- coding: utf-8 -*-
"""
libscore_esm.py —— ESM-1v zero-shot 变异打分适配器(fair-esm, 2026-10-08 新增)
===================================================================================
定位: 预筛管线「打分」步骤的第一个真实模型适配器 —— 用 ESM-1v(Meier 2021, MIT)对
单点突变做 zero-shot 伪对数似然打分, 输出 {variant_id: {esm1v_score}}。挂在
orchestrator 的 score 步骤(scores_file 之外的「适配器计算」路径)。

打分口径(Meier et al. 2021, PNAS 118:e2016239118):
  对每个突变 (site, wt→mut), mask 该位点 → 取该位点 logits → log-softmax →
  score = log P(mut) − log P(wt)。5 个种子模型(esm1v_t33_650M_UR90S_1..5)取平均。
  正值 = 突变比 WT 更"被模型接受"(倾向保留功能), 负值 = 更不被接受。

设计决策(记录备查):
  - **零依赖 import**: torch / fair-esm 全部延迟到函数内 import, 模块级只依赖
    标准库 → mock 单测(不加载真模型)在 CI 无 GPU 可绿; 真跑需 torch+fair-esm
    + esm1v 权重(见 deploy/ 腾讯云 GPU 镜像方案)。
  - **打分只是代理指标**: 输出喂布尔闸用, 不预测 kcat/Km(与管线统一口径)。
  - 设备: 缺省 cpu(本机无 GPU 也能慢跑); 生产用 cuda(腾讯云 GPU 实例)。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, Sequence, Tuple

LOG = logging.getLogger("dockpipeline")

ESM1V_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)


def _default_models_dir(subdir: str) -> str:
    """探测标准权重目录(DOCK_MODELS_DIR 环境变量 / 新旧布局), 不存在返回空。

    空 = 回退 fair-esm torch.hub 自动下载(外网, 仅本机/无 COS 时)。
    2026-10-14: 部署已迁到 /home/ubuntu/dockbuddy(不用 root/不进系统目录),
    旧硬编码 /opt/dockbuddy 已废弃; 优先 DOCK_MODELS_DIR, 回退新旧布局探测。
    """
    roots = [os.environ.get("DOCK_MODELS_DIR", ""),
             "/home/ubuntu/dockbuddy/models",
             "/opt/dockbuddy/models"]
    for root in roots:
        if not root:
            continue
        d = os.path.join(root, subdir)
        if os.path.isdir(d):
            return d
    for root in roots:
        if not root:
            continue
        d = os.path.join(root, "_common", "assets")
        if os.path.isdir(d):
            return d
    return ""


def _variant_fields(v) -> Tuple[int, str, str, str]:
    """从 Variant(有 .site/.from_aa/.to_aa) 或 dict{'site','from','to'} 提取字段。

    返回 (site 1-based, from_aa, to_aa, variant_id)。variant_id 缺省按 F205G 拼。
    """
    if isinstance(v, dict):
        site = int(v["site"])
        wt, mt = str(v["from"]).upper(), str(v["to"]).upper()
        vid = str(v.get("id") or v.get("variant_id") or f"{wt}{site}{mt}")
    else:  # libmutlib.Variant 或同类 dataclass
        site = int(getattr(v, "site"))
        wt = str(getattr(v, "from_aa")).upper()
        mt = str(getattr(v, "to_aa")).upper()
        vid = str(getattr(v, "id", None) or getattr(v, "label", None) or f"{wt}{site}{mt}")
    return site, wt, mt, vid


def logit_diff(logits: Sequence[float], wt_idx: int, mt_idx: int) -> float:
    """纯 Python: log P(mut) − log P(wt) = logits[mt] − logits[wt]。

    说明: log_softmax 的 log-sum-exp 归一化项在两处相减时**数学恒等抵消**, 故
    zero-shot 伪对数似然比就是目标位点 logits 的差值。抽成纯函数便于无 torch 单测。
    """
    return float(logits[mt_idx]) - float(logits[wt_idx])


def esm1v_single_score(model, alphabet, seq: str, site: int,
                       wt_aa: str, mut_aa: str) -> float:
    """给定已加载的 esm1v 模型, 算单点突变 (site, wt→mut) 的 zero-shot 分。

    纯函数(除 torch 取 logits 外), 便于用假模型做单测。site 为 1-based 残基号。
    """
    tokens = alphabet.encode(seq)
    if site < 1 or site >= len(tokens) - 1:      # 越界(BOS/EOS 之外)
        raise ValueError(f"位点 {site} 超出序列范围 (长 {len(seq)})")
    masked = tokens.copy()
    masked[site] = alphabet.mask_idx
    import torch
    device = next(model.parameters()).device      # 输入须与模型同设备(GPU 真跑才暴露)
    with torch.no_grad():
        logits = model(torch.tensor([masked], device=device))["logits"]   # [1, L, vocab]
    row = [float(x) for x in logits[0, site]]
    return logit_diff(row, alphabet.get_idx(wt_aa), alphabet.get_idx(mut_aa))


def load_esm1v_seed(models_dir: str, device: str, seed_index: int):
    """加载 esm1v 第 seed_index 个种子模型并置 eval+device, 返回 (model, alphabet)。"""
    import esm

    if models_dir:
        # fair-esm 的 load_model_and_alphabet_local 接收**单个 .pt 文件路径**(非目录),
        # 文件名 stem 决定模型名; esm1v 每个种子一个文件。
        path = os.path.join(models_dir, f"esm1v_t33_650M_UR90S_{seed_index + 1}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"esm1v 权重缺失: {path}")
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(path)
    else:
        name = f"esm1v_t33_650M_UR90S_{seed_index + 1}"
        model, alphabet = esm.pretrained.load_model_and_alphabet(name)
    return model.eval().to(device), alphabet


def esm1v_scores(wt_seq: str, variants: Iterable[Any], *,
                 models_dir: str = "", device: str = "auto",
                 seeds: Sequence[int] = ESM1V_SEEDS) -> Dict[str, float]:
    """对一批单点突变做 ESM-1v 打分, 返回 {variant_id: esm1v_score}。

    参数:
      wt_seq: WT 序列(1-letter)。
      variants: 可迭代, 每项为 libmutlib.Variant 或 dict{'site','from','to'}。
      models_dir: esm1v 权重本地目录(缺省走 fair-esm 自动下载)。
      device: 'cuda' / 'cpu' / 'auto'(缺省, 有 GPU 自动用 cuda)。
      seeds: 参与的种子索引(缺省 5 个全用)。
    """
    from dockbuddy_common import resolve_device
    device = resolve_device(device)
    seq = "".join(str(wt_seq).split())
    fields = [_variant_fields(v) for v in variants]
    models_dir = models_dir or _default_models_dir("esm1v")
    scores: Dict[str, float] = {}
    for seed_index in seeds:
        model, alphabet = load_esm1v_seed(models_dir, device, seed_index)
        for site, wt, mt, vid in fields:
            s = esm1v_single_score(model, alphabet, seq, site, wt, mt)
            scores[vid] = scores.get(vid, 0.0) + s
    n = max(1, len(seeds))
    return {vid: round(v / n, 6) for vid, v in scores.items()}


__all__ = ["logit_diff", "esm1v_single_score", "esm1v_scores",
           "load_esm1v_seed", "ESM1V_SEEDS"]
