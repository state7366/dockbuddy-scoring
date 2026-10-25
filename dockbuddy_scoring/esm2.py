# -*- coding: utf-8 -*-
"""
libscore_esm2.py —— ESM-2 zero-shot 变异打分适配器(2026-10-11 新增)
===================================================================================
定位: 打分模型矩阵的补充 —— 用 ESM-2(Lin et al. 2023, Science 379:1123, MIT)
对单点突变做 zero-shot 伪对数似然打分, 输出 {variant_id: esm2_score}。与 esm1v
并列(esm1v 是 5 种子集成, esm2 是单模型), 供多模型交叉比较。

打分口径(与 esm1v 完全同源, Meier et al. 2021):
  mask 目标位点 → 取该位点 logits → log-softmax 归一化项在相减时数学恒等抵消 →
  score = logits[mut] − logits[wt]。正值 = 突变比 WT 更"被模型接受"。

与 esm1v 的差异(诚实标注):
  - esm1v: 5 个独立种子模型(esm1v_t33_650M_UR90S_1..5)平均, 抗种子方差。
  - esm2: 单模型(esm2_t33_650M_UR50D 等 8M~15B 多档), 无种子平均; 规模更大时
    结构信息内隐更强, 但零样本突变效应排序仍为代理指标, 不预测 kcat/Km。

设计决策(记录备查):
  - **复用 esm.py 的纯函数**: esm1v_single_score / logit_diff 对任何 fair-esm
    的 mask 语言模型通用(只依赖 model/alphabet 的 mask_idx 与 logits), 故此处
    不复制打分核心, 只做「单模型加载 + 循环」的薄封装。变体字段解析局部复制
    esm.py._variant_fields(私有, 不跨模块依赖)。
  - **零依赖 import**: torch / fair-esm 延迟到函数内 import(同 esm.py 纪律),
    mock 单测无 GPU 可绿。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, Tuple

LOG = logging.getLogger("dockpipeline")

# 缺省模型: 与 esm1v 同规模(650M)便于横向对比; 用户可 --model 换更小/更大档。
ESM2_DEFAULT = "esm2_t33_650M_UR50D"


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
    else:
        site = int(getattr(v, "site"))
        wt = str(getattr(v, "from_aa")).upper()
        mt = str(getattr(v, "to_aa")).upper()
        vid = str(getattr(v, "id", None) or getattr(v, "label", None) or f"{wt}{site}{mt}")
    return site, wt, mt, vid


def load_esm2(model_name: str, device: str, models_dir: str = ""):
    """加载单个 ESM-2 模型并置 eval+device, 返回 (model, alphabet)。

    models_dir: 本地权重目录(含 <model_name>.pt); 空则走 fair-esm torch.hub 自动下载
      (从 dl.fbaipublicfiles.com 外网拉, 违反内网铁律, 仅本机/无 COS 时回退)。
    """
    import esm

    if models_dir:
        path = os.path.join(models_dir, f"{model_name}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"esm2 权重缺失: {path}")
        # 打分只需 masked LM logits, 不需要 contact regression 头; 用 core 加载跳过
        # -contact-regression.pt(该文件不入 COS, 避免 260MB 冗余外置)。
        import torch
        # torch>=2.6 默认 weights_only=True 会拦截 fair-esm checkpoint(含
        # argparse.Namespace); 权重来自本地 COS/官方发布, 属可信来源。
        try:
            model_data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # torch<1.13 无此参数
            model_data = torch.load(path, map_location="cpu")
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(model_name, model_data, None)
    else:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    return model.eval().to(device), alphabet


def esm2_scores(wt_seq: str, variants: Iterable[Any], *,
                model_name: str = ESM2_DEFAULT, device: str = "auto",
                models_dir: str = "") -> Dict[str, float]:
    """对一批单点突变做 ESM-2 zero-shot 打分, 返回 {variant_id: esm2_score}。

    参数:
      wt_seq: WT 序列(1-letter)。
      variants: 可迭代, 每项为 Variant 或 dict{'site','from','to'}。
      model_name: ESM-2 模型名(缺省 esm2_t33_650M_UR50D; 也支持
        esm2_t6_8M_UR50D / esm2_t30_150M_UR50D / esm2_t36_3B_UR50D / ...)。
      device: 'cuda' / 'cpu' / 'auto'(缺省, 有 GPU 自动用 cuda)。
      models_dir: 本地权重目录(空则走 fair-esm 自动下载)。
    """
    from dockbuddy_common import resolve_device
    from dockbuddy_scoring.esm import esm1v_single_score  # 打分核心(对任何 fair-esm mask 模型通用)

    device = resolve_device(device)
    seq = "".join(str(wt_seq).split())
    fields = [_variant_fields(v) for v in variants]
    models_dir = models_dir or _default_models_dir("esm2")
    model, alphabet = load_esm2(model_name, device, models_dir)
    scores: Dict[str, float] = {}
    for site, wt, mt, vid in fields:
        scores[vid] = esm1v_single_score(model, alphabet, seq, site, wt, mt)
    return {vid: round(v, 6) for vid, v in scores.items()}


__all__ = ["esm2_scores", "load_esm2", "ESM2_DEFAULT"]
