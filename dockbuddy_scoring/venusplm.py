# -*- coding: utf-8 -*-
"""
venusplm.py —— VenusPLM 突变效应打分适配器 (2026-10-11)
===================================================================================
定位: 预筛管线「打分」步骤的序列 PLM 适配器 —— 用 VenusPLM(上海交大 Liang Hong 组,
AI4Protein, 2025)对单点突变做 **zero-shot 打分**, 输出 {variant_id: {venusplm_score}}。

接入方式(经联网核实官方 README + HF 仓库):
  - pip `vplm`(HuggingFace 风格接口)。
  - 权重 HF: `AI4Protein/VenusPLM-300M`(302M 参数, 24 层/1024 维/16 头/RoPE,
    36-token 词表, fp32 `model.safetensors` = 1,208,482,648 B ≈ 1.2GB)。
  - 依赖: torch>=2.5.0 + transformers; flash-attn 可选(仅 flash_attn 后端需要, 不装即
    用 `sdpa`/`naive` 后端)。
  - 许可: **CC-BY-NC-ND-4.0**(非商业), 学术研究可用。

打分公式(官方 README「Protein Design (Mutant Effect Prediction)」一节 + 官方
benchmark/proteingym_single.py, 已逐字核实一致):
  ```
  logits = outputs.logits.log_softmax(dim=-1).squeeze()[1:-1]   # [L, 36], 去掉 <cls>/<eos>
  score  = logits[idx, mt_token] - logits[idx, wt_token]        # idx = 0-based 位点
  ```
  即「突变残基的 log-softmax 概率 − 野生型残基的 log-softmax 概率」的对数几率。
  **方向: 分数通常 ≤ 0(给干净 WT 序列, WT 残基几乎处处是 argmax); 更接近 0 = 突变
  更可容忍/更似天然, 更负 = 更有害/更破坏序列上下文。与 ESM-1v 伪对数似然同向。**
  (官方对 ProteinGym 用 abs(spearman) 打分相关, 正因打分方向与实验体系取向有关。)
  ⚠️ 注意: 这是「序列可行性/容忍度」代理, 不是「活性/kcat 提升」预测, 也不预测
  kcat/Km。

设计决策(诚实, 2026-10-11):
  - **single-pass 扫描**: VenusPLM 一次前向即可得全部 L×20 的 logits, 因此对任意一批
    单点突变(乃至整蛋白全饱和 L×19)只做 **一次** forward, 成本 O(L) 而非 O(突变数)。
    这对「全饱和突变打分」极其友好: 252 aa 全饱和(4788 突变)= 一次前向, 秒~分钟级。
  - **零依赖 import**: torch/vplm 延迟 import; 本机未部署抛 RuntimeError(带部署指引),
    与 esm1v/saprot/venusrem 同口径。
  - **WT 身份硬校验**: 每个 variant 的 `from` 必须等于 seq[site-1], 不符跳过并告警
    (与 saprot/venusrem 同口径, 防止编号错位静默错突)。
  - 打分只是代理指标, 不预测 kcat/Km。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Tuple

LOG = logging.getLogger("dockpipeline")

_DEFAULT_MODEL = "AI4Protein/VenusPLM-300M"

_VENUSPLM_HINT = (
    "VenusPLM 打分需要 vplm(pip) + VenusPLM-300M 权重(HF: AI4Protein/VenusPLM-300M, "
    "1.2GB) + torch>=2.5。本机未部署。请在 GPU/CPU 实例(见 deploy/setup_v9.sh 或 "
    "deploy/镜像安装正向更新计划.md)或按 github.com/AI4Protein/VenusPLM 部署的环境运行, "
    "或改用 scoring.adapter=scores_file/esm1v/foldx。"
)

# 标准 20 种氨基酸单字母(用于从 vocab 里取 token id)。
_STD_AA = "ACDEFGHIKLMNPQRSTVWY"


def _fields(v) -> Tuple[str, str, str, int]:
    """把一个 Variant(dict 或对象)解析为 (variant_id, wt, mt, site)。"""
    if isinstance(v, dict):
        site = int(v["site"]); wt = str(v["from"]).upper(); mt = str(v["to"]).upper()
        vid = str(v.get("id") or v.get("variant_id") or f"{wt}{site}{mt}")
        return vid, wt, mt, site
    vid = str(getattr(v, "id", None) or getattr(v, "label", None) or "")
    return (vid, str(getattr(v, "from_aa")).upper(), str(getattr(v, "to_aa")).upper(),
            int(getattr(v, "site")))


def _extract_scores(logits_rows: List[List[float]], seq: str,
                    aa2token: Dict[str, int],
                    parsed: List[Tuple[str, str, str, int]]) -> Tuple[Dict[str, float], int]:
    """从 [L, V] 的 log-softmax 对数几率矩阵提取各突变 score(纯函数, 便于单测)。

    score = logits_rows[site-1][mt_token] - logits_rows[site-1][wt_token](≤0, 更接近 0
    = 更可容忍)。返回 ({variant_id: score}, n_skip)。WT 身份/越界/词表外硬校验, 不符跳过。
    """
    L = len(logits_rows)
    out: Dict[str, float] = {}
    n_skip = 0
    for vid, wt, mt, site in parsed:
        if site < 1 or site > L:
            LOG.warning("venusplm 跳过越界位点 %s%d%s(序列长 %d): %s", wt, site, mt, L, vid)
            n_skip += 1
            continue
        if seq[site - 1] != wt:
            LOG.warning("venusplm 跳过位点残基不符 %s%d%s(序列=%s): %s",
                        wt, site, mt, seq[site - 1], vid)
            n_skip += 1
            continue
        mt_token = aa2token.get(mt)
        wt_token = aa2token.get(wt)
        if mt_token is None or wt_token is None:
            LOG.warning("venusplm 跳过不在词表的残基 %s/%s: %s", wt, mt, vid)
            n_skip += 1
            continue
        out[vid] = float(logits_rows[site - 1][mt_token]
                         - logits_rows[site - 1][wt_token])
    return out, n_skip


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    """对一批单点突变做 VenusPLM 打分, 返回 {variant_id: venusplm_score}。

    参数:
      wt_seq: WT 序列(1-letter)。
      variants: 可迭代 Variant / dict。
      model_name / weights_dir: HF 仓库 id(缺省 AI4Protein/VenusPLM-300M)或本地快照目录。
      attn_impl: 'sdpa'(缺省) / 'naive' / 'flash_attn'(需装 flash-attn 且 bf16)。
      device: 'cpu' / 'cuda' / 'auto'(缺省自动探测)。
      hf_endpoint: 可选, 设 HF_ENDPOINT 镜像(如 https://hf-mirror.com)。
      cache_dir: 可选, HF 缓存目录。
      local_files_only: 是否只从本地缓存/目录加载(缺省 False)。

    说明: **单次前向** 得到全位点×20aa 的 logits 后逐突变提取, 因此全饱和突变
    (L×19 个)也只需一次 forward, 时间与突变数几乎无关。
    """
    try:
        import torch  # noqa: F401
        from vplm import TransformerConfig, TransformerForMaskedLM, VPLMTokenizer
    except ImportError as e:
        raise RuntimeError(_VENUSPLM_HINT) from e

    # HF 镜像(国内直连不畅时), 仅当未显式设置时才写入环境
    hf_ep = str(cfg.get("hf_endpoint", "") or "")
    if hf_ep:
        os.environ.setdefault("HF_ENDPOINT", hf_ep)

    model_name = str(cfg.get("model_name") or cfg.get("weights_dir")
                     or cfg.get("model_dir") or _DEFAULT_MODEL)
    attn_impl = str(cfg.get("attn_impl", "sdpa") or "sdpa")

    seq = "".join(str(wt_seq).split()).upper()
    parsed: List[Tuple[str, str, str, int]] = [_fields(v) for v in variants]

    # 1) 加载模型 + 分词器(官方用法: TransformerConfig + from_pretrained)
    LOG.info("VenusPLM 加载模型 %s(attn_impl=%s)...", model_name, attn_impl)
    cache_dir = str(cfg.get("cache_dir", "") or "") or None
    local_only = bool(cfg.get("local_files_only", False))
    kw = {}
    if cache_dir:
        kw["cache_dir"] = cache_dir
    if local_only:
        kw["local_files_only"] = True
    config = TransformerConfig.from_pretrained(model_name, attn_impl=attn_impl, **kw)
    model = TransformerForMaskedLM.from_pretrained(model_name, config=config, **kw)
    tokenizer = VPLMTokenizer.from_pretrained(model_name, **kw)

    from dockbuddy_common import resolve_device
    device = resolve_device(str(cfg.get("device", "auto") or "auto"))
    model.to(device)
    model.eval()
    LOG.info("VenusPLM 模型就绪(device=%s), WT 序列 %d aa", device, len(seq))

    # 2) 单次前向 → 全位点 log-softmax 对数几率矩阵 [L, 36]
    enc = tokenizer(seq, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=False,
        )
    logits = outputs.logits.log_softmax(dim=-1).squeeze()[1:-1]  # 去掉 <cls>/<eos>
    L = int(logits.shape[0])
    if L != len(seq):
        LOG.warning("VenusPLM logits 位点数 %d 与 WT 序列长度 %d 不一致, 以 logits 为准", L, len(seq))

    # 3) 转纯 Python 列表后逐突变提取(纯函数, 便于单测)
    logits_rows = logits.detach().cpu().tolist()
    vocab = tokenizer.get_vocab()
    aa2token = {aa: vocab[aa] for aa in _STD_AA if aa in vocab}
    out, n_skip = _extract_scores(logits_rows, seq, aa2token, parsed)
    LOG.info("VenusPLM 打分完成: %d/%d 突变(跳过 %d)", len(out), len(parsed), n_skip)
    return out


__all__ = ["score_variants"]
