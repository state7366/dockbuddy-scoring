# -*- coding: utf-8 -*-
"""
libscore_tranception.py —— Tranception zero-shot 突变打分适配器(2026-10-11 新增)
===================================================================================
定位: 打分模型矩阵的补充 —— 用 Tranception(Notin et al. 2022, ICML; autoregressive
Transformer + 推理时检索)对单点突变做 zero-shot 打分。ProteinGym 取代榜长期居首
(含 indel), 是「ESM 掩码语言模型」之外的另一类零样本打分范式。

打分口径(与 ESM 掩码打分不同):
  Tranception 是**自回归**模型, 沿序列方向逐位预测, score = log P(mut_seq) −
  log P(wt_seq)(全长似然比, 非单点 logits 差)。可选 retrieval: 用同源序列
  (MSA)在推理时增强上下文, 进一步提升相关度。

接入方式(与 geostab/evmutation 同款「独立 repo 薄适配器」):
  - 仓库 github.com/OATML-Markslab/Tranception; 权重 HuggingFace
    (OATML-Markslab/Tranception_{Small,Medium,Large}) 约 0.5~2.5GB。
  - 依赖: torch + transformers + tokenizers。真跑需 GPU 实例 + repo + 权重;
    本机未部署。

诚实边界:
  - 本适配器是**官方 repo 的调用包装**, 不重实现模型; 缺 repo/权重时抛 RuntimeError
    (带部署指引), 与 evmutation/geostab 同口径。
  - 打分只是代理指标, 不预测 kcat/Km; 调用约定按官方 README 校准, **部署后需实测
    对齐**(本机无 repo+权重, 未真跑验证)。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

LOG = logging.getLogger("dockpipeline")

_TRANCEPTION_HINT = (
    "Tranception 打分需要其官方仓库(github.com/OATML-Markslab/Tranception, 克隆后"
    " --repo-dir 指向它) + checkpoint(Tranception_{Small,Medium,Large}_checkpoint.zip"
    " 解压到 repo_dir/, 下载 https://marks.hms.harvard.edu/tranception/) + "
    "transformers 4.17 + tokenizers 0.11.6 + datasets + biopython 1.78。"
    "**版本硬约束**: Tranception(2022)依赖 transformers 4.17(transformers.file_utils/"
    "modeling_utils.Conv1D 等旧 API), 不兼容 transformers 5.x; 且老 tokenizers 0.11.6 "
    "无 Python 3.13 wheel。本机(Python 3.13 + transformers 5.19)无法运行, 请在 "
    "Python 3.8 的 conda 环境(官方 tranception_env.yml)或 GPU 实例部署后运行。"
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
    """构造单点突变后的完整序列(1-based site; 越界/身份不符抛 ValueError)。"""
    i = site - 1
    if not (0 <= i < len(wt_seq)):
        raise ValueError(f"位点 {site} 越界(WT 序列长度 {len(wt_seq)})")
    if wt_seq[i].upper() != wt.upper():
        raise ValueError(f"位点 {site} 与 WT 不符: 序列[{i}]={wt_seq[i]} != {wt}")
    return wt_seq[:i] + mt + wt_seq[i + 1:]


def score_variants(wt_seq: str, variants: Iterable[Any], **cfg) -> Dict[str, float]:
    """对一批单点突变做 Tranception zero-shot 打分, 返回 {variant_id: tranception_score}。

    参数:
      wt_seq: WT 序列(1-letter)。
      variants: 可迭代 Variant / dict。
      repo_dir: Tranception 仓库路径(必填)。
      model_name: Tranception_{Small,Medium,Large}(缺省 Small; checkpoint 须在
        repo_dir/<model_name>/ 下)。
      scoring_mirror: 是否双向打分(L2R + R2L, 缺省 False 只 L2R 提速)。
      batch_size_inference: 推理 batch(缺省 1)。
    """
    import json
    import os
    import sys

    # Biopython ≥1.80 移除了 Bio.Align.Applications(ClustalOmegaCommandline), 而
    # Tranception(2022, 需 Biopython 1.78)在 msa_utils 模块级 import 它。无检索打分
    # 不涉 ClustalOmega, 此处注入占位模块绕过版本冲突(不降级 Biopython)。
    import types as _types
    if "Bio.Align.Applications" not in sys.modules:
        _fake = _types.ModuleType("Bio.Align.Applications")
        _fake.ClustalOmegaCommandline = type("ClustalOmegaCommandline", (), {})
        sys.modules["Bio.Align.Applications"] = _fake

    repo_dir = str(cfg.get("repo_dir", "") or "")
    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        import tranception  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_TRANCEPTION_HINT) from e

    from transformers import PreTrainedTokenizerFast
    import pandas as pd
    from tranception import config as tranception_config
    from tranception import model_pytorch as tranception_model
    from tranception.utils import scoring_utils

    model_name = str(cfg.get("model_name", "Tranception_Small") or "Tranception_Small")
    # checkpoint_dir: 可选, 指向 checkpoint 目录(COS 下载到 models/tranception/Tranception_Small/);
    # 缺省用 repo_dir/<model_name>(官方 zip 解压到仓库根)。
    checkpoint = str(cfg.get("checkpoint_dir", "") or "") or os.path.join(repo_dir, model_name)
    if not os.path.isdir(checkpoint):
        raise RuntimeError(f"Tranception checkpoint 缺失: {checkpoint}(需下载 "
                           f"{model_name}_checkpoint.zip 解压到 repo_dir/ 或经 checkpoint_dir 指向)")

    tokenizer_path = os.path.join(repo_dir, "tranception", "utils", "tokenizers",
                                  "Basic_tokenizer")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    # Tranception Basic_tokenizer 的 special token 未登记 pad/eos/bos/unk, 但 vocab 里
    # 有 [PAD]/[SEP]/[CLS]/[UNK](id 3/4/5/6 附近); encode_batch 里 padding=True 需要 pad。
    # 显式登记, 与 vocab 已有 id 对齐(实测 pad_token 不设会 ValueError)。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "[SEP]"
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "[CLS]"
    if tokenizer.unk_token is None:
        tokenizer.unk_token = "[UNK]"

    with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["retrieval_aggregation_mode"] = None  # 无检索(检索需 MSA)
    config = tranception_config.TranceptionConfig(**config_dict)
    config.tokenizer = tokenizer

    model = tranception_model.TranceptionLMHeadModel.from_pretrained(
        pretrained_model_name_or_path=checkpoint, config=config)
    model.eval()

    seq = "".join(str(wt_seq).split())
    muts, vids = [], []
    for v in variants:
        vid, wt, mt, site = _fields(v)
        muts.append(f"{wt}{site}{mt}")
        vids.append(vid)

    # score_mutants 里 `target_seq in DMS_data.mutated_sequence.values` 直接读原始
    # DMS_data 的 mutated_sequence 列(不读内部 df 副本), 故必须预先塞入完整突变序列列。
    mutated = [scoring_utils.get_mutated_sequence(seq, m) for m in muts]
    DMS_data = pd.DataFrame({"mutant": muts, "mutated_sequence": mutated})
    all_scores = model.score_mutants(
        DMS_data, target_seq=seq,
        scoring_mirror=bool(cfg.get("scoring_mirror", False)),
        batch_size_inference=int(cfg.get("batch_size_inference", 1) or 1),
        num_workers=0)

    out: Dict[str, float] = {}
    for vid, ms in zip(vids, mutated):
        row = all_scores[all_scores["mutated_sequence"] == ms]
        if not row.empty:
            out[vid] = float(row["avg_score"].iloc[0])
    return out


__all__ = ["score_variants", "_mutated_sequence"]
