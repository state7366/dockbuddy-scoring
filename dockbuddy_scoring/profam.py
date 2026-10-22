# -*- coding: utf-8 -*-
"""
scoring/profam.py —— ProFam-1 蛋白家族语言模型打分适配器(2026-10-12 新增)
===================================================================================
定位: 接入 Wells et al. 2025 (bioRxiv 10.1101/2025.12.19.695431) 的 ProFam-1 ——
一个 **251M 参数自回归蛋白家族语言模型(pfLM)**, 在「同源家族序列集合」上做
next-token 预测, 显式建模家族内进化约束。打分口径是「家族条件对数似然」:
把候选序列放进家族上下文(同源序列作为 prompt), 计算其平均对数似然; 突变效应 =
log-lik[突变序列] − log-lik[WT序列]。**正值 = 突变在家族上下文中更可能(更可接受)**,
方向与 esm1v/esm2 的「正=偏好」一致。

与 LFB 的关系(口径区分, 避免混用):
  - LFB = **后处理校正层**, 在现有 esm2 单序列打分上做「家族平均去漂变」, 不引入新权重。
  - ProFam-1 = **端到端家族预训练模型**, 训练时就以「家族集合」为输入单元, 学到家族级
    保守性与共进化模式; 是独立打分器, 填补打分矩阵「家族级进化约束」这一空白。

参考指标(论文): ProteinGym 零样本 Spearman ≈ 0.47(替换) / 0.53(indel)。

设计决策(记录备查):
  - 零依赖 import: profam 及其重依赖(hydra/lightning/sklearn/numba)延迟到函数内 import,
    主 pipeline 其它模型加载不受影响。
  - 单例加载: 251M checkpoint(~1GB)只加载一次, 存模块级缓存, 重复打分复用。
  - dtype 缺省 float32(Windows CPU 无 bfloat16 加速; 有 CUDA 时可由 cfg.dtype 覆盖)。
  - 打分只对单点变体显式构造序列; 多点/双点由调用方显式传入复合序列(见 score_sequences)。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOG = logging.getLogger("dockpipeline")

# 权重落位约定(与 deploy/models/profam1/assets 一致; 可用 cfg.checkpoint 覆盖)
_DEFAULT_CHECKPOINT = "deploy/models/profam1/assets/checkpoints/last.ckpt"

# 模块级单例缓存: (checkpoint, dtype) -> ProFam 实例
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _variant_fields(v) -> Tuple[int, str, str, str]:
    """从 Variant 或 dict 提取 (site 1-based, from_aa, to_aa, variant_id)。"""
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


def _mutate_seq(wt_seq: str, site: int, mt: str) -> str:
    """把 wt_seq 在 site(1-based) 位置替换为 mt。"""
    seq = "".join(str(wt_seq).split())
    i = site - 1
    if i < 0 or i >= len(seq):
        raise ValueError(f"site {site} 越界(序列长 {len(seq)})")
    return seq[:i] + mt.upper() + seq[i + 1:]


def _read_family_file(family_file: str) -> List[str]:
    """读 family MSA 文件(fasta/a2m/a3m) → 对齐序列列表(去小写插入, 保留大写+gap)。

    profam 的 prompt 须为 list[str](对齐同源序列, 含 gap); 此前直接传路径字符串, 导致
    model.score 把字符串逐字符当序列 → hamming 相似度 assert 失败。
    """
    seqs: List[str] = []
    cur: List[str] = []
    for line in open(family_file, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        else:
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    out = []
    for s in seqs:
        s2 = "".join(c for c in s if c.isupper() or c in "-.")
        if s2:
            out.append(s2)
    return out


def _fix_tokenizer(model) -> None:
    """修复 profam 0.1.8 的 tokenizer pickle bug(workaround)。

    现象: LlamaLitModule.load_from_checkpoint 恢复的 tokenizer 经 pickle 序列化,
    `__init__` 未重跑, 导致 `_sep_token`/`_bos_token` 等实例属性缺失, 调用
    `model.tokenizer.sep_token` 抛 `AttributeError`。此处用 profam 包自带
    tokenizer 文件 + 官方 config 的特殊 token 重新实例化并替换。
    """
    tok = getattr(model, "tokenizer", None)
    if tok is not None and hasattr(tok, "_sep_token") and tok._sep_token is not None:
        return
    import os
    import profam.data as pdata
    from profam.data.tokenizers import ProFamTokenizer
    tok_file = os.path.join(os.path.dirname(pdata.__file__), "profam_tokenizer.json")
    model.tokenizer = ProFamTokenizer(
        tokenizer_file=tok_file,
        unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[start-of-document]", sep_token="[SEP]", mask_token="?",
        seq_struct_sep_token="|", add_bos_token=True, add_document_token=True)


def _load_model(checkpoint: str, device: str, dtype: str):
    """加载 ProFam-1 模型(单例缓存)。"""
    key = (checkpoint, dtype)
    if key not in _MODEL_CACHE:
        from profam import ProFam
        LOG.info("加载 ProFam-1 checkpoint: %s (dtype=%s)", checkpoint, dtype)
        model = ProFam(checkpoint=checkpoint or None, device=device,
                       dtype=dtype, auto_download=False)
        _fix_tokenizer(model._model)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def score_sequences(sequences: List[str], *, family_file: str = "",
                    family_seqs: Optional[Sequence[str]] = None,
                    checkpoint: str = "", device: str = "cpu",
                    dtype: str = "float32", ensemble_size: int = 3,
                    use_diversity_weights: bool = True,
                    seed: int = 42) -> List[float]:
    """对一批序列做家族条件打分, 返回每序列的 mean log-likelihood 列表。

    参数:
      sequences: 候选序列列表(与 WT 等长)。
      family_file: 家族 MSA 文件(FASTA / a2m / a3m), 作为家族上下文 prompt。
      family_seqs: 内联家族序列列表(与 family_file 二选一)。
      checkpoint: 本地 last.ckpt 路径(缺省 deploy/models/profam1/assets/checkpoints/last.ckpt)。
      device: 'cpu' / 'cuda'。
      dtype: 'float32' / 'bfloat16'。
      ensemble_size: 采样集成次数(缺省 3)。
    """
    from profam import ProFam  # noqa: F401  (延迟 import 纪律)
    model = _load_model(checkpoint or _DEFAULT_CHECKPOINT, device, dtype)

    prompt: Optional[Any] = None
    if family_file:
        prompt = _read_family_file(family_file)
    elif family_seqs:
        prompt = list(family_seqs)

    result = model.score(
        sequences=list(sequences),
        prompt=prompt,
        ensemble_size=int(ensemble_size),
        use_diversity_weights=bool(use_diversity_weights),
        per_residue=False,
        seed=int(seed),
    )
    return [float(x) for x in result.scores]


def score_variants(wt_seq: str, variants: Iterable[Any], *,
                   family_file: str = "", family_seqs: Optional[Sequence[str]] = None,
                   checkpoint: str = "", device: str = "cpu",
                   dtype: str = "float32", ensemble_size: int = 3,
                   use_diversity_weights: bool = True, seed: int = 42,
                   **cfg) -> Dict[str, float]:
    """ProFam-1 打分入口(接入 registry 的 score 型 adapter `profam`)。

    对每个单点变体显式构造突变序列, 与 WT 一起批量家族条件打分; 返回
    {variant_id: Δlog-lik = score[mut] − score[wt]}(正=突变更可能/更可接受)。
    """
    seq = "".join(str(wt_seq).split())
    fields = [_variant_fields(v) for v in variants]
    muts = [_mutate_seq(seq, site, mt) for site, _wt, mt, _vid in fields]

    seqs = [seq] + muts
    scores = score_sequences(
        seqs, family_file=family_file, family_seqs=family_seqs,
        checkpoint=checkpoint, device=device, dtype=dtype,
        ensemble_size=ensemble_size, use_diversity_weights=use_diversity_weights,
        seed=seed)
    wt_score = scores[0]
    out: Dict[str, float] = {}
    for (site, _w, _m, vid), s in zip(fields, scores[1:]):
        out[vid] = round(s - wt_score, 6)
    return out


__all__ = ["score_variants", "score_sequences", "_mutate_seq", "score_variants"]
