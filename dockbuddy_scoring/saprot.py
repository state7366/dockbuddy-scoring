# -*- coding: utf-8 -*-
"""
libscore_saprot.py —— SaProt(结构感知 PLM) 打分适配器(2026-10-08; 2026-10-09 修正)
===================================================================================
定位: 预筛管线「打分」步骤的结构感知适配器 —— 用 SaProt(ICLR'24 spotlight)对单点
突变做 zero-shot 打分, 输出 {variant_id: {saprot_score: float}}。相比 ESM-1v 的纯
序列打分, SaProt 把 Foldseek 编码的 3Di 结构令牌拼进序列, 对活性口袋等结构敏感位点
更准(ProteinGym 榜首级)。

接入方式(2026-10-09 在腾讯云 GPU 实例实测核实, **无需 conda**):
  - 仓库 github.com/YaoYinYing/SaProt(系统 Python 直接可用)。
  - 权重 HuggingFace: SaProt_35M_AF2 / SaProt_650M_AF2 / SaProt_650M_PDB。
  - 关键依赖: Foldseek 二进制(PDB→3Di 结构令牌) + transformers 4.28.0 + torch。
  - 官方打分: `SaprotFoldseekMutationModel.predict_mut(combined_seq, "V3A")`。

设计决策(诚实, 2026-10-09 实测修正):
  - **必须喂「结构感知序列」而非纯序列**: SaProt 的 predict_mut 期望 `combined_seq`
    (残基+3Di 令牌对, 如 "MdEvVp..."), 由 `get_struc_seq(foldseek, pdb, [chain])`
    生成。纯氨基酸序列(单字符)不在 446 词表中 → 全映射 UNK → 打分无意义。这是
    之前版本的科学正确性 bug, 现已修正: `score_variants` 内部先跑 Foldseek 把 WT
    结构编码成 combined_seq 再打分。
  - **零依赖 import**: torch/saprot 延迟 import; 本机未部署抛 RuntimeError(带部署
    指引), 与 esm1v/evmutation 同口径。
  - 打分只是代理指标, 不预测 kcat/Km。
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Iterable

LOG = logging.getLogger("dockpipeline")

_SAPROT_HINT = (
    "SaProt 打分需要 torch + SaProt 包 + Foldseek 二进制 + SaProt_*_AF2 权重; "
    "本机未部署。请在腾讯云 GPU 镜像(见 deploy/setup_v9.sh)或按 github.com/"
    "YaoYinYing/SaProt 部署的环境运行, 或改用 scoring.adapter=scores_file/esm1v/foldx。"
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
    """对一批单点突变做 SaProt 打分, 返回 {variant_id: saprot_score}。

    参数:
      wt_seq: WT 序列(1-letter, 仅用于与结构解析出的序列做一致性校验)。
      variants: 可迭代 Variant / dict。
      structure_path: WT 结构 PDB/mmCIF 绝对路径(**必填**, 见下)。
      chain: 要编码的链 ID(缺省 'A')。
      weights_dir: SaProt 权重目录(含 SaProt_650M_AF2 目录, 必填)。
      saprot_repo: SaProt 仓库根目录(即含 SaProt/ 子包的目录, 如 /opt/SaProt)。
        缺省空; 云实例上因仓库不在 sys.path 需显式传入, 本适配器会先注入再 import。
      foldseek_path: Foldseek 二进制路径(缺省 'foldseek')。
      plddt_mask: 是否对 pLDDT<阈值区域掩码结构令牌(缺省 True, AF2 结构建议开启)。
      plddt_threshold: pLDDT 掩码阈值(缺省 70.0)。
      device: 'cpu'/'cuda'/'auto'(缺省自动探测)。

    说明: SaProt 打分依赖结构。structure_path 应给 **WT 结构**(如 AlphaFold 模型
    CIF, 含 B-factor pLDDT), 而非对接后的 pose; 结构链需覆盖 variants 里的 site。
    """
    # SaProt 仓库根(含 SaProt/ 子包)通常不在 sys.path —— 先注入再 import(2026-10-10
    # 云实例实测: /opt/SaProt 仓库 clone 存在、包在 /opt/SaProt/SaProt/, 但不注入则
    # `import SaProt` 失败)。兼容直接给仓库根或内层 SaProt/ 目录两种写法。
    saprot_repo = str(cfg.get("saprot_repo", "") or "")
    for _p in ([saprot_repo, os.path.join(saprot_repo, "SaProt")]
               if saprot_repo else []):
        if _p and os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        import torch  # noqa: F401
        from SaProt.model.saprot.saprot_foldseek_mutation_model import \
            SaprotFoldseekMutationModel
        from SaProt.utils.foldseek_util import get_struc_seq
        from SaProt.utils.weights import PretrainedModel
    except ImportError as e:
        raise RuntimeError(_SAPROT_HINT) from e

    weights_dir = str(cfg.get("weights_dir", "") or "")
    structure_path = str(cfg.get("structure_path", "") or cfg.get("receptor_pdb", "") or "")
    chain = str(cfg.get("chain", "A") or "A")
    foldseek_path = str(cfg.get("foldseek_path", "foldseek") or "foldseek")
    plddt_mask = bool(cfg.get("plddt_mask", True))
    plddt_threshold = float(cfg.get("plddt_threshold", 70.0) or 70.0)

    if not weights_dir:
        raise RuntimeError("libscore_saprot 需要 weights_dir(SaProt 权重目录)")
    if not structure_path:
        raise RuntimeError(
            "libscore_saprot 需要 structure_path(WT 结构 PDB/mmCIF 绝对路径); "
            "SaProt 是结构感知模型, 必须先由 Foldseek 把结构编码成 3Di 令牌序列")
    from dockbuddy_common import resolve_device
    device = resolve_device(str(cfg.get("device", "auto") or "auto"))

    # 1) 加载模型。直接拼 config_path, 绕过 SaProt 的 PretrainedModel —— 其
    #    _fetch_model 有 bug(`self._fetch_model(self.model_name, force_redownload=True)`
    #    多传位置参数 → TypeError), 且权重已由 COS 内网拉到位无需 HuggingFace 下载。
    config_path = os.path.join(weights_dir, "SaProt_650M_AF2")
    model = SaprotFoldseekMutationModel(
        foldseek_path=foldseek_path, config_path=config_path, load_pretrained=True)
    model.eval().to(device)

    # 2) Foldseek 把 WT 结构编码成「结构感知序列」(残基+3Di 令牌对)
    parsed = get_struc_seq(foldseek_path, structure_path, [chain],
                           plddt_mask=plddt_mask, plddt_threshold=plddt_threshold)
    if chain not in parsed:
        raise RuntimeError(
            f"libscore_saprot: 结构 {structure_path} 中未找到链 {chain!r} "
            f"(可用链: {sorted(parsed)})")
    seq, _struc_seq, combined_seq = parsed[chain]

    # 3) WT 序列一致性校验(结构解析出的 seq vs 传入 wt_seq)
    wt = "".join(str(wt_seq).split()).upper()
    if wt and wt != seq.upper():
        LOG.warning(
            "libscore_saprot: 传入 wt_seq(%d aa) 与结构链 %s 解析序列(%d aa)不一致; "
            "以结构解析序列为准", len(wt), chain, len(seq))

    out: Dict[str, float] = {}
    for v in variants:
        vid, wt_aa, mt, site = _fields(v)
        mut = f"{wt_aa}{site}{mt}"
        if site < 1 or site > len(seq):
            LOG.warning("saprot 打分跳过越界位点 %s(链长 %d): %s", mut, len(seq), vid)
            continue
        if seq[site - 1].upper() != wt_aa.upper():
            LOG.warning("saprot 打分跳过位点残基不符 %s(结构=%s): %s",
                        mut, seq[site - 1], vid)
            continue
        try:
            val = model.predict_mut(combined_seq, mut)
            if hasattr(val, "item"):           # tensor → 标量
                val = val.item()
            out[vid] = float(val)
        except Exception as e:  # noqa: BLE001
            LOG.warning("saprot 打分失败 %s: %s", vid, e)
    return out


__all__ = ["score_variants"]
