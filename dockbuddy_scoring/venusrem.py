# -*- coding: utf-8 -*-
"""
libscore_venusrem.py —— VenusREM(检索增强 PLM) 打分适配器(2026-10-08)
===================================================================================
定位: 预筛管线「打分」步骤的检索增强适配器 —— 用 VenusREM(ISMB/ECCB 2025, ProteinGym
榜首)对单点突变做 zero-shot 打分, 输出 {variant_id: {venusrem_score: float}}。相比
单模型 PLM, VenusREM 检索同源序列(MSA)+结构(Foldseek)增强表示, 精度更高。

接入方式(经联网核实, 无编程 API, pip `vrh` = VenusREM-Harness):
  - pip `vrh`(CLI + 本地 dashboard http://127.0.0.1:8765)
  - 仓库 github.com/ai4protein/VenusREM
  - 依赖: ESM-2/SaProt/ProSST 骨干 + MSA(JackHMMER/EVCouplings a2m) + Foldseek。

设计决策(诚实, 2026-10-08 更新):
  - **零依赖 import**: vrh/torch 延迟 import; 本机未部署抛 RuntimeError(带部署指引)。
  - **真跑状态(已实测)**: 在腾讯云 GPU 实例(vrh, ESM-2 650M 骨干)上对 PBATase 真实序列
    真跑通: F205G=-4.4797、S126A=-10.0905(见 `/result/scores/pbatase.csv`)。调用约定
    `vrh --fasta --mutants` 已校准, 无需 ProSST 骨干。
  - 打分只是代理指标, 不预测 kcat/Km。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

LOG = logging.getLogger("dockpipeline")

_VENUSREM_HINT = (
    "VenusREM 打分需要 vrh(pip, VenusREM-Harness) + ESM-2/SaProt/ProSST 骨干权重 + "
    "MSA(JackHMMER)+Foldseek; 本机未部署。请在腾讯云 GPU 镜像(见 deploy/setup_v9.sh)"
    "或按 github.com/ai4protein/VenusREM 部署的环境运行, 或改用 scoring.adapter="
    "scores_file/esm1v/foldx。"
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
    """对一批单点突变做 VenusREM 打分, 返回 {variant_id: venusrem_score}。

    参数:
      wt_seq: WT 序列(1-letter)。
      variants: 可迭代 Variant / dict。
      model: vrh 骨干键(缺省 esm2; 权重由 vrh 内部缓存自动定位, 无需显式 model_dir)。
    """
    try:
        import vrh  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_VENUSREM_HINT) from e

    # esm2 baseline 走 COS 内网: 设 ESM2_LOCAL_PATH 让 vrh 从本地路径加载(避免 HF 外网下载)
    import os
    _esm2_local = str(cfg.get("esm2_local_path", "") or "")
    if not _esm2_local:
        # 权重统一落位家目录(不用 root/不进 /opt); DOCK_MODELS_DIR 由 profile.d 注入
        # (默认 /home/ubuntu/dockbuddy/models)。esm2_hf 是 vrh 用的 HF 格式 esm2 骨干。
        _models_dir = os.environ.get("DOCK_MODELS_DIR", "/home/ubuntu/dockbuddy/models")
        _cand = os.path.join(_models_dir, "esm2_hf")
        if os.path.isdir(_cand):
            _esm2_local = _cand
    if _esm2_local:
        os.environ["ESM2_LOCAL_PATH"] = _esm2_local

    seq = "".join(str(wt_seq).split())
    muts = []
    for v in variants:
        vid, wt, mt, site = _fields(v)
        muts.append((vid, f"{wt}{site}{mt}"))

    # 官方 harness: vrh.api.score(fasta, mutants, model, out_dir) → DataFrame
    # (2026-10-10 云实例校准: vrh 无 api.score_mutations, 只有 api.score + api.build_score_argv)
    import os
    import tempfile
    import pandas as pd
    from vrh.api import score

    with tempfile.TemporaryDirectory() as td:
        fasta = os.path.join(td, "wt.fasta")
        with open(fasta, "w", encoding="utf-8") as f:
            f.write(f">wt\n{seq}\n")
        mutcsv = os.path.join(td, "mutants.csv")
        with open(mutcsv, "w", encoding="utf-8") as f:
            f.write("mutant\n" + "\n".join(m for _, m in muts) + "\n")
        outdir = os.path.join(td, "result")
        model = str(cfg.get("model", "esm2") or "esm2")
        try:
            df = score(fasta=fasta, mutants=mutcsv, model=model, out_dir=outdir)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"libscore_venusrem 打分失败: {e}") from e

    if df is None or df.empty:
        return {}
    mutant_col = next((c for c in df.columns
                       if str(c).lower() in ("mutant", "mutation", "variant")),
                      df.columns[0])
    score_col = next((c for c in df.columns
                      if c != mutant_col and df[c].dtype.kind in "fi"), None)
    if score_col is None:
        return {}
    label_to_vid = {m: vid for vid, m in muts}
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        m = str(row[mutant_col]).strip()
        vid = label_to_vid.get(m)
        if vid is None:
            continue
        try:
            out[vid] = float(row[score_col])
        except (TypeError, ValueError):
            continue
    return out


__all__ = ["score_variants"]
