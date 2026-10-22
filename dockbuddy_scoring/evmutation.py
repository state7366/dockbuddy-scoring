# -*- coding: utf-8 -*-
"""
libscore_evmutation.py —— EVmutation(共进化统计能量) 打分适配器(2026-10-08)
===================================================================================
定位: 预筛管线「打分」步骤的进化线索适配器 —— 用 EVmutation(Marks lab, Nat
Biotechnol 2017)从 MSA 推断的无向图模型算单点突变的统计能量差(ΔE), 输出
{variant_id: {evmutation_score: float}}。

接入方式(经联网核实, 无托管 REST API, 仅本地部署):
  - pip `evcouplings`(MIT) + `plmc` 二进制(C 编译) + `jackhmmer` + UniRef 序列库。
  - 流程: jackhmmer 搜 a3m → plmc 训练图模型(J/h 参数) → 算各单点突变 ΔE。
  - 参考: github.com/debbiemarkslab/EVcouplings(已取代 legacy EVmutation 仓库)。

设计决策(诚实):
  - **零依赖 import**: evcouplings/plmc 全部延迟 import; 本机未部署时抛 RuntimeError
    (带部署指引), 与 esm1v/saprot 同口径。
  - **未在本机实测**: 本机无 evcouplings+plmc+数据库, 真跑需腾讯云 GPU 镜像部署
    (见 deploy/setup_v9.sh); 调用约定按官方 notebooks 校准, 部署后需验证。
  - **与 libdock.dca 的分工**: dca.py 给的是 MI+APC 残基对共进化耦合(弱信号);
    EVmutation 给的是 plmDCA 级图模型的单点突变统计能量(更严格), 是它的升级版。
  - 打分只是代理指标, 不预测 kcat/Km。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

LOG = logging.getLogger("dockpipeline")

_EV_HINT = (
    "EVmutation 打分需要 evcouplings(pip, MIT) + plmc 二进制 + jackhmmer + UniRef "
    "序列数据库; 本机未部署。请在腾讯云 GPU 镜像(见 deploy/setup_v9.sh)或已装 "
    "evcouplings 的环境运行, 或改用 scoring.adapter=scores_file/esm1v/foldx。"
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
    """对一批单点突变做 EVmutation 统计能量打分, 返回 {variant_id: evmutation_score}。

    参数:
      wt_seq: WT 序列(1-letter)。
      variants: 可迭代 Variant / dict。
      model_params: 已训练 EVmutation 模型参数目录(plmc 从 a3m 训练产出, 必填)。
    返回: {variant_id: 统计能量差 ΔE}(正值 = 能量升高 = 突变更不利于折叠/功能;
    负值 = 能量降低 = 突变更被家族序列支持)。
    """
    try:
        import evcouplings  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_EV_HINT) from e

    model_params = str(cfg.get("model_params", "") or "")
    if not model_params:
        raise RuntimeError(
            "libscore_evmutation 需要 model_params(plmc 从 a3m 训练出的 EVmutation "
            "模型参数文件 .params)。训练流程: a3m→a2m(去小写插入段)→plmc -f query "
            "-le 16 -lh 0.01 -m 100 -g 训练, 见 deploy/ 与 evcouplings 官方 notebooks。")

    # 官方 API(实测校准, evcouplings 1R19):
    #   CouplingsModel(params) + predict_mutation_table(model, df[mutant])
    import pandas as pd
    from evcouplings.mutate import CouplingsModel, predict_mutation_table

    model = CouplingsModel(model_params)
    muts: list = []
    vids: list = []
    for v in variants:
        vid, wt, mt, site = _fields(v)
        muts.append(f"{wt}{site}{mt}")
        vids.append(vid)

    table = pd.DataFrame({"mutant": muts})
    result = predict_mutation_table(model, table)

    out: Dict[str, float] = {}
    for vid, val in zip(vids, result["prediction_epistatic"]):
        try:
            fv = float(val)
        except (TypeError, ValueError):
            continue                       # NaN(突变未覆盖/无效) 跳过
        out[vid] = fv
    return out


__all__ = ["score_variants"]
