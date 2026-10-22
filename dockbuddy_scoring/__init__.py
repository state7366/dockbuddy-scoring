# -*- coding: utf-8 -*-
"""dockbuddy_scoring —— 蛋白突变打分适配器(从 DockBuddy pipeline/scoring 抽出的解耦版本)。

26 个打分模型的适配器, 入口函数名各异(esm1v_scores/esm2_scores/score_variants/...),
按子模块直接导入:
    from dockbuddy_scoring.esm2 import esm2_scores
    from dockbuddy_scoring.saprot import score_variants
    from dockbuddy_scoring.foldx import score_variants
"""
from . import msa_stats  # noqa: F401
from .foldx_engine import FoldXError, foldx_ddg  # noqa: F401

__version__ = "0.1.1"

__all__ = ["__version__", "FoldXError", "foldx_ddg", "msa_stats"]
