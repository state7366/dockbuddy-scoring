# dockbuddy-scoring

DockBuddy 的蛋白突变打分适配器（26 个模型），从 `pipeline/scoring/` 抽出的解耦版本。

## 安装

```bash
pip install dockbuddy-scoring \
  --index-url https://pypi.cnb.cool/vcinex/bioinfo/-/packages/simple
```

依赖 `dockbuddy-common`（通用工具）。各模型的 torch/transformers 等按需懒加载。

## 使用

各适配器入口函数名不同，按子模块导入：

```python
from dockbuddy_scoring.esm2 import esm2_scores
from dockbuddy_scoring.esm import esm1v_scores
from dockbuddy_scoring.saprot import score_variants
from dockbuddy_scoring.foldx import score_variants
from dockbuddy_scoring.foldx_engine import foldx_ddg, FoldXError
```

## 模型清单

| 类别 | 模型 |
|---|---|
| 纯序列 | esm1v / esm2 / venusplm / venusrem / tranception |
| MSA | ddgun / lfb / evmutation / msa_transformer / profam |
| 结构 | esm_if / foldx / saprot / thermompnn / solubility_mpnn / catpred / geostab |
| blocked/restricted | promep / deepsequence / eve / unirep / muformer / rosetta |

权重路径可由 `dockbuddy-models` 解析（`hub.weight_path(...)`），本包只负责打分逻辑。
