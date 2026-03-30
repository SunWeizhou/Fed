# WDiscOOD Sandbox

`wdiscood/` 收纳当前项目中所有非主线的 WDiscOOD / WDisc-Energy 探索脚本与结果。

这里的内容包括：

- `evaluate_wdiscood.py`
- `evaluate_wdisc_energy.py`
- `grid_search_gamma.py`
- `grid_search_wdiscood_lambda.py`
- `test_act_wdiscood.py`
- `test_wdiscood_original_dims.py`
- `test_centering_bug.py`
- `evaluate_wdiscood_auto_alpha.py`
- `verify_wdiscood_fisher_tail.py`
- `results/`

说明：

- 这些脚本不属于当前论文主线的 `Fed-ViM + empirical alpha` 正式结果链路。
- 当前正式主线仍以根目录训练代码、`advanced_fedvim.py`、`evaluate_baselines.py` 和 `paper_tools/` 主表生成链路为准。
- 若直接运行本目录脚本，请在仓库根目录执行，例如：
  `python3 wdiscood/evaluate_wdiscood.py ...`
