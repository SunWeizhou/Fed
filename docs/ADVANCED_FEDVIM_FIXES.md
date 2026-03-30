# advanced_fedvim.py 修改说明（已同步到当前实现）

## 概述

这份文档保留为历史修订说明，但其结论已经同步到当前代码口径。当前 `advanced_fedvim.py` 的关键实现如下：

1. **ACT 仅作为后处理使用**
   - 训练阶段先得到 `Original Fed-ViM` checkpoint。
   - 后处理阶段再用 ACT 在相关矩阵上确定更合适的子空间维度 `k`。
   - 最终投影矩阵仍在全局协方差上做 PCA 后取前 `k` 个方向。
   - 其理论动机是：在“原始 ViM 的固定 `k` 设定 -> 本文联邦基线的方差贡献率 heuristic”之后，进一步用随机矩阵理论缓解高维 PCA 的谱尾噪声和主方向估计不稳问题。

2. **后处理阶段不再访问训练集**
   - 当前实现使用 `create_test_loaders_only()`。
   - 评估只读取 `test / near_ood / far_ood`。
   - 所需的全局均值、协方差和样本数均从 checkpoint 中恢复。

3. **Alpha 统一改为经验校准**
   - 原始 `Fed-ViM` 与 `ACT-FedViM` 现在都调用 `utils/ood_utils.py` 中的
     `estimate_vim_alpha_empirical(...)`。
   - 正式论文结果在 ID 训练特征上重新校准 `alpha`，避免解析近似造成架构偏差。

## 当前正确口径

- `Original ViM / Fed-ViM`
  - 联邦训练得到的基线模型
  - 子空间由全局协方差的固定方差阈值策略确定
  - `alpha` 由 ID 训练特征经验估计

- `ACT-FedViM`
  - 在相同 checkpoint 上进行 ACT 后处理
  - ACT 负责确定新的 `k`
  - 新子空间仍由全局协方差 PCA 构建
  - `alpha` 同样由 ID 训练特征经验估计

## 与旧版说明的区别

这份文档早期版本曾建议：

- 使用训练集 loader 重新构造协方差
- 使用训练数据重新校准 `alpha`

这些做法已经**不再适用**当前正式评估口径。当前仓库统一采用更稳健的协议：

- 协方差来自联邦训练期间保存的充分统计量
- 后处理为获得正式结果会访问 ID 训练特征做 `alpha` 校准
- `alpha` 以经验校准为主，解析估计保留为快速近似/对照

## 使用建议

当前推荐命令保持不变：

```bash
python advanced_fedvim.py \
    --checkpoint <path/to/checkpoint.pth> \
    --data_root ./Plankton_OOD_Dataset \
    --batch_size 32 \
    --image_size 320 \
    --device cuda
```

输出结果应理解为：

- `Original ViM`：联邦化 ViM 基线
- `ACT-FedViM`：同一 checkpoint 上的 ACT 后处理增强版本
