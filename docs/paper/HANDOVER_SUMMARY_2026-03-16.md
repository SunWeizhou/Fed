# ACT-FedViM 项目交接总结

更新时间：2026-03-18

## 1. 项目当前定位

当前论文题目仍然使用 `ACT-FedViM`，但当前代码与论文口径已经统一为：

`ACT-FedViM = 在同一个 Fed-ViM checkpoint 上进行 ACT 后处理子空间修正，再执行 ViM 风格 OOD 检测。`

这不是训练阶段把 ACT 内生进联邦优化，而是两阶段流程：

1. `Stage 1`：训练联邦化 `Fed-ViM` 基线。
2. `Stage 2`：对训练好的 checkpoint 做 ACT 类后处理并重评 OOD。

## 2. 当前正式协议

- 联邦划分：`5 clients`，`Dirichlet alpha = 0.1`
- 主训练方法：`Fed-ViM`
- 正式 OOD 打分：

```python
Score(x) = Energy(x) - alpha * Residual(x)
```

- 正式 `alpha`：
  - 使用客户端 ID-train 统计量聚合得到的经验 `alpha`
  - `alpha = |mean(Energy)| / mean(Residual)`

说明：

- `alpha` 是后处理评估参数，不参与训练损失，不参与参数更新，也不参与 early stopping。
- 因此修改 `alpha` 估计方式后，重评旧 checkpoint 是学术上合理的。

## 3. 当前论文逻辑

当前论文 motivation 已经统一为以下演化路径：

1. 原始 ViM 使用较刚性的固定 `k` 或经验规则。
2. 当前联邦基线采用方差贡献率 heuristic。
3. `ACT-FedViM` 试图用更有统计依据的方式替代这个 heuristic。

同时，必须注意一个关键事实：

- 当前 ACT 并没有稳定全面优于所有 heuristic。
- 当前最强的 ACT 风格变体其实是 `ACT-corr-reordered`，而不是最初的 `ACT-FedViM`。

## 4. 当前已经验证过的方法

主比较方法：

- `Variance-0.90`
- `Variance-0.95`
- `Variance-0.99`
- `ViM-paper`
- `ACT-FedViM`
- `ACT-cov-reordered`
- `ACT-corr-reordered`
- `MSP`
- `Energy`

对应总表：

- [grouped_ood_table.md](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/grouped_ood_table.md)

### 当前主结论

- `MSP / Energy` 明显弱于 ViM 系方法。
- `ACT-cov-reordered` 无效。
- `ACT-corr-reordered` 是当前最有希望的 ACT 改进方向。
- `ViM-paper` 仍然是很强的竞争者，并且经常优于当前 ACT 变体。

## 5. 已删除的探索性分支

以下两类实验已确认不纳入当前项目主线，相关脚本与结果产物已从工作区删除：

- 一类是额外打分项分支
- 一类是相关矩阵前景裁剪 / 收缩分支

当前交接与论文叙事不再依赖这些分支。

## 6. 数据与特征层面的理解

当前对数据特征的认识已经比较清楚：

- 图像以黑/暗背景为主
- 前景稀疏
- 颜色通道高度相关
- OOD 差异更多来自形态、尺寸、结构，而不是颜色
- 类分布长尾明显

这解释了为什么：

- 当前 ACT 往往会把 `k` 选得偏小
- 相关矩阵空间通常比协方差空间更有希望
- aggressive truncation 容易伤性能

## 7. 当前建议保留与放弃的路线

### 值得保留

- `ACT-corr-reordered`

### 当前应停止投入

- `ACT-cov-reordered`

## 8. 关键文件

主训练与评估：

- [train_federated.py](/home/dell7960/桌面/FedRoD/Fed-ViM/train_federated.py)
- [server.py](/home/dell7960/桌面/FedRoD/Fed-ViM/server.py)
- [client.py](/home/dell7960/桌面/FedRoD/Fed-ViM/client.py)
- [advanced_fedvim.py](/home/dell7960/桌面/FedRoD/Fed-ViM/advanced_fedvim.py)
- [utils/ood_utils.py](/home/dell7960/桌面/FedRoD/Fed-ViM/utils/ood_utils.py)

探索性脚本：

- [evaluate_act_reordered_spaces.py](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/evaluate_act_reordered_spaces.py)

结果入口：

- [grouped_ood_table.md](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/grouped_ood_table.md)
- [nine_way_summary.md](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/nine_way_summary.md)

## 9. 当前最稳的对外表述

当前最安全的论文表述应是：

- `ACT` 提供了一种有统计依据的子空间选择思路
- `ACT-corr-reordered` 证明相关矩阵空间的 ACT 使用方式更有潜力
- 但当前实验还不能证明 ACT 稳定全面优于强 heuristic（尤其是 `ViM-paper`）

这意味着后续写论文时，应避免使用：

- “ACT 在所有 backbone 上显著优于原方法”
- “已删除的探索性分支仍然值得继续扩展”

而更适合写成：

- `ACT` 是一个有竞争力的、仍在优化中的统计驱动替代方案
