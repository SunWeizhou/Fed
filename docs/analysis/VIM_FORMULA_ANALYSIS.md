# ViM Score Formula Analysis and Verification

## Executive Summary

当前仓库采用的是**联邦化 ViM-style ID score**，统一公式为：

$$
\text{Score}(x) = \text{Energy}(x) - \alpha \cdot \text{Residual}(x)
$$

其中：

- `Energy(x) = logsumexp(logits(x))`
- `Residual(x) = ||(I - PP^T)(f(x) - \mu_global)||_2`
- `alpha` 由 ID 训练特征经验估计

该分数在仓库中被当作 **ID score** 使用：

- 分数更高：更可能是 ID
- 分数更低：更可能是 OOD

在计算 `AUROC / AUPR / FPR95` 时，会对该分数取反，转成 OOD score。

## Why This Form Works

对 ID 样本而言：

- `Energy` 通常更高
- `Residual` 通常更低

因此 `Energy - alpha * Residual` 更大。

对 OOD 样本而言：

- `Energy` 通常更低
- `Residual` 通常更高

因此该分数更小。

一个简单示例：

| Metric | ID | OOD |
|--------|----|-----|
| Energy | 6.0 | 3.0 |
| Residual | 2.0 | 8.0 |
| Score = E - αR (`α=3`) | 0 | -21 |

这说明当前实现中，**ID score 高于 OOD score**。

## Relation to Original ViM

这不是“逐字复现原论文 virtual-logit pipeline”，而是针对联邦场景做的 ViM-style 改造：

- 使用 `mu_global` 做中心化
- 使用联邦聚合统计量重构全局协方差
- 正式结果使用经验 `alpha`，解析 `alpha` 仅作为快速近似/对照
- 在指标计算时将 ID score 映射成 OOD score

## Repository Convention

当前代码口径应统一理解为：

```python
energy = torch.logsumexp(logits, dim=1)
residual = torch.norm((features - mu_global) - ((features - mu_global) @ P) @ P.T, dim=1)
score_id = energy - alpha * residual
score_ood = -score_id
```

## Practical Note

`Energy-only` baseline 仍可能使用 `-logsumexp(logits)` 作为单独的 OOD 分数，但那是 **Energy baseline** 的实现细节，不应与 ViM-style score 混淆。
