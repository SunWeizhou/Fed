# Fed-ViM 文档快速参考（2025-02-20 更新版）

## 文档结构

### 1. `docs/paper/ACT_FedViM_论文草稿.md`（621 行）
**用途**：学术论文草稿，适合投稿 NeurIPS/ICML/ICLR/TPAMI

**主要章节**：
- 摘要：背景、方法、结果、结论
- 引言：研究背景、现有方法局限性、本文贡献
- 相关工作：联邦学习、OOD 检测、随机矩阵理论
- 方法：问题设定、ACT-FedViM 框架、隐私分析、复杂度分析
- 实验：数据集、实验设置、**主要结果**、**ACT 性能分析**、消融实验、可视化
- 讨论与结论：贡献总结、局限性、未来工作、最佳实践建议

**核心更新**（v2.0）：
- 表 1：9 种模型的完整性能表格
- 表 2：ACT 子空间压缩效果与 AUROC 提升
- 4.4.1 节：架构类型差异分析（CNN vs Transformer）
- 4.4.2 节：Top 提升模型的详细分析
- 4.4.3 节：ACT vs. 方差贡献率对比
- 5.3 节：最佳实践建议（模型选择指南）
- ViM 公式：修正为 `energy - α × residual`（原版）

---

### 2. `docs/analysis/FedViM_数学原理分析.md`（844 行）
**用途**：技术文档，深入理解算法原理和实现细节

**主要章节**：
1. 联邦学习数学基础：问题设定、FedAvg、统计量聚合
2. ViM OOD 检测原理：主子空间提取、残差计算、能量函数、评分函数、Alpha 校准
3. 随机矩阵理论与 ACT 算法：Marchenko-Pastur 分布、Stieltjes 变换、ACT 详解
4. 浮游生物数据集应用：**最新实验结果**、ACT 性能分析、数据集特性、模型配置
5. 关键数学公式汇总：核心公式、重要推导、量纲分析
6. 实践指南与最佳实践：模型选择、ACT 配置、故障排查

**核心更新**（v2.0）：
- 表 1：9 种模型的完整性能总结（包含 ACT k 和压缩率）
- 4.2 节：ACT 性能提升分析（按架构类型和原始 k 分类）
- 表 2：模型特定配置（auto-configured）
- 5.2 节：重要推导（协方差重构、残差统计、Alpha 与 k 的关系）
- 5.3 节：量纲一致性分析（典型数值表格）
- 6 节：实践指南与最佳实践（新增章节）

---

### 3. `docs/analysis/UPDATE_SUMMARY_2025-02-20.md`（195 行）
**用途**：更新摘要，快速了解最新变化

**主要内容**：
- 更新文件列表
- 主要更新内容（5 大部分）
- 下一步建议（论文投稿、补充实验、代码发布、可视化）

---

## 核心数据速查

### 最佳性能模型

| 指标 | 模型 | 数值 |
|------|------|------|
| **ID 准确率** | ResNet101 | 96.97% |
| **Near-OOD AUROC** | ConvNeXt-Base | 97.02% |
| **Far-OOD AUROC** | MobileNetV3-Large | 97.37% |
| **最高 ACT 提升** | EfficientNetV2-S | +8.72% |
| **最大压缩率** | EfficientNetV2-S | 92.6% |
| **最快推理加速** | EfficientNetV2-S | 18.8× |

### 架构类型对比

| 架构 | 平均 Near-OOD 提升 | 适用场景 |
|------|-------------------|---------|
| **CNN** | +3.61% | 高冗余特征空间，ACT 效果显著 |
| **Transformer** | ±0.17% | 原始 k 已较小，ACT 主要用于压缩 |
| **轻量级** | +0.18% | 边缘设备，性能稳定 |

### ACT 收益分类

| 原始 k 范围 | 平均提升 | 典型模型 |
|------------|---------|---------|
| **大 k（≥500）** | +4.95% | ResNet101/50, DenseNet169, EfficientNetV2-S |
| **中 k（300-500）** | +1.98% | ConvNeXt-Base |
| **小 k（<300）** | +0.12% | ViT, DeiT, MobileNetV3 |

---

## 关键公式

### ViM 评分函数（当前仓库）
```
ViM-Score(x) = energy(x) - α × residual(x)
```
- `energy(x) = logsumexp(logits) ∈ [0, log C]`（有界）
- `residual(x) = ‖(I - PPᵀ)(z - μ)‖₂ ∈ [0, ∞)`（无界）
- `α = |E̅_train| / R̅_train`：由 ID 训练特征经验估计的平衡系数

### ACT 阈值
```
s = 1 + √(p / (n - 1))
```
- `p`：特征维度
- `n`：训练样本数
- 典型值：1.17 - 1.28（Plankton 数据集）

### ACT 修正特征值
```
λⱼᴄ = -1 / m̅ₙⱼ(λⱼ)
```
- `m̅ₙⱼ(z) = -(1 - ρⱼ)/z + ρⱼ mₙⱼ(z)`
- `ρⱼ = (p - j) / (n - 1)`

### Alpha 与 k 的关系
```
E[Rₖ²] = Σᵢ₌ₖ₊₁ᴰ λᵢ
αₖ = |E̅| / √(E[Rₖ²])
```
当 k 增加时，αₖ 必须增大。

---

## 最佳实践速查

### 模型选择决策树

```
你的需求是什么？
├─ 最佳综合性能 → ResNet101
├─ 最高 Near-OOD → ConvNeXt-Base（注意：freeze_bn=1）
├─ 最高 Far-OOD → MobileNetV3-Large（训练最快）
├─ 计算资源受限 → EfficientNetV2-S（压缩率 92.6%）
├─ 边缘设备部署 → MobileNetV3-Large（轻量级）
└─ Transformer 偏好 → DeiT-Base（稳定 96%+）
```

### 配置检查清单

- [ ] ConvNeXt/Swin：`freeze_bn=1`（关键！）
- [ ] DeiT-Base：`image_size ≤ 224`（限制）
- [ ] 大模型：使用梯度累积（`accumulation_steps = 6-8`）
- [ ] ACT：CNN 模型强烈推荐（+3.61%），Transformer 可选（±0.17%）

### 故障排查速查表

| 症状 | 常见原因 | 解决方案 |
|------|---------|---------|
| **AUROC < 0.5** | freeze_bn=0（ConvNeXt/Swin） | 设置 freeze_bn=1 |
| **AUROC < 0.5** | ViM 公式错误 | 使用 `energy - α × residual` |
| **AUROC < 0.5** | Energy 计算错误 | 使用 `logsumexp` 而非 `max_logit` |
| **AUROC 0.6-0.8** | 训练不充分 | 增加轮数或检查学习率 |
| **AUROC 0.6-0.8** | 子空间维度过大 | 使用 ACT 重新计算 k |
| **Near-OOD << Far-OOD** | 特征相似度高 | 使用更大容量模型 |

---

## 文档使用指南

### 场景 1：准备论文投稿
1. 阅读 `docs/paper/ACT_FedViM_论文草稿.md`
2. 根据目标会议调整格式（NeurIPS/ICML/ICLR/TPAMI）
3. 引用最新的实验结果（表 1、表 2）
4. 强调 ACT 的理论保证和实验验证

### 场景 2：理解算法原理
1. 阅读 `docs/analysis/FedViM_数学原理分析.md` 的第 2-3 章
2. 关注第 5 章的数学推导和量纲分析
3. 实现时参考代码位置标注（如 `server.py:327`）

### 场景 3：选择最佳模型
1. 查看 `docs/analysis/FedViM_数学原理分析.md` 的 6.1 节（模型选择推荐）
2. 参考 `docs/analysis/UPDATE_SUMMARY_2025-02-20.md` 的最佳实践建议
3. 根据计算资源和性能需求权衡

### 场景 4：调优和故障排查
1. 查看 `docs/analysis/FedViM_数学原理分析.md` 的 6.4 节（故障排查）
2. 检查关键配置（freeze_bn、image_size、accumulation_steps）
3. 参考 Alpha 与 k 的关系，重新校准 α

---

## 更新时间线

- **2025-02-20**（v2.0）：
  - 更新为最新的 9 模型实验结果
  - 添加 ACT 性能分析章节（架构差异、k 分类）
  - 修正 ViM 公式为 `energy - α × residual`（原版）
  - 添加量纲分析与实践指南
  - 创建快速参考文档

- **2025-02-17**（v1.0）：
  - 初版完成
  - 包含 7 种模型的实验结果
  - 基础数学原理分析

---

## 相关文件

- 论文草稿：`docs/paper/ACT_FedViM_论文草稿.md`
- 数学分析：`docs/analysis/FedViM_数学原理分析.md`
- 更新摘要：`docs/analysis/UPDATE_SUMMARY_2025-02-20.md`
- 快速参考：`docs/analysis/QUICK_REFERENCE_2025-02-20.md`

---

**版本**：v2.0
**日期**：2025-02-20
**Git commit**：0b6135d
**维护者**：Fed-ViM 项目组
