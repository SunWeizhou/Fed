# OOD 信息与协方差谱结构不一致的问题说明

日期：2026-03-19

## 1. 当前想讨论的核心问题

我们在当前联邦 OOD 检测实验中观察到，一个关键现象是：

**OOD 检测所需的“有效主子空间”并不一定与协方差矩阵按特征值大小排序得到的主成分子空间一致。**

换句话说，ViM 体系默认采用的 PCA 主子空间假设，只能部分刻画 ID 数据的正常结构，但并不足以稳定解释 Near-OOD / Far-OOD 的分离机制。

这意味着，问题不应被简单理解为“选择多大的 `k`”，而更应理解为：

**如何找到与 ID/OOD 结构分布真正匹配的方向集合。**

## 2. 为什么这个问题值得单独讨论

ViM 及其变体都依赖一个共同前提：

1. 先构造一个主子空间。
2. 再把离开该子空间的残差视为 OOD 信号。

但从当前实验看，这个前提存在明显局限：

1. 高方差方向不一定就是最有 OOD 判别力的方向。
2. ACT 找到的统计显著信号维度，也不一定就是 OOD 检测最需要保留的方向。
3. 原始 `ViM-paper` 在一些模型上表现好，更可能是因为它保留了足够宽的方向覆盖，而不是因为它准确找到了“真正的 OOD 主子空间”。

## 3. 当前使用过的 ViM 系方法

本轮讨论中，实际重点比较的是以下几种 ViM 系方法：

1. `ViM-paper`
   - 使用原文 heuristic 固定 `k`
   - 子空间方向来自协方差矩阵 PCA
2. `Variance-0.90 / 0.95 / 0.99`
   - 使用方差贡献率选择 `k`
   - 子空间方向来自协方差矩阵 PCA
3. `ACT-FedViM`
   - ACT 只用于选择 `k`
   - 方向仍然来自协方差矩阵 PCA
4. `ACT-corr-reordered`
   - 在相关矩阵空间中做 ACT 修正
   - 按修正谱重排方向
   - 残差在标准化特征空间中计算
5. `ACT-corr soft spectral`
   - 不再硬截断到单一 `k`
   - 使用连续谱权重构造 soft residual
   - 当前保留了 `g=1` 和 `g=5` 两个版本用于分析

补充说明：

- `virtual-logit` 公平对照已经做过，结果表明在固定同一子空间和 residual 的前提下，`Energy - alpha * Residual` 与显式 `virtual logit + softmax p0` 在 AUROC 上基本等价，因此 **virtual-logit 不是当前性能瓶颈**。

## 4. `ACT-corr soft spectral` 的原理

`ACT-corr soft spectral` 是在 `ACT-corr-reordered` 基础上的一个连续化版本。它的出发点是：

- 原始 ACT 的硬截断会把方向分成两类：
  - 主子空间方向
  - 残差方向
- 但在当前任务里，很多接近阈值的边界方向未必是纯噪声，也未必应该被完全保留。

因此，`soft spectral` 不再用单一硬阈值直接决定“保留前 `k` 维、丢弃其余维”，而是对全部方向施加一个连续权重。

具体步骤是：

1. 先从联邦聚合统计量重构全局协方差矩阵，再转换为相关矩阵。
2. 对相关矩阵做 ACT 修正，得到整条修正谱 `lambda_i^C` 以及阈值 `s`。
3. 对每个谱方向定义一个连续残差权重：

   `w_i = sigmoid(-gamma * (lambda_i^C - s))`

其中：

- 当 `lambda_i^C >> s` 时，该方向更像稳定信号方向，`w_i` 接近 `0`
- 当 `lambda_i^C << s` 时，该方向更像噪声方向，`w_i` 接近 `1`
- 当 `lambda_i^C ≈ s` 时，该方向只被部分计入残差

随后，对样本在各正交方向上的投影系数 `c_i` 做加权，得到 soft residual：

`R_soft^2 = Σ_i w_i * c_i^2`

最终仍然与能量项结合，形成 ViM-style 评分：

`Score(x) = Energy(x) - alpha * R_soft(x)`

其中 `alpha` 仍然通过 ID 训练特征经验校准得到。

这个设计的意义在于：

- 它不再假设谱边界是绝对硬切分的。
- 它允许靠近 ACT 阈值的边界方向以“部分残差、部分主空间”的方式参与 OOD 评分。
- 因而，它本质上是在测试：
  **是否可以通过连续谱加权，缓解硬 ACT 截断过于激进的问题。**

## 5. 主要实验现象

### 4.1 仅靠 `k` 的大小并不能解释性能

如果仅从 `k` 出发，会看到明显矛盾：

- `convnext_base` 上，大 `k` 方法更好  
  `ViM-paper = 95.32 (k=512)`，`Variance-0.99 = 95.41 (k=313)`，而 `ACT-corr = 94.76 (k=65)`

- `densenet169` 上，小 `k` 方法反而更好  
  `ViM-paper = 89.34 (k=1000)`，而 `ACT-FedViM = 95.87 (k=99)`，`ACT-corr = 95.29 (k=99)`

- `resnet50` 上，多种不同 `k` 的方法接近打平  
  `ViM-paper = 95.77 (k=1000)`，`ACT-corr = 95.77 (k=141)`，`Variance-0.95 = 95.68 (k=567)`

这说明：

**OOD 检测并不是一个“`k` 越大越好”或“`k` 越小越好”的单调问题。**

### 4.2 单纯扩大等效维度，也不能稳定提升性能

我们进一步尝试了 `ACT-corr soft spectral`，希望通过连续加权而不是硬截断，缓解 ACT 对维度筛选过严的问题。

结果是：

- 将 soft 权重变平，等效保留维度确实明显变大。
- 但 Near/Far-OOD AUROC 并没有稳定提升。
- 多数 CNN 上，`g=5` 反而优于更平缓的 `g=1/2/3`。

这说明：

**问题也不只是 ACT 选出的 `k` 偏小。**

即使把边界方向更宽松地保留下来，也不能保证性能更好。

### 4.3 当前更像是“方向选择原则”出了问题

综合来看，当前结果更支持以下判断：

- `ViM-paper` 强，不是因为“最大方差方向天然正确”，而是因为它保留了足够宽的经验性方向覆盖。
- `ACT` 有时失败，也不是因为随机矩阵理论完全无用，而是因为“统计显著性”不等于“OOD 判别相关性”。
- 当前 OOD 任务真正需要的主子空间，更像是一个与 ID/OOD 结构分布相匹配的方向集合，而不是单纯按谱强度排序得到的主成分集合。

## 6. 图示证据

下面四张散点图展示了 `k` 与 Near/Far-OOD AUROC 的关系。  
横轴是子空间维度 `k`；对于 soft spectral 方法，横轴使用其等效维度 `effective_signal_dim`。

### 5.1 CNN - Near-OOD

![CNN Near-OOD K Scatter](../../paper_tools/figures/vim_k_scatter_cnn_near_focus93_notext.png)

### 5.2 CNN - Far-OOD

![CNN Far-OOD K Scatter](../../paper_tools/figures/vim_k_scatter_cnn_far_focus93_notext.png)

### 5.3 Transformer - Near-OOD

![Transformer Near-OOD K Scatter](../../paper_tools/figures/vim_k_scatter_transformer_near_focus93_notext.png)

### 5.4 Transformer - Far-OOD

![Transformer Far-OOD K Scatter](../../paper_tools/figures/vim_k_scatter_transformer_far_focus93_notext.png)

从图上可以直接看到两点：

1. 不同模型的高性能点分布在非常不同的 `k` 区间。
2. 高 AUROC 点并没有稳定贴着某一种统一的谱选择原则分布。

这进一步支持：

**PCA 主成分子空间的“普适结构”并不存在。**

## 7. 当前最稳妥的结论

基于现有实验，我认为可以向导师汇报以下结论：

1. ViM 体系中的主子空间并不能简单地由协方差谱结构决定。
2. OOD 检测目标关注的是“哪些方向能稳定表征 ID 正常结构，并把 OOD 扰动推入残差空间”，这和“哪些方向解释最大方差”不是同一个问题。
3. 原始 `ViM-paper` 在部分模型上表现强，更多体现为一种宽覆盖 heuristic，而不是谱结构理论已经充分解决了主子空间定义问题。
4. `ACT-corr-reordered` 是当前最有希望的 ACT 改进方向，但它仍未统一解决“主子空间定义”问题。
5. `soft spectral` 的尝试表明，单纯把 ACT 变得更软、更宽，并不能自动得到更好的 OOD 子空间。

## 8. 下午汇报时可直接使用的一句话

> 当前实验表明，OOD 检测所需的主子空间并不等同于按协方差谱强度排序得到的 PCA 主子空间；问题的关键不只是选多大的 `k`，而是如何找到与 ID/OOD 结构分布真正匹配的方向集合。

## 9. 相关结果文件

- 主结果表：[grouped_ood_table.md](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/grouped_ood_table.md)
- 绘图脚本：[plot_vim_k_scatter.py](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/plot_vim_k_scatter.py)
- ACT 重排实验脚本：[evaluate_act_reordered_spaces.py](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/evaluate_act_reordered_spaces.py)
- ACT soft spectral 实验脚本：[evaluate_act_soft_spectral.py](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/evaluate_act_soft_spectral.py)

## 10. WDiscOOD 的理论内容

除了 ViM 体系，我们还补充考察了 `WDiscOOD` 及其相关变体。  
它的核心并不是 PCA 主子空间，而是通过 **Whitened LDA (WLDA)** 对高维特征空间做一次统计判别式分解。

### 10.1 第一阶段：统计准备（Whitened LDA）

目标是利用 ID 训练特征，构造一个能够将特征空间分解为“判别方向”和“残余方向”的投影体系。

#### 10.1.1 数据白化

首先计算类内散度矩阵：

`S_w = Σ_i (z_i - μ_{c_i})(z_i - μ_{c_i})^T`

然后对其做特征分解：

`S_w = V_w Λ_w V_w^T`

得到白化变换：

`x = S_w^{-1/2} z = V_w Λ_w^{-1/2} V_w^T z`

白化的作用是：

- 消除类内相关性
- 让特征空间更接近各向同性
- 提升后续判别分析的数值稳定性

#### 10.1.2 判别方向求解

在白化空间中，进一步构造类间散度 `S_b`，求解 Fisher 判别准则：

`J(w) = (w^T S_b w) / (w^T S_w w)`

实际实现中会加入一个很小的正则项，求解：

`(S_w + ρI)^{-1} S_b w = λ w`

其中，特征值越大，表示该方向上的类间可分性越强。

### 10.2 第二阶段：空间划分

WDiscOOD 将白化后的特征空间分成两个互补子空间。

#### 10.2.1 判别子空间 `W_D`

取前 `N_D` 个最大判别式方向组成矩阵 `W = [w_1, ..., w_{N_D}]`，并定义：

`g(x) = W^T x`

这个空间的作用是：

- 最大化 ID 类别之间的分离
- 让不同 ID 类在该空间中形成更清晰的簇结构

对于 OOD 检测，计算样本到最近 ID 类中心的距离：

`s_g(x) = - min_c || g(x) - μ_c^{WD} ||_2`

如果样本在这个最具判别力的空间里离所有 ID 类都很远，它很可能是 OOD。

#### 10.2.2 残余子空间 `W_DR`

再考虑 `W_D` 的正交补空间，也就是残余子空间。设投影基为 `Q`，则：

`h(x) = (I - QQ^T) x`

这个空间的作用是：

- 去除强类别判别方向
- 保留 ID 数据共享的背景统计结构

对应的 OOD 分数项为：

`s_h(x) = - || h(x) - μ^{WDR} ||_2`

它衡量的是：

**样本是否符合 ID 数据在非判别方向上的通用统计背景。**

### 10.3 WDiscOOD 的融合打分

原始 `WDiscOOD` 分数是这两部分的线性组合：

`s(x) = s_g(x) + α s_h(x)`

在我们当前的实现和对比中，实际出现了三种 WDisc 系方法：

1. `Pure WDiscOOD`
   - 使用原始 WLDA 几何分数
   - 公式近似为：`-(disc_dist + λ * resid_dist)`
2. `WDiscOOD`
   - 在几何分数基础上再融合 `Energy`
   - 公式近似为：`Energy - α * (disc_dist + λ * resid_dist)`
3. `WDisc-Energy`
   - 丢弃判别子空间项，只保留残余子空间项与 `Energy`
   - 公式近似为：`Energy - γ * resid_dist`

这里一个很重要的区别是：

- ViM 系方法主要依赖一阶、二阶聚合统计量
- WDisc 系方法依赖**按类别聚合的统计量**

因此，WDisc 更适合被描述为：

**label-aware federated aggregated methods**

而不是完全 label-free 的 Fed-ViM 风格方法。

## 11. 三种 WDisc 方法与三种 ViM 方法的结果对比

下表给出当前 6 种代表方法在 9 个模型上的平均表现：

| 方法 | 方法类别 | Near 平均 AUROC | Far 平均 AUROC | 综合平均 | 单模型夺冠次数 |
|------|----------|-----------------|----------------|----------|----------------|
| Pure WDiscOOD | WDisc | 95.85% | 96.29% | 96.07% | 5/9 |
| WDisc-Energy | WDisc | 94.90% | 95.14% | 95.02% | 3/9 |
| WDiscOOD | WDisc | 94.24% | 94.45% | 94.35% | 1/9 |
| ACT-FedViM | ViM | 93.79% | 93.70% | 93.75% | 0/9 |
| ACT-corr-reordered | ViM | 93.75% | 93.67% | 93.71% | 0/9 |
| ViM-paper | ViM | 93.91% | 93.13% | 93.52% | 0/9 |

从这个表可以看出：

1. 当前三种 WDisc 方法整体都强于三种 ViM 方法。
2. `Pure WDiscOOD` 是当前最强的方法，说明 WLDA 的几何分解在这套数据上非常有效。
3. `WDisc-Energy` 也很强，说明“只保留残余子空间 + Energy”的策略具有稳定价值。
4. ViM 系方法的主要问题仍然是：主子空间的定义方式不足以稳定对应 OOD 判别目标。

这也进一步反衬出本文当前真正的核心问题：

**OOD 检测所需的信息结构，并不一定与 PCA/协方差谱排序得到的主子空间一致。**
