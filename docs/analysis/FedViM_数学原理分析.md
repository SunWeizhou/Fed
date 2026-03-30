# Fed-ViM 数学原理分析

> 联邦学习框架下的 OOD 检测：基于全局 PCA 子空间构建与随机矩阵理论的子空间维度自适应选择

---

## 目录

1. [联邦学习数学基础](#1-联邦学习数学基础)
2. [ViM OOD 检测原理](#2-vim-ood-检测原理)
3. [随机矩阵理论与 ACT 算法](#3-随机矩阵理论与-act-算法)
4. [浮游生物数据集应用](#4-浮游生物数据集应用)
5. [关键数学公式汇总](#5-关键数学公式汇总)

---

## 1. 联邦学习数学基础

### 1.1 问题设定

设共有 $N$ 个客户端，每个客户端 $i$ 拥有本地数据集 $\mathcal{D}_i = \{(x_j^{(i)}, y_j^{(i)})\}_{j=1}^{n_i}$，其中 $n_i$ 为客户端 $i$ 的样本数量，$\sum_{i=1}^N n_i = n_{\text{total}}$。

全局目标函数为经验风险最小化：

$$
\min_{\theta \in \mathbb{R}^d} \mathcal{L}(\theta) = \sum_{i=1}^N \frac{n_i}{n_{\text{total}}} \mathcal{L}_i(\theta)
$$

其中 $\mathcal{L}_i(\theta) = \frac{1}{n_i} \sum_{j=1}^{n_i} \ell(f_\theta(x_j^{(i)}), y_j^{(i)})$ 是客户端 $i$ 的本地损失函数，$f_\theta$ 是由参数 $\theta$ 参数化的神经网络。

### 1.2 FedAvg 算法

FedAvg（Federated Averaging）通过迭代式的本地训练与服务器聚合实现联邦学习。

#### 1.2.1 客户端本地更新

在通信轮 $t$，客户端 $i$ 接收全局参数 $\theta_t$，执行 $E$ 轮本地 SGD 更新：

$$
\theta_{t+1}^{(i)} = \theta_t - \eta \nabla \mathcal{L}_i(\theta_t)
$$

其中 $\eta$ 是学习率。在代码实现中（`client.py:169-400`），本地训练采用以下策略：

**学习率调度（Warmup + Cosine Decay）**：

$$
\eta_t = \begin{cases}
\eta_{\text{warm}} + (\eta_{\text{base}} - \eta_{\text{warm}}) \cdot \frac{t}{T_w} & t < T_w \quad \text{(Warmup 阶段)} \\
\eta_{\min} + (\eta_{\text{base}} - \eta_{\min}) \cdot \frac{1 + \cos(\pi \cdot \frac{t - T_w}{T - T_w})}{2} & t \geq T_w \quad \text{(Cosine Decay 阶段)}
\end{cases}
$$

其中：
- $T_w = 5$ 为 warmup 轮数
- $\eta_{\min} = 0.1 \cdot \eta_{\text{base}}$ 为最小学习率
- $T = 50$ 为总通信轮数

**梯度累积**：

为处理大模型（如 ConvNeXt），代码采用梯度累积：

$$
\text{有效 Batch Size} = \text{batch\_size} \times \text{accumulation\_steps}
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{K} \sum_{k=1}^K \nabla \mathcal{L}_i(\theta_t; \mathcal{B}_k)
$$

其中 $K$ 为累积步数，$\mathcal{B}_k$ 为第 $k$ 个 mini-batch。

#### 1.2.2 服务器端聚合

服务器收集所有客户端的参数更新，执行加权平均（`server.py:71-103`）：

$$
\theta_{t+1}^{\text{global}} = \sum_{i=1}^N \frac{n_i}{n_{\text{total}}} \theta_{t+1}^{(i)}
$$

权重 $w_i = \frac{n_i}{\sum_j n_j}$ 反映了客户端数据量的贡献。

### 1.3 隐私保护的统计量聚合

Fed-ViM 的核心创新在于客户端只共享一阶和二阶统计量，而非原始特征数据。

#### 1.3.1 本地统计量计算

客户端 $i$ 在本地训练数据上计算充分统计量（`client.py:487-511`）：

$$
\begin{aligned}
s_i^{(1)} &= \sum_{x \in \mathcal{D}_i} f_\theta(x) \in \mathbb{R}^D \\
s_i^{(2)} &= \sum_{x \in \mathcal{D}_i} f_\theta(x) f_\theta(x)^T \in \mathbb{R}^{D \times D} \\
c_i &= |\mathcal{D}_i| \in \mathbb{N}
\end{aligned}
$$

其中 $f_\theta(x)$ 是网络倒数第二层的特征表示（DenseNet121 中 $D=1024$）。

#### 1.3.2 全局统计量重构

服务器聚合客户端统计量（`server.py:119-144`）：

$$
\begin{aligned}
\text{total\_count} &= \sum_{i=1}^N c_i \\
\text{global\_sum\_z} &= \sum_{i=1}^N s_i^{(1)} \\
\text{global\_sum\_zz}^T &= \sum_{i=1}^N s_i^{(2)}
\end{aligned}
$$

利用方差分解公式重构全局均值和协方差矩阵：

$$
\begin{aligned}
\mu_{\text{global}} &= \frac{\text{global\_sum\_z}}{\text{total\_count}} = \mathbb{E}[Z] \\
\mathbb{E}[ZZ^T] &= \frac{\text{global\_sum\_zz}^T}{\text{total\_count}} \\
\Sigma_{\text{global}} &= \mathbb{E}[ZZ^T] - \mu_{\text{global}}\mu_{\text{global}}^T
\end{aligned}
$$

**关键性质**：上述重构基于恒等式 $\text{Cov}(Z) = \mathbb{E}[ZZ^T] - \mathbb{E}[Z]\mathbb{E}[Z]^T$，在数学上是精确的，无需近似。

---

## 2. ViM OOD 检测原理

### 2.1 核心思想

ViM（Variance in Mahalanobis Distance）利用训练数据的特征协方差结构，通过主成分分析（PCA）提取主子空间。OOD 样本的特征在该子空间上的投影误差（残差）显著大于 ID（In-Distribution）样本。

### 2.2 主子空间提取

#### 2.2.1 特征值分解

对全局协方差矩阵 $\Sigma_{\text{global}}$ 进行特征值分解（`server.py:146-188`）：

$$
\Sigma_{\text{global}} v_k = \lambda_k v_k, \quad k = 1, \ldots, D
$$

其中 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_D \geq 0$ 是特征值，$v_k$ 是对应的特征向量。

#### 2.2.2 子空间维度选择

**方法一：方差贡献率**

选择最小的 $k$ 使得累积方差贡献率达到阈值（默认 95%）：

$$
\frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^D \lambda_i} \geq 0.95
$$

主子空间投影矩阵为：

$$
P_k = [v_1, v_2, \ldots, v_k] \in \mathbb{R}^{D \times k}
$$

**方法二：ACT 算法**

详见第 3 节。

### 2.3 残差计算

给定测试样本 $x$，提取特征 $z = f_\theta(x) \in \mathbb{R}^D$。

#### 2.3.1 中心化

$$
z_c = z - \mu_{\text{global}}
$$

#### 2.3.2 子空间投影与重构

$$
\begin{aligned}
z_{\text{proj}} &= P_k P_k^T z_c = \sum_{i=1}^k (v_i^T z_c) v_i \\
z_{\text{recon}} &= P_k P_k^T z_c
\end{aligned}
$$

$z_{\text{recon}}$ 是 $z_c$ 在主子空间上的最佳重构（L2 意义下）。

#### 2.3.3 残差（重构误差）

$$
\begin{aligned}
\text{residual}(x) &= \|z_c - z_{\text{recon}}\|_2 \\
&= \|(I - P_k P_k^T)(z - \mu_{\text{global}})\|_2 \\
&= \sqrt{(z - \mu)^T (I - P_k P_k^T)(z - \mu)}
\end{aligned}
$$

**几何解释**：残差是样本特征到主子空间的垂直距离。投影矩阵 $I - P_k P_k^T$ 将向量投影到主子空间的正交补空间（残差空间）。

### 2.4 能量函数

能量（Energy）捕获模型的整体置信度（`server.py:222-224`）：

$$
\text{energy}(x) = \log\left(\sum_{j=1}^C \exp(\text{logit}_j(x))\right)
$$

其中 $\text{logit}(x) = W f_\theta(x) + b$ 是网络输出，$C$ 是类别数（Plankton 数据集 $C = 54$）。

**数学性质**：
- $\max_j \text{logit}_j \leq \text{energy} \leq \log(C) + \max_j \text{logit}_j$
- 对于校准良好的模型，ID 样本的 energy 通常低于 OOD 样本

### 2.5 ViM 评分函数

ViM 将残差和能量组合成最终的 OOD 分数（`server.py:327`）：

$$
\text{ViM-Score}(x) = \text{energy}(x) - \alpha \cdot \text{residual}(x)
$$

其中 $\alpha > 0$ 是平衡系数。

**量纲分析**：
- $\text{energy}(x) = \log\sum_{j=1}^C \exp(\text{logit}_j(x)) \in [0, \log C]$：有界量纲
  - 对于 Plankton 数据集（$C = 54$），$\log 54 \approx 3.99$
  - 对于 CIFAR-10（$C = 10$），$\log 10 \approx 2.30$
- $\text{residual}(x) = \|(I - P_k P_k^T)(z - \mu)\|_2 \in [0, \infty)$：无界量纲
  - 取决于特征 $z$ 的尺度（通常为 L2 归一化或标准归一化）
  - 典型值范围：ID 样本 10-50，OOD 样本 50-200
- $\alpha$：平衡系数，校准两项到相同数值尺度

**检测逻辑**：
- ID 样本：小残差（接近主子空间）+ 小能量（高置信度）→ **低 ViM 分数**
- OOD 样本：大残差（远离主子空间）+ 大能量（低置信度）→ **高 ViM 分数**

$$
\text{OOD} \iff \text{ViM-Score}(x) > \tau
$$

**为什么使用 `energy - α × residual`**：

根据原版 ViM 论文（Sun et al., NeurIPS 2022），该公式的物理意义为：
- **Energy**：模型的整体置信度（越低表示越自信）
- **Residual**：特征到主子空间的距离（越大表示越异常）
- **α**：权重系数，平衡两项的数值尺度

该公式确保：
1. 高能量（低置信度）→ 高 OOD 分数
2. 高残差（远离子空间）→ 低 OOD 分数（通过减法）
3. 两项互补，共同提升检测性能

### 2.6 Alpha 系数校准

#### 2.6.1 问题动机

$\alpha$ 的作用是平衡残差和能量的数值尺度。不同模型的特征分布、不同子空间维度 $k$ 都会影响最优 $\alpha$ 值。

#### 2.6.2 基于 ID 训练特征的经验校准

当前正式实验口径已经统一改为**经验 alpha**。不论是原始 `Fed-ViM`，还是 `ACT-FedViM` 后处理，均在 ID 训练特征上重新计算平均能量与平均残差。

给定全局均值 $\mu_{\text{global}}$ 与当前子空间投影矩阵 $P$，可估计：

$$
\begin{aligned}
\bar{R}_{\text{train}} &= \frac{1}{N}\sum_{i=1}^N \|(I-PP^T)(z_i-\mu_{\text{global}})\|_2 \\
\bar{E}_{\text{train}} &= \frac{1}{N}\sum_{i=1}^N \operatorname{logsumexp}(v_i)
\end{aligned}
$$

然后统一计算

$$
\alpha = \frac{|\bar{E}_{\text{train}}|}{\bar{R}_{\text{train}} + \epsilon}
$$

其中 $\epsilon = 10^{-8}$ 是数值稳定项。该实现位于 `utils/ood_utils.py` 的 `estimate_vim_alpha_empirical(...)`。

#### 2.6.3 历史说明

仓库早期版本曾保留“聚合统计量解析校准”等近似写法；当前正式主线已经收敛为单一的经验校准方案。

##### v1.0 方法（经验公式，已弃用）

$$
\begin{aligned}
\mathbb{E}[\|R\|^2] &= \text{tr}((I - P_k P_k^T) \Sigma_{\text{global}} (I - P_k P_k^T)^T) \\
\bar{R}_{\text{est}} &= \sqrt{\max(0, \mathbb{E}[\|R\|^2])} \\
\bar{E}_{\text{est}} &= \log(C) - \delta \quad (\delta \approx 0.3 \text{ 为经验常数}) \\
\alpha_{\text{est}} &= \frac{|\bar{E}_{\text{est}}|}{\bar{R}_{\text{est}} + \epsilon}
\end{aligned}
$$

**问题**：$\log(C)$ 表示最大熵分布（完全均匀分布）的能量，与 ID 样本的平均能量无直接关系，估计不准确。

##### v2.0 方法（Jensen 不等式 + 真实分类器）

**核心思想**：将全局均值 $\mu_{\text{global}}$ 作为"虚拟平均样本"，通过真实分类器计算能量，获得 $\mathbb{E}[\text{Energy}(z)]$ 的严谨下界。

$$
\begin{aligned}
\mathbb{E}[\|R\|^2] &= \text{tr}((I - P_k P_k^T) \Sigma_{\text{global}} (I - P_k P_k^T)^T) \\
\bar{R}_{\text{est}} &= \sqrt{\max(0, \mathbb{E}[\|R\|^2])} \\
\bar{E}_{\text{est}} &= \text{Energy}(\mu_{\text{global}}) = \log\left(\sum_{j=1}^C \exp(W^T \mu_{\text{global}} + b_j)\right) \\
\alpha_{\text{est}} &= \frac{|\bar{E}_{\text{est}}|}{\bar{R}_{\text{est}} + \epsilon}
\end{aligned}
$$

**数学推导（Jensen 不等式）**：

能量函数 $f(z) = \log\sum_{j=1}^C \exp(w_j^T z + b_j)$ 是凸函数（Hessian 半正定）。根据 Jensen 不等式：

$$
\mathbb{E}[f(z)] \geq f(\mathbb{E}[z]) = f(\mu_{\text{global}})
$$

因此，$\text{Energy}(\mu_{\text{global}})$ 是 $\mathbb{E}[\text{Energy}(z)]$ 的**下界**。

对于近似高斯分布的特征，一阶近似（Jensen 下界）非常准确：

$$
\mathbb{E}[\text{Energy}(z)] \approx \text{Energy}(\mu_{\text{global}}) + \mathcal{O}(\text{Var}(z))
$$

当特征方差较小时，高阶项可忽略，该估计接近真实均值。

**实现细节**：

1. **单层线性分类器**（ResNet, EfficientNet, ConvNeXt）：

$$
\text{logits}(\mu) = W \mu + b \in \mathbb{R}^C
$$

2. **MLP 分类器**（DenseNet, Swin）：

$$
\begin{aligned}
h &= \sigma(W_1 \mu + b_1) \\
\text{logits}(\mu) &= W_2 h + b_2
\end{aligned}
$$

3. **最终能量**：

$$
\bar{E}_{\text{est}} = \text{logsumexp}(\text{logits}(\mu))
$$

**v2.0 vs v1.0 对比**（Plankton 数据集，$C=54$）：

| 方法 | 能量估计值 | 数学性质 | 准确性 |
|------|-----------|---------|--------|
| v1.0 经验公式 | $\log(54) - 0.3 \approx 3.69$ | 最大熵分布能量 | 不准确 |
| v2.0 Jensen | $\text{Energy}(\mu_{\text{global}}) \approx 2.8-3.5$ | 真实期望下界 | **严谨** |

**推导**：对于零均值高斯向量 $z \sim \mathcal{N}(0, \Sigma)$，投影到子空间 $P$ 后的残差范数平方期望为：

$$
\mathbb{E}[\|(I - PP^T)z\|^2] = \text{tr}((I - PP^T)\Sigma(I - PP^T)^T)
$$

---

## 3. 随机矩阵理论与 ACT 算法

### 3.1 问题动机

传统的 PCA 子空间维度选择基于方差贡献率，但在高维小样本（$p \gg n$）场景下，样本特征值会系统性高估真实特征值，导致过拟合。ACT（Adjusted Correlation Thresholding）通过随机矩阵理论修正这种偏差。

### 3.2 随机矩阵理论基础

#### 3.2.1 Marchenko-Pastur 分布

设 $X \in \mathbb{R}^{p \times n}$ 的元素为 i.i.d. 零均值、单位方差随机变量，样本相关矩阵 $R = \frac{1}{n} XX^T$ 的特征值在 $n, p \to \infty$ 且 $\rho = p/n \to c$ 时收敛于 Marchenko-Pastur 分布：

$$
\lambda_{\pm} = \left(1 \pm \sqrt{\rho}\right)^2
$$

对于相关矩阵（对角线为 1），噪声特征值的理论上界为：

$$
\lambda_{\max}^{\text{noise}} = (1 + \sqrt{\rho})^2
$$

#### 3.2.2 Stieltjes 变换

Stieltjes 变换是分析随机矩阵谱分布的关键工具：

$$
m_n(z) = \frac{1}{p} \sum_{j=1}^p \frac{1}{\lambda_j - z}, \quad z \in \mathbb{C}^+
$$

它描述了特征值分布的柯西变换。

### 3.3 ACT 算法详解

#### 3.3.1 协方差到相关矩阵转换（`advanced_fedvim.py:23-49`）

为消除特征尺度影响，首先将协方差矩阵转换为相关矩阵：

$$
R_{ij} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii} \Sigma_{jj}}}
$$

矩阵形式：

$$
R = D^{-1/2} \Sigma D^{-1/2}, \quad D = \text{diag}(\Sigma)
$$

**数值稳定处理**：

$$
\begin{aligned}
D_{ii}^{\text{clamped}} &= \max(D_{ii}, \epsilon), \quad \epsilon = 10^{-6} \\
R &= \frac{\Sigma}{D^{1/2} (D^{1/2})^T + \epsilon}
\end{aligned}
$$

#### 3.3.2 最优阈值计算（`advanced_fedvim.py:94-99`）

对于相关矩阵 $R$，理论阈值为：

$$
s = 1 + \sqrt{\frac{p}{n - 1}} = 1 + \sqrt{\rho}
$$

其中：
- $p$：特征维度（如 DenseNet121 的 $p = 1024$）
- $n$：训练样本总数（Plankton 训练集 $n \approx 26034$）
- $\rho = p/(n-1)$：维度-样本比

**解释**：阈值 $s$ 是纯噪声相关矩阵最大特征值的渐近上界。样本特征值若显著超过 $s$，则很可能对应真实信号。

#### 3.3.3 特征值偏差修正（`advanced_fedvim.py:101-160`）

样本特征值 $\lambda_j$ 是真实特征值 $\Lambda_j$ 的有偏估计。ACT 通过 Stieltjes 变换构造无偏估计 $\lambda_j^C$。

**步骤 1**：计算修正的 Stieltjes 变换 $m_{nj}(z)$

对于第 $j$ 个特征值（降序），定义：

$$
m_{nj}(z) = \frac{1}{p - j} \left[ \sum_{l=j+1}^p \frac{1}{\lambda_l - z} + \frac{4}{3\lambda_j + \lambda_{j+1} - 4z} \right]
$$

其中第二项是 Fan et al. (2020) 引入的修正项，处理 $z$ 接近 $\lambda_j$ 时的奇异性。

**步骤 2**：计算 $\underline{m}_{nj}(z)$

$$
\underline{m}_{nj}(z) = -(1 - \rho_j) \frac{1}{z} + \rho_j m_{nj}(z), \quad \rho_j = \frac{p - j}{n - 1}
$$

**步骤 3**：计算修正特征值

$$
\lambda_j^C = -\frac{1}{\underline{m}_{nj}(\lambda_j) + \epsilon}
$$

**直观理解**：
- 当 $\lambda_j$ 来自信号时，$m_{nj}(\lambda_j)$ 较小，$\lambda_j^C \approx \lambda_j$（几乎无修正）
- 当 $\lambda_j$ 来自噪声时，$m_{nj}(\lambda_j)$ 较大，$\lambda_j^C < \lambda_j$（向下修正）

#### 3.3.4 最优 k 确定（`advanced_fedvim.py:162-175`）

$$
k_{\text{opt}} = \max\{j : \lambda_j^C > s\}
$$

即最后一个超过阈值的修正特征值的索引。

**边界情况处理**：如果没有特征值超过阈值，则 $k_{\text{opt}} = 1$（至少保留一个主成分）。

### 3.4 数学性质分析

#### 3.4.1 一致性

Fan et al. (2020) 证明了在弱假设下，$k_{\text{opt}}$ 依概率收敛于真实因子数量 $r$：

$$
\lim_{p, n \to \infty} \mathbb{P}(\hat{k}_{\text{ACT}} = r) = 1
$$

#### 3.4.2 计算复杂度

- 特征值分解：$O(p^3)$（一次性）
- 偏差修正：$O(p^2)$（每个特征值）
- 总复杂度：$O(p^3)$（由分解主导）

对于 $p = 1024$，单次 ACT 计算约需 1-2 秒。

### 3.5 Alpha 重校准

**关键观察**：当 $k$ 改变时，残差分布变化，因此 $\alpha$ 必须重新校准（`advanced_fedvim.py:369-390`）。

**原因**：

$$
\begin{aligned}
\mathbb{E}[R_k^2] &= \text{tr}((I - P_k P_k^T) \Sigma (I - P_k P_k^T)^T) \\
&= \text{tr}(\Sigma) - \text{tr}(P_k P_k^T \Sigma) \\
&= \text{tr}(\Sigma) - \sum_{i=1}^k \lambda_i
\end{aligned}
$$

当 $k$ 增加时，$\sum_{i=1}^k \lambda_i$ 增大，$\mathbb{E}[R_k^2]$ 减小，因此需要更大的 $\alpha$。

---

## 4. 浮游生物数据集应用

### 4.1 最新实验结果（9 种模型，v2.0 Jensen 能量估计）

**表 1：完整性能总结**

| 模型 | ID 准确率 | Near-OOD AUROC | Far-OOD AUROC | ACT k | 压缩率 |
|------|-----------|----------------|---------------|-------|--------|
| ResNet101 | 96.97% | **96.92%** | 96.86% | 141 | 75.3% |
| EfficientNetV2-S | 96.67% | 96.03% | 96.49% | 68 | **92.6%** |
| ConvNeXt-Base | 96.26% | **96.92%** | 96.52% | 116 | 61.2% |
| ResNet50 | 96.60% | 96.67% | 95.54% | 138 | 75.3% |
| DeiT-Base | 96.09% | 95.96% | 96.30% | 69 | 78.2% |
| MobileNetV3-Large | 95.54% | 96.13% | **97.36%** | 90 | 78.7% |
| DenseNet169 | 95.92% | 95.13% | 96.69% | 99 | 76.0% |
| ViT-B/16 | 95.68% | 96.08% | 94.82% | 73 | 72.5% |
| ViT-B/32 | 95.37% | 95.65% | 94.21% | 71 | 73.3% |

**关键统计**：
- 平均 ID 准确率：**96.09%**（范围：95.37% - 96.97%）
- 平均 Near-OOD AUROC：**96.17%**（范围：95.13% - 96.92%）
- 平均 Far-OOD AUROC：**96.09%**（范围：94.21% - 97.36%）
- 平均 ACT 压缩率：**77.4%**（范围：61.2% - 92.6%）

> **v2.0 更新**: 使用 Jensen 不等式能量估计 `Energy(μ_global)`，相比 v1.0 经验公式，AUROC 平均变化 -0.01%（可忽略）。

### 4.2 ACT 性能提升分析

**按架构类型分类**：

| 架构类型 | 模型 | 平均原始 k | 平均 ACT k | 平均压缩率 | 平均 AUROC 提升 |
|---------|------|-----------|-----------|-----------|----------------|
| **CNN** | ResNet×2, DenseNet×1, EfficientNet×1, ConvNeXt×1 | 610 | 112 | **81.6%** | **+3.61%** |
| **Transformer** | ViT×2, DeiT×1 | 283 | 71 | **74.9%** | **+0.17%** |
| **轻量级** | MobileNetV3×1 | 422 | 90 | 78.7% | +0.18% |

**按原始 k 大小分类**：

| 原始 k 范围 | 模型 | 平均 ACT k | 平均压缩率 | 平均 AUROC 提升 |
|------------|------|-----------|-----------|----------------|
| **大 k（≥500）** | ResNet101/50, DenseNet169, EfficientNetV2-S | 112 | **81.6%** | **+4.95%** |
| **中 k（300-500）** | ConvNeXt-Base | 116 | 61.2% | +1.98% |
| **小 k（<300）** | ViT×2, DeiT, MobileNetV3 | 76 | 74.9% | +0.12% |

**Top 3 提升模型**：

1. **EfficientNetV2-S**：+8.72%（Near-OOD）
   - 特征维度：1280
   - 原始 k：922（72.0%）→ ACT k：68（5.3%）
   - 压缩率：**92.6%**
   - 推理加速：**18.8×**（从 O(1280²) 降至 O(68×1280)）

2. **DenseNet169**：+4.11%
   - 特征维度：1664
   - 原始 k：413（24.8%）→ ACT k：99（5.9%）
   - 压缩率：76.0%

3. **ResNet101**：+3.27%
   - 特征维度：2048
   - 原始 k：570（27.8%）→ ACT k：141（6.9%）
   - 压缩率：75.3%

### 4.3 数据集特性

Plankton（浮游生物）数据集具有以下特点：

| 属性 | 值 |
|------|-----|
| 类别数 $C$ | 54 |
| 训练样本数 $n$ | ~26,034 |
| 测试样本数 | ~6,508 |
| Near-OOD | 水下生物（如鱼、虾） |
| Far-OOD | 自然物体（CIFAR 类别） |
| 图像分辨率 | $320 \times 320$ |
| 标签分布 | 非 Dirichlet（长尾分布） |

**特征维度-样本比**（$\rho = p/(n-1)$）：

| 模型 | $p$ | $n$ | $\rho$ | ACT 阈值 $s$ |
|------|-----|-----|--------|-------------|
| DenseNet169 | 1664 | 26034 | 0.064 | 1.25 |
| ResNet101/50 | 2048 | 26034 | 0.079 | 1.28 |
| EfficientNetV2-S | 1280 | 26034 | 0.049 | 1.22 |
| ConvNeXt-Base | 1024 | 26034 | 0.039 | 1.20 |
| ViT/DeiT | 768 | 26034 | 0.030 | 1.17 |
| MobileNetV3 | 960 | 26034 | 0.037 | 1.19 |

小 $\rho$ 值（0.03-0.08）表明噪声水平低，ACT 阈值 $s \approx 1.17-1.28$ 接近理论下界。

### 4.2 Dirichlet 分割

为模拟联邦学习场景，训练数据按 Dirichlet 分布 $\text{Dir}(\alpha)$ 划分（`data_utils.py`）：

$$
p_{ik} \sim \text{Dir}(\alpha), \quad n_{ik} \sim \text{Multinomial}(n_i, p_i)
$$

其中：
- $n_{ik}$：客户端 $i$ 分配到类别 $k$ 的样本数
- $\alpha$：浓度参数，$\alpha$ 越小分布越不均匀

**代码实现**（$\alpha = 0.5$）：

```python
# client.py 中的实际分配
proportions = np.random.dirichlet([alpha] * num_classes, n_clients)
```

### 4.3 模型特定配置

不同架构在 Plankton 数据集上的最优配置（`config.py:135-257`）：

| 模型 | 特征维度 $D$ | freeze_bn | 批次大小 | 累积步数 | 有效 BS |
|------|--------------|-----------|----------|----------|---------|
| DenseNet121 | 1024 | 0 | 32 | 1 | 32 |
| ResNet50 | 2048 | 0 | 32 | 1 | 32 |
| ConvNeXt-Tiny | 768 | **1** | 16 | 6 | 96 |
| Swin-T | 768 | **1** | 16 | 6 | 96 |
| ViT-B/16 | 768 | 0* | 16 | 6 | 96 |

*注：ViT 使用 LayerNorm，freeze_bn 无效。

### 4.4 BatchNorm 冻结的数学解释

**问题**：ConvNeXt/Swin 在 freeze_bn=0 时 OOD 检测失效（AUROC ≈ 0.25）。

**原因**：

BatchNorm 在训练时更新：

$$
\begin{aligned}
\mu_{\text{BN}}^{(t)} &= \beta \mu_{\text{BN}}^{(t-1)} + (1 - \beta) \bar{x}^{(t)} \\
\sigma_{\text{BN}}^{2(t)} &= \beta \sigma_{\text{BN}}^{2(t-1)} + (1 - \beta) s^{2(t)}
\end{aligned}
$$

其中 $\bar{x}^{(t)}$ 和 $s^{2(t)}$ 是当前 mini-batch 的均值和方差。

对于 OOD 样本：
- 如果 BN 更新，特征会被"归一化"到 ID 的统计量
- 这导致 OOD 样本的特征也落在主子空间附近
- 残差失去区分度

**解决方案**：freeze_bn=1 使 BN 使用训练集统计量，OOD 样本特征保持"异常"。

### 4.5 数据增强

#### 4.5.1 Mixup（`client.py:142-166`）

$$
\begin{aligned}
\lambda &\sim \text{Beta}(\alpha, \alpha), \quad \alpha = 1.0 \\
\tilde{x} &= \lambda x_i + (1 - \lambda) x_j \\
\tilde{y} &= \lambda y_i + (1 - \lambda) y_j
\end{aligned}
$$

损失函数：

$$
\mathcal{L}_{\text{mixup}} = \lambda \ell(f_\theta(\tilde{x}), y_i) + (1 - \lambda) \ell(f_\theta(\tilde{x}), y_j)
$$

#### 4.5.2 傅里叶增强（`client.py:96-119`）

$$
\begin{aligned}
\mathcal{F}(x) &= \text{FFT}(x) \\
A &= |\mathcal{F}(x)|, \quad \phi = \angle \mathcal{F}(x) \\
A' &= (1 - \beta) A + \beta A_{\text{target}} \\
x_{\text{aug}} &= \text{IFFT}(A' e^{i\phi})
\end{aligned}
$$

其中 $\beta = 0.4$ 控制混合强度。

### 4.6 评估指标

#### 4.6.1 AUROC（Area Under ROC Curve）

$$
\text{AUROC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) dt
$$

其中：
- $\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$（真阳性率）
- $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$（假阳性率）

**解释**：AUROC 衡量 OOD 检测器区分 ID 和 OOD 的能力，1.0 为完美，0.5 为随机。

#### 4.6.2 分类准确率

$$
\text{Accuracy} = \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} \mathbb{I}(\hat{y}_i = y_i)
$$

---

## 5. 关键数学公式汇总

### 5.1 核心公式

| 公式 | 含义 | 位置 |
|------|------|------|
| $\theta_{t+1} = \sum_i \frac{n_i}{n_{\text{total}}} \theta_{t+1}^{(i)}$ | FedAvg 聚合 | server.py:101 |
| $\mu = \frac{1}{n}\sum z$ | 全局均值 | server.py:138 |
| $\Sigma = \mathbb{E}[ZZ^T] - \mu\mu^T$ | 全局协方差重构 | server.py:144 |
| $\Sigma v_k = \lambda_k v_k$ | 特征值分解 | server.py:152 |
| $P_k = [v_1, \ldots, v_k]$ | 主子空间投影矩阵 | server.py:176 |
| $R = \|(I - PP^T)(z - \mu)\|_2$ | 残差（重构误差） | server.py:231 |
| $E = \log\sum_j \exp(\text{logit}_j)$ | 能量（LogSumExp） | server.py:224 |
| $S = E - \alpha R$ | **ViM 分数（原版公式）** | server.py:327 |
| $\alpha = |\bar{E}_{train}| / \bar{R}_{train}$ | Alpha 校准系数（经验训练特征） | utils/ood_utils.py |
| $\bar{E}_{est} = \text{Energy}(\mu_{\text{global}})$ | 平均能量估计 | utils/ood_utils.py |
| $s = 1 + \sqrt{p/(n-1)}$ | ACT 阈值 | advanced_fedvim.py:97 |
| $\lambda_j^C = -1/\underline{m}_{nj}(\lambda_j)$ | ACT 修正特征值 | advanced_fedvim.py:158 |

### 5.2 重要推导

#### 5.2.1 全局协方差重构的精确性

$$
\begin{aligned}
\text{global\_sum\_z} &= \sum_{i=1}^N \sum_{j=1}^{n_i} z_j^{(i)} \\
\text{global\_sum\_zz}^T &= \sum_{i=1}^N \sum_{j=1}^{n_i} z_j^{(i)} (z_j^{(i)})^T \\
\mu_{\text{global}} &= \frac{\text{global\_sum\_z}}{n_{\text{total}}} = \mathbb{E}[Z] \\
\mathbb{E}[ZZ^T] &= \frac{\text{global\_sum\_zz}^T}{n_{\text{total}}} \\
\Sigma_{\text{global}} &= \mathbb{E}[ZZ^T] - \mathbb{E}[Z]\mathbb{E}[Z]^T
\end{aligned}
$$

**性质**：基于恒等式 $\text{Cov}(Z) = \mathbb{E}[ZZ^T] - \mathbb{E}[Z]\mathbb{E}[Z]^T$，在数学上是精确的，无需近似。

#### 5.2.2 残差的统计学解释

对于零均值特征 $z \sim \mathcal{N}(0, \Sigma)$：

$$
\begin{aligned}
\mathbb{E}[\|R\|^2] &= \mathbb{E}[\|(I - P_k P_k^T)z\|^2] \\
&= \text{tr}((I - P_k P_k^T) \Sigma (I - P_k P_k^T)^T) \\
&= \text{tr}(\Sigma) - \text{tr}(P_k P_k^T \Sigma) \\
&= \sum_{i=1}^D \lambda_i - \sum_{i=1}^k \lambda_i \\
&= \sum_{i=k+1}^D \lambda_i
\end{aligned}
$$

**直观理解**：残差范数平方的期望等于噪声子空间的特征值之和。

#### 5.2.3 Alpha 与 k 的关系

当 $k$ 改变时，$\alpha$ 必须重新校准：

$$
\begin{aligned}
\mathbb{E}[R_k^2] &= \sum_{i=k+1}^D \lambda_i \\
\bar{R}_k &= \sqrt{\mathbb{E}[\|R_k\|^2]} \\
\alpha_k &= \frac{|\bar{E}|}{\bar{R}_k + \epsilon}
\end{aligned}
$$

当 $k$ 增加时：
- $\sum_{i=k+1}^D \lambda_i$ 减小（噪声特征值减少）
- $\bar{R}_k$ 减小
- $\alpha_k$ **增大**（需要更大的权重平衡残差）

### 5.3 量纲一致性分析

**ViM 分数的量纲**：

$$
\text{ViM-Score} = \underbrace{\log\sum_{j=1}^C \exp(\text{logit}_j)}_{\text{量纲：}\log(\text{概率})} - \alpha \cdot \underbrace{\|(I - PP^T)(z - \mu)\|_2}_{\text{量纲：特征}}
$$

- Energy：无量纲（或理解为 $\log$-量纲）
- Residual：取决于特征的量纲（通常为 L2 归一化）
- $\alpha$：单位为 $1/\text{特征}$，将残差转换为无量纲

**校准后的典型值**（Plankton 数据集）：

| 量 | ID 样本 | OOD 样本 | 单位 |
|----|---------|----------|------|
| Energy | 2.5 - 3.5 | 4.5 - 6.0 | 无量纲 |
| Residual | 15 - 30 | 40 - 80 | 特征单位 |
| $\alpha$ | 0.10 - 0.15 | - | 1/特征单位 |
| ViM-Score | 0 - 2 | 3 - 8 | 无量纲 |

**关键点**：$\alpha$ 的作用是确保残差项与能量项在数值上可比，避免某一项主导。

---

## 参考文献

1. **FedAvg**: McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.

2. **ViM**: Sun et al. "ViM: Out-of-Distribution Detection with Variance in Mahalanobis Distance." NeurIPS 2022.

3. **ACT**: Fan et al. "Estimating the Number of Factors by Adjusted Eigenvalues Thresholding." JASA 2020.

4. **Marchenko-Pastur**: Marchenko & Pastur. "Distribution of eigenvalues for some sets of random matrices." Math. USSR-Sbornik 1967.

---

## 6. 实践指南与最佳实践

### 6.1 模型选择推荐

基于 9 种模型的全面实验，针对不同应用场景的推荐：

| 应用场景 | 推荐模型 | 理由 | 配置要点 |
|---------|---------|------|---------|
| **最佳综合性能** | ResNet101 | ID: 96.97%, Near-OOD: 96.90%, Far-OOD: 96.87% | freeze_bn=0, BS=16×4 |
| **最高 Near-OOD** | ConvNeXt-Base | AUROC: 97.02% | **freeze_bn=1**（关键）, BS=8×8 |
| **最高 Far-OOD** | MobileNetV3-Large | AUROC: 97.37%，训练速度最快 | freeze_bn=0, BS=64 |
| **计算资源受限** | EfficientNetV2-S | 压缩率 92.6%，推理加速 18.8× | freeze_bn=0, BS=32 |
| **边缘设备部署** | MobileNetV3-Large | 轻量级架构，高效推理 | freeze_bn=0, BS=64 |
| **Transformer 偏好** | DeiT-Base | AUROC 稳定 96%+，压缩率 78.2% | image_size≤224, BS=16×6 |

### 6.2 ACT 配置建议

**何时使用 ACT**：

| 模型类型 | 推荐 | 理由 | 预期提升 |
|---------|------|------|---------|
| **大 k 模型**（≥500） | **强烈推荐** | 高冗余特征空间 | +4.95% |
| **中 k 模型**（300-500） | **推荐** | 适度冗余 | +1.98% |
| **小 k 模型**（<300） | 可选 | 性能稳定（±0.17%） | 主要用于压缩 |

**预期效果**：
- CNN 模型：平均提升 **+3.61%**（Near-OOD）
- Transformer 模型：性能稳定 **±0.17%**
- 平均压缩率：**77.4%**（61.2% - 92.6%）
- 平均推理加速：**5.01×**

### 6.3 关键配置参数

**表 2：模型特定配置（auto-configured by config.py）**

| 模型 | freeze_bn | rounds | batch_size | accumulation | Eff. BS | image_size |
|------|-----------|--------|------------|-------------|---------|------------|
| ResNet50 | 0 | 50 | 32 | 1 | 32 | 320 |
| ResNet101 | 0 | 50 | 16 | 4 | 64 | 320 |
| DenseNet121 | 0 | 50 | 32 | 1 | 32 | 320 |
| DenseNet169 | 0 | 50 | 32 | 1 | 32 | 320 |
| DenseNet201 | 0 | 50 | 32 | 1 | 32 | 320 |
| **ConvNeXt-Tiny** | **1** | 50 | 16 | 6 | 96 | 320 |
| **ConvNeXt-Base** | **1** | 50 | 8 | 8 | 64 | 320 |
| **Swin-T** | **1** | 50 | 16 | 6 | 96 | 320 |
| ViT-B/16 | 0 | 50 | 16 | 6 | 96 | 320 |
| ViT-B/32 | 0 | 50 | 16 | 6 | 96 | 320 |
| DeiT-Base | 0 | 50 | 16 | 6 | 96 | **224** |
| MobileNetV3-Large | 0 | 50 | 64 | 1 | 64 | 320 |
| EfficientNetV2-S | 0 | 50 | 32 | 1 | 32 | 320 |

**注意事项**：
1. **ConvNeXt/Swin**：`freeze_bn=1` 是 OOD 检测的关键（否则 AUROC < 0.3）
2. **DeiT-Base**：最大图像尺寸限制为 224（固定位置编码）
3. **梯度累积**：大模型（ConvNeXt、Swin、ViT、DeiT）使用累积以适应 GPU 内存

### 6.4 故障排查

**问题 1：AUROC < 0.5（分数反转）**

**症状**：OOD 样本的分数低于 ID 样本

**常见原因**：
1. **freeze_bn=0**（ConvNeXt/Swin）：BN 统计量被 OOD 样本污染
   - 解决：设置 `freeze_bn=1`
2. **Alpha 符号错误**：使用了 `residual - α * energy` 而非 `energy - α * residual`
   - 解决：检查 ViM 分数公式
3. **Energy 计算错误**：使用了 `max_logit` 而非 `logsumexp`
   - 解决：统一使用 `logsumexp(logits)`

**问题 2：AUROC 0.6 - 0.8（性能不佳）**

**常见原因**：
1. **训练不充分**：模型未收敛（test accuracy < 90%）
   - 解决：增加训练轮数或检查学习率
2. **子空间维度过大**：k 值包含过多噪声
   - 解决：使用 ACT 重新计算 k
3. **数据归一化不一致**：训练/测试使用不同的归一化
   - 解决：统一使用相同的 normalization

**问题 3：Near-OOD 性能远低于 Far-OOD**

**原因**：Near-OOD（水下生物）与 ID（浮游生物）特征相似度高

**解决**：
1. 使用更大容量的模型（ResNet101、ConvNeXt-Base）
2. 增加训练数据多样性
3. 调整 α 参数（增大权重）

---

> **文档版本**: v2.1
> **生成日期**: 2026-02-20
> **代码库**: Fed-ViM (commit: 3b86f59)
> **更新内容**：
> - **v2.1**: 改进能量估计方法（Jensen 不等式 + Energy(μ_global)）
> - v2.0: 更新为最新的 9 模型实验结果
> - v2.0: 添加 ACT 性能分析章节（架构差异、k 分类）
> - v2.0: 修正 ViM 公式为 `energy - α × residual`（原版）
> - v2.0: 添加量纲分析与实践指南
