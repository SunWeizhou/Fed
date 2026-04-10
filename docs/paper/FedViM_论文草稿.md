# FedViM：面向多中心海洋浮游生物监测的联邦分布外检测方法研究

## 摘要

海洋浮游生物图像监测通常由多个单位分别建设和维护本地数据中心。由于原始图像及其采样元信息可能关联采样海域、设备布放位置、时间节点和任务背景，跨中心直接汇聚原始数据往往受到数据治理和隐私约束。在此类多中心协作场景下，系统不仅需要提升联邦学习模型的分类精度，还需要识别未见类别、鱼卵、鱼尾、气泡和颗粒等非目标样本，即分布外（Out-of-Distribution, OOD）样本。

围绕这一需求，本文提出 `FedViM`，一种面向多中心海洋浮游生物监测的分布外检测方法。该方法能够利用各客户端上传的一阶与二阶特征充分统计量，通过联邦化重构出全局样本特征均值与协方差，进而构建出 ViM（Virtual-logit Matching）所需统计量，使得在不共享原始图像和单个样本特征的条件下完成 OOD 检测。本文进一步提出 `ACT-FedViM` 作为 `FedViM` 的扩展方法，使用 ACT（Adjusted Correlation Thresholding）机制替代原始 ViM 中的 fixed-k 经验设定，为主子空间维度 `k` 提供统计驱动的自适应选择。

本文在基于 DYB-PlanktonNet 构建的 OOD 数据划分上，对五个由联邦学习方式训练得到的 CNN 模型进行了评估。实验结果表明，`FedViM` 与 `ACT-FedViM` 的整体表现均优于 `MSP` 与 `Energy` 基线。`FedViM` 为多中心海洋浮游生物监测提供了一条可实现的联邦后处理 OOD 检测路径；`ACT-FedViM` 则在此基础上进一步降低了主子空间维度，在保持良好检测性能的同时提升了部署友好性。

**关键词**：联邦学习；分布外检测；ViM；ACT；海洋浮游生物；图像识别

## Abstract

Plankton image monitoring in the ocean is typically conducted by multiple institutions, each maintaining its own local data center. Because raw images and their associated metadata may reveal sensitive information such as sampling regions, device deployment locations, time stamps, and mission backgrounds, direct cross-center sharing of raw data is often constrained by data governance and privacy requirements. In such multi-center collaborative settings, the system must not only improve the classification accuracy of federated learning models, but also identify unseen categories and non-target samples such as fish eggs, fish tails, bubbles, and particles, namely out-of-distribution (OOD) samples.

To address this need, this paper proposes **FedViM**, an OOD detection method for multi-center marine plankton monitoring. By aggregating first- and second-order feature sufficient statistics uploaded from each client, the proposed method reconstructs the global feature mean and covariance, thereby obtaining the statistics required by **ViM** (Virtual-logit Matching) and enabling OOD detection without sharing raw images or individual sample features. Furthermore, this paper proposes **ACT-FedViM** as an extension of **FedViM**, replacing the fixed-$k$ heuristic in the original ViM with the **ACT** (Adjusted Correlation Thresholding) mechanism, so as to provide a statistically driven adaptive selection of the principal subspace dimension $k$.

Experiments are conducted on an OOD split constructed from **DYB-PlanktonNet**, using five CNN backbones trained under a federated learning setting. The results show that both **FedViM** and **ACT-FedViM** outperform the **MSP** and **Energy** baselines overall. **FedViM** provides a practical federated post-hoc OOD detection pathway for multi-center marine plankton monitoring, while **ACT-FedViM** further reduces the principal subspace dimension and improves deployment efficiency while maintaining strong detection performance.

**Keywords**: federated learning; out-of-distribution detection; ViM; ACT; marine plankton; image recognition

## 第1章 绪论

### 1.1 研究背景与意义

浮游生物是海洋生态系统中的基础组成部分，其丰度、群落结构和时空分布与营养盐循环、藻华暴发、食物网结构以及海洋生态安全密切相关。对浮游生物进行持续、准确和高效的监测，不仅关系到海洋生态过程的科学认识，也直接服务于海洋环境评估、生态预警和资源管理等实际需求[1]。随着显微成像设备、流式成像平台和深度学习识别技术的发展，浮游生物监测已经由低通量人工镜检逐步转向高通量图像分析[2-4]。在这一过程中，如何利用深度学习分类模型对海洋浮游生物进行更高效、更自动化的监测，已经逐渐成为海洋浮游生物监测领域的重要问题。

然而，面向真实业务场景的海洋浮游生物监测并不是一个理想化的集中式封闭分类任务。一方面，在本文关注的多中心业务场景中，不同海域、不同科研平台或不同业务单位往往分别建设和维护本地数据中心。不同中心在采样环境、成像设备、类别分布和标注进度上存在明显差异；与此同时，原始图像及其采样元信息还可能关联采样位置、设备布放和任务背景等敏感内容，因此跨中心直接汇聚原始数据往往会受到数据治理、隐私保护和传输成本等因素约束。对于这类多中心协作场景，如何在不直接上传原始图像数据和单个样本特征的前提下利用各中心数据，已经成为需要解决的重要问题。联邦学习通过“数据保留在本地、模型更新在中心间聚合”的方式，为此类多中心协作场景提供了可行方案[5-7]。

另一方面，海洋浮游生物监测具有显著的开放世界特征。即使分类模型已经学习了训练集中定义的已知类别（in-distribution，ID），但真实场景下仍不可避免地出现未见浮游生物类别、鱼卵、鱼尾、气泡和颗粒杂波等样本。如果系统对这些样本给出高置信度误判，将直接影响监测结果的可靠性。因此，在真实部署条件下，仅仅提高 ID 类别上的分类准确率并不足以保证系统可用性，模型还需要具备对未知类别的拒识能力。因此，多中心海洋浮游生物监测所面临的，不只是分类精度问题，更是在原始图像数据不能汇聚的约束下进行分布外样本检测的问题[8-10]。

基于上述背景，本文关注的核心问题是：**在不共享原始图像的多中心海洋浮游生物监测场景下，如何为联邦化的图像识别任务构建可实现、可复用的 OOD 检测方法。**对于这一问题，后处理式 OOD 检测能展现出良好的适用性。对于已经完成联邦训练的分类模型，后处理方法能够在不重新设计训练框架、不引入额外生成模型的前提下提供 OOD 检测能力[12-14]。这种即插即用与轻量化的特性，使其能够更容易地部署在现有的多中心海洋浮游生物监测系统中。围绕这一思路，研究联邦场景下的后处理 OOD 检测方法，不仅具有明确的理论意义，也具有较强的实际应用价值。

### 1.2 国内外研究现状

海洋浮游生物图像识别已经从低通量人工镜检逐步发展到自动分类、高通量分析和现场部署[2-4]。相关研究一方面关注显微成像或流式成像条件下的自动识别流程，另一方面也面向边缘部署、实时监测和海上自适应采样等应用方向展开探索[3][4]。这表明，海洋浮游生物图像识别正在从实验室条件下的离线分类任务逐步走向真实业务场景。然而，与封闭集图像分类不同，真实海洋环境中的样本分布会受到海域差异、成像条件、季节变化和设备差异等多种因素影响，系统面临的并不只是已知类别之间的分类问题，还需要能够识别出开放世界中的未知物种。已有研究已经从数据分布偏移（dataset shift）[8]、开放集识别（open-set recognition）[9] 和分布外（out-of-distribution, OOD）检测[10] 等角度讨论了海洋浮游生物监测系统在真实部署中的失效风险。这说明，海洋浮游生物识别在实际应用中不仅是分类问题，也是开放世界的识别问题。

在多中心海洋监测场景下，这一问题又进一步受到数据协作方式的约束，而无法采用集中式训练。现有工作普遍将联邦学习视为多中心数据协作的重要框架。FedAvg[5] 建立了最经典的参数平均范式，其基本思想是各客户端保留原始数据，仅上传模型参数在服务器进行聚合。此后，大量研究围绕其通信效率、非独立同分布数据（non-IID）和隐私保护等问题展开，逐步扩展了联邦学习的理论与应用边界[6][7]。现有研究大体可以分为两类：一类主要关注标准联邦分类训练的准确率、收敛性和系统效率[5][6]；另一类则更强调异构场景下的隐私保护、个性化适配和应用部署[7]。随着 OOD 问题在真实部署中的重要性不断提高，已有工作也开始关注联邦场景下的 OOD 检测[11]。但现有联邦 OOD 方法往往与训练阶段的额外模块或生成式结构高度耦合，带来额外计算开销和实现复杂度。相比之下，对于已经完成联邦训练的分类模型，后处理式（post-hoc）OOD 检测无需重新设计训练框架，且兼具轻量化与“即插即用”的适配性，更适合作为多中心联邦识别系统的扩展。

后处理式 OOD 检测方法以已训练完成的分类模型为基础，不需要重新训练完整系统，部署成本较低，也更容易与现有任务兼容。现有方法大体可以分为两类：一类主要利用输出空间信息，另一类进一步利用特征空间结构。输出空间方法中，`MSP`[12] 通过最大 softmax 概率刻画模型置信度，是最经典的 OOD 基线之一；`Energy`[13] 则利用 logits 的 log-sum-exp 构造能量分数，在多个视觉任务上表现出较强竞争力。与这类方法相比，特征空间方法进一步利用中间特征与分类结果之间的几何关系。ViM（Virtual-logit Matching）[14] 通过主子空间与残差空间分解，结合特征空间结构与 logits 来刻画样本偏离 ID 分布的程度，在多个OOD检测任务上表现突出。在海洋浮游生物 OOD benchmark 上，Han 等[10] 对 `22` 种方法进行了统一比较，结果显示 ViM 在该基准上整体表现突出，并在 Far-OOD 场景中优势较为明显。从方法结构上来看，ViM 依赖全局特征均值与协方差，使其能够适配统计量可聚合的联邦学习场景。

尽管 ViM 为后处理式 OOD 检测提供了较有竞争力的技术路线，但在其实际应用中仍存在一个关键问题，即主子空间维度 `k` 的设定通常依赖经验选择[14]。固定维度方案（fixed-k）虽然实现简单，但不同模型的网络结构和特征维度差异明显，经验式 fixed-k 设定往往缺乏统一的统计依据，并可能在不同模型之间带来适配性差异。此外，当 `k` 取值过大时，部署阶段需要保存的投影矩阵规模和计算 OOD 分数的开销都会增加，不利于后处理模块的轻量化落地。在高维统计场景中，当特征维度与样本量处于相近量级时，样本协方差的特征值容易受到噪声膨胀影响，因此如何从样本协方差或相关矩阵中估计有效维度，已成为一条独立的重要研究线索。ACT（Adjusted Correlation Thresholding）[15] 属于这一方向的代表方法，它通过相关矩阵谱修正与阈值判别估计因子数量，从而为高维特征空间中的有效维度选择提供数据驱动依据。就 ViM 而言，这类方法为主子空间维度选择提供了可借鉴的统计工具。

综上，已有研究分别从浮游生物开放环境识别、多中心数据协作、后处理式 OOD 检测以及高维统计选维等方面提供了重要基础：真实海洋监测提出了面向 OOD 样本的检测需求，多中心协作使这一需求需要在联邦框架下讨论，ViM 为已完成训练的分类模型提供了 OOD 检测路径，而高维统计中的有效维度选择方法则为 ViM 的主子空间构造提供了统计学支撑。然而，现有研究仍缺少一种能够面向多中心海洋浮游生物监测的分布外检测方法，使其在不共享原始图像的条件下复用联邦分类模型，并以后处理方式同时实现 OOD 检测与主子空间维度自适应选择。

### 1.3 本文主要工作

针对上述研究缺口，本文围绕联邦后处理 OOD 检测展开研究。在不共享原始图像和单个样本特征的前提下，将 ViM 所需统计量的计算改写为联邦充分统计量聚合过程，构造 `FedViM`；同时，针对其主子空间维度选择问题，引入 ACT 构造 `ACT-FedViM`，以实现具有统计学依据的自适应选择。具体而言，本文主要完成了以下工作：

1. 面向多中心海洋浮游生物监测场景，将 ViM 引入联邦学习框架，提出了 `FedViM`。各客户端仅上传一阶与二阶特征充分统计量，服务器据此完成全局均值与协方差重构，在不共享原始图像和单个样本特征的条件下获得 ViM 所需的全局统计量。
2. 针对 `FedViM` 主子空间维度的选择问题，提出了 `ACT-FedViM` 方法，用统计驱动的方式代替固定的维度选择，在保持 OOD 检测性能的同时进一步压缩主子空间维度，增强对不同分类模型的适应性。
3. 在五个 CNN 模型上对 `MSP`、`Energy`、`FedViM`、`ACT-FedViM` 以及对应的 pooled ViM 参考方法进行了系统评估。实验表明，`ACT-FedViM` 在保持竞争性 OOD 检测性能的同时，将主子空间维度平均压缩 `86.8%`；`FedViM` 与 `Pooled-ViM` 的结果也保持了基本一致。

全文结构如下：第 2 章给出 `FedViM` 与 `ACT-FedViM` 的方法设计；第 3 章介绍数据集、联邦设置、实现细节与评估指标；第 4 章展示五个 CNN 模型上的实验结果，并对 `FedViM` 与 `ACT-FedViM` 的性能进行分析；第 5 章给出全文结论，并讨论当前工作的局限与后续研究方向。

---

## 第2章 方法

### 2.1 问题设定与整体框架

设共有 $N$ 个客户端，每个客户端对应一个本地数据中心或数据持有单位。客户端 $i$ 持有本地有标签的 ID 训练集

$$
\mathcal{D}_i=\{(x_j^{(i)},y_j^{(i)})\}_{j=1}^{n_i},
$$

其中，$x_j^{(i)}$ 表示浮游生物图像，$y_j^{(i)}$ 表示对应的 ID 类别标签，$n_i=|\mathcal{D}_i|$ 为客户端 $i$ 的样本数。各客户端之间不共享原始图像。我们关注的任务是：在联邦学习得到的分类模型基础上，通过后处理方式构建 OOD 检测器，使系统能够识别测试样本是否为 OOD 样本。

![图 2-1 FedViM 与 ACT-FedViM 的联邦后处理流程图](../../paper_tools/figures/FedViM流程图.png)

图 2-1 FedViM 与 ACT-FedViM 的联邦后处理流程图

算法整体框架如图 2-1 所示，可概括为以下三个环节：

1. **联邦统计量重构**：客户端在本地 ID 训练集上计算一阶与二阶特征统计量，服务器据此重构全局特征均值与协方差；
2. **主子空间构造**：在上述全局统计量基础上，`FedViM` 采用固定维度 $k$ 构造主子空间，`ACT-FedViM` 则利用 ACT[15] 自适应确定 $k$。在主子空间确定后，各客户端进一步计算本地能量项与残差项的标量统计量，服务器聚合得到全局经验校准系数 $\alpha$。
3. **ViM 分数计算**：在得到主子空间、校准系数 $\alpha$ 与特征均值 $\mu$ 后，给定测试样本 $x$，计算其残差和 logit，并利用校准系数 $\alpha$ 加权求和得到最终的 OOD 评分。

### 2.2 FedViM 与 ACT-FedViM 的构造

ViM[14] 依赖全局特征均值与协方差来描述 ID 分布的特征结构。为在不访问原始图像和逐样本特征的条件下获得这些统计量，客户端 $i$ 在本地 ID 训练集上计算如下充分统计量：

$$
\begin{aligned}
s_i^{(1)} &= \sum_{x\in\mathcal{D}_i} f_\theta(x),\\
s_i^{(2)} &= \sum_{x\in\mathcal{D}_i} f_\theta(x)f_\theta(x)^\top,\\
n_i &= |\mathcal{D}_i|.
\end{aligned}
$$

服务器对各客户端上传结果进行聚合：

$$
\begin{aligned}
S^{(1)} &= \sum_{i=1}^{N}s_i^{(1)},\\
S^{(2)} &= \sum_{i=1}^{N}s_i^{(2)},\\
n &= \sum_{i=1}^{N}n_i.
\end{aligned}
$$

由此可得全局特征均值与协方差：

$$
\mu_{\text{global}}=\frac{S^{(1)}}{n},
\qquad
\Sigma_{\text{global}}=\frac{S^{(2)}}{n}-\mu_{\text{global}}\mu_{\text{global}}^\top.
$$

上述结果表明，ViM 所需的全局统计量可以通过联邦聚合充分统计量得到，而不需要服务器访问原始图像或样本级特征。基于此，可在联邦场景下复现 ViM 的特征几何结构。

在此基础上，我们首先构造 `FedViM`。该方法采用固定维度 $k$ 构造 ViM 主子空间。考虑到不同模型的特征维度存在差异，实验中对较低维模型取 $k=512$，对较高维模型取 $k=1000$。在得到协方差矩阵 $\Sigma_{\text{global}}$ 后，对其进行特征分解，并取前 $k$ 个协方差主方向构造主子空间矩阵

$$
P\in\mathbb{R}^{D\times k},
$$

其中 $D$ 为特征维度。由此得到固定维度版本的联邦 ViM，即 `FedViM`。

进一步地，本文在相同联邦统计量基础上引入 ACT，以构造 `ACT-FedViM`，为主子空间维度 $k$ 提供数据驱动的选择依据。具体而言，先将全局协方差矩阵转换为相关矩阵：

$$
R=D_\Sigma^{-1/2}\Sigma_{\text{global}}D_\Sigma^{-1/2},
$$

其中，$D_\Sigma$ 为由 $\Sigma_{\text{global}}$ 对角元素构成的对角矩阵。对 $R$ 进行特征分解，设其降序特征值为

$$
\lambda_1\ge \lambda_2\ge \cdots \ge \lambda_p,
$$

其中 $p$ 为特征维度。根据 ACT，定义阈值

$$
s=1+\sqrt{\frac{p}{n-1}},
$$

并通过离散 Stieltjes 变换对样本特征值进行偏差修正，得到修正特征值 $\lambda_j^C$。最终采用如下规则确定自适应维度：

$$
k_{\text{ACT}}=\max\{j:\lambda_j^C>s\}.
$$

在得到 $k_{\text{ACT}}$ 后，仍对 $\Sigma_{\text{global}}$ 进行 PCA，并取前 $k_{\text{ACT}}$ 个协方差主方向构造主子空间矩阵

$$
P\in\mathbb{R}^{D\times k_{\text{ACT}}}.
$$

因此，`ACT-FedViM` 与 `FedViM` 在主子空间的构造基础上保持一致，区别仅在于前者利用 ACT 自适应确定维度 $k$，而后者采用固定经验设定。

### 2.3 联邦经验校准与 OOD 评分

给定测试样本 $x$，记其特征为 $z=f_\theta(x)$，分类头输出 logits 为 $g_\theta(x)\in\mathbb{R}^{C}$，其中 $C$ 为 ID 类别数。ViM 的残差项定义为

$$
\text{Residual}(x)=\left\|(I-PP^\top)(z-\mu_{\text{global}})\right\|_2.
$$

能量项定义为

$$
\text{Energy}(x)=\log\sum_{c=1}^{C}\exp(g_\theta(x)_c).
$$

为平衡能量项与残差项的量纲，本文采用经验方式估计校准系数 $\alpha$：

$$
\alpha=\frac{\mathbb{E}_{ID}[\text{Energy}(x)]}{\mathbb{E}_{ID}[\text{Residual}(x)]},
$$

在联邦场景下，上式中的期望通过客户端本地统计量聚合获得。在主子空间 $P$ 与全局均值 $\mu_{\text{global}}$ 固定后，客户端 $i$ 在本地 ID 训练集上计算

$$
S_i^{E}=\sum_{x\in\mathcal{D}_i}\text{Energy}(x),\qquad
S_i^{R}=\sum_{x\in\mathcal{D}_i}\text{Residual}(x),\qquad
n_i=|\mathcal{D}_i|.
$$

服务器聚合得到

$$
S^{E}=\sum_{i=1}^{N}S_i^{E},\qquad
S^{R}=\sum_{i=1}^{N}S_i^{R},\qquad
n=\sum_{i=1}^{N}n_i.
$$

于是，

$$
\mathbb{E}_{ID}[\text{Energy}(x)]\approx \frac{S^{E}}{n},
\qquad
\mathbb{E}_{ID}[\text{Residual}(x)]\approx \frac{S^{R}}{n},
$$

从而得到联邦经验校准系数

$$
\alpha=\frac{S^{E}/n}{S^{R}/n}.
$$

最终，定义样本 $x$ 的联合评分为

$$
\text{Score}(x)=\text{Energy}(x)-\alpha\cdot \text{Residual}(x).
$$

当 $\text{Score}(x)$ 较小时，样本更可能偏离 ID 分布，因此更可能是 OOD 样本。由此，`FedViM` 与 `ACT-FedViM` 的评分框架保持一致，二者的差异仅在于主子空间矩阵 $P$ 的构造方式不同。

在实际部署中，可在验证集上根据目标假阳性率或目标召回率选取判别阈值 $\tau$。当 $\text{Score}(x)<\tau$ 时，将样本判定为 OOD；否则判定为 ID。

此外，部署阶段需要保存主子空间矩阵 $P\in\mathbb{R}^{D\times k}$，而 ViM 打分中涉及的投影与残差计算复杂度近似为 $O(Dk)$。因此，更小的 $k$ 将直接减少后处理模块的存储开销与计算开销，这也是 `ACT-FedViM` 相较于 fixed-k `FedViM` 的一个重要部署优势。

---

## 第3章 实验设计

### 3.1 数据集、OOD 划分与联邦实验设定

本文实验基于 Li 等发布于 IEEE Dataport 的 `DYB-PlanktonNet` 数据集[16] 和 Han 等[10] 的工作构建了 OOD 检测任务。结合本文研究目标，实验数据被组织为以下四个部分：

- `D_ID_train`：`54` 个 ID 类别，共 `26,034` 张图像；
- `D_ID_test`：`54` 个 ID 类别，共 `2,939` 张图像；
- `D_Near_test`：`26` 个 Near-OOD 类别，共 `1,516` 张图像；
- `D_Far_test`：`12` 个 Far-OOD 类别，共 `17,031` 张图像。

其中，`D_ID_train` 用于联邦分类训练与后处理统计量估计，`D_ID_test` 用于评估联邦模型在已知类别上的分类性能，`D_Near_test` 与 `D_Far_test` 分别用于评估模型在相近未知类和明显非目标类上的 OOD 检测能力。Near-OOD 主要由与 ID 浮游生物在形态结构或生态属性上较为接近、但不属于训练目标的类别构成；Far-OOD 则包含鱼卵、鱼尾、气泡以及多类颗粒杂波，更接近真实海洋监测中可能出现的非目标样本。上述划分同时覆盖了“相近未知类”和“明显非目标类”两种更贴近实际部署需求的 OOD 场景。

在训练阶段，本文从 `D_ID_train` 中固定划出 `10%` 样本作为服务端验证集，用于 early stopping 和最佳 checkpoint 选择；`D_ID_test`、`D_Near_test` 与 `D_Far_test` 仅用于最终评估，不参与模型训练与模型选择。

为模拟多中心数据协作场景，本文采用 Dirichlet 分布将 `D_ID_train` 划分到 `5` 个客户端，参数取 `alpha = 0.1`，以构造高度异构的非独立同分布数据分片。全局训练采用 FedAvg 聚合，客户端参与比例设为 `1.0`，即每一轮通信均使用全部客户端参与参数更新。正式训练统一设置为 `50` 个通信轮次，每轮执行 `4` 个本地 epoch。上述设定旨在在较强数据异构条件下考察联邦分类模型的训练效果，以及后续 `FedViM` 与 `ACT-FedViM` 的联邦后处理可行性。

### 3.2 对比方法、模型范围与实现细节

为考察所提方法在不同网络结构下的适用性，本文选取五个具有代表性的 CNN 模型作为实验对象，包括 `DenseNet169`、`ResNet50`、`ResNet101`、`EfficientNetV2-S` 和 `MobileNetV3-Large`。其中，`DenseNet169`、`ResNet50` 和 `ResNet101` 代表较为经典的卷积神经网络结构，`EfficientNetV2-S` 与 `MobileNetV3-Large` 则代表兼顾效率与性能的现代轻量化 CNN 结构。通过在不同模型上进行统一比较，可以更全面地评估联邦后处理 OOD 方法的稳定性与适配性。

在方法对照方面，本文共使用六种后处理方法。第一类为输出空间基线 `MSP` 与 `Energy`，它们直接基于冻结后的 `best_model` 在 `D_ID_test`、`D_Near_test` 与 `D_Far_test` 上计算分数，用于提供经典后处理参照。第二类为本文主方法 `FedViM` 与 `ACT-FedViM`，分别对应联邦场景下的 fixed-k ViM 与 ACT 自适应选维 ViM。第三类为参考方法 `Pooled-ViM` 与 `Pooled-ACT-ViM`，它们在相同 `best_model` 上将五个客户端的 ID 训练分片集中汇聚后计算 ViM 所需统计量，用于验证联邦统计量聚合是否保持原始 ViM 的判别行为。前一类比较主要用于说明方法优越性，后一类比较主要用于说明联邦化 ViM 的合理性。

在训练实现上，五个 CNN 模型采用统一的联邦训练框架。主要配置如下：

- 全局随机种子设置为 `42`；
- 通信轮次为 `50`，每轮本地训练 `4` 个 epoch；
- 基础学习率设为 `0.001`；
- 优化器采用带动量的 SGD，动量为 `0.9`；
- 权重衰减设为 `1e-4`；
- 学习率调度采用 `5` 轮 warmup 与 cosine decay 相结合的方式。

后处理评估统一遵循如下流程：首先，利用联邦训练得到五个 backbone，并以服务端验证集准确率选择 `best_model`；随后固定 `best_model` 参数，不再进行任何梯度更新。对于 `FedViM` 与 `ACT-FedViM`，各客户端在本地 ID 训练分片上提取特征并分别计算 `sum_z`、`sum_zzT` 与样本数，服务器聚合得到全局特征均值、协方差以及主子空间 `P`；在 `P` 与全局均值固定后，各客户端进一步计算本地能量项与残差项的标量统计量，服务器聚合得到经验校准系数 `\alpha`，再据此执行 OOD 检测。对于 `Pooled-ViM` 与 `Pooled-ACT-ViM`，则将相同的五个客户端 ID 训练分片集中汇聚后，直接在整体训练集上计算 ViM 所需全部统计量。这样的设计使得联邦方法与 pooled 参考方法在样本范围、分类模型与评分公式上保持一致，区别仅在于统计量的获得方式。

### 3.3 评估指标与分析维度

为全面评估联邦后处理 OOD 检测方法的性能，本文从分类能力、近域拒识能力、远域拒识能力以及后处理轻量化收益四个维度进行分析，并采用如下指标进行衡量。

首先，使用 ID 分类准确率（Accuracy）衡量联邦分类模型在 `D_ID_test` 上的基础识别能力。该指标反映分类模型本身对已知类别的判别效果，是后续 OOD 检测的基础。

其次，使用 Near-OOD AUROC 衡量模型对 `D_Near_test` 中相近未知类别的区分能力。由于 Near-OOD 样本在形态结构上通常与 ID 类别较为接近，因此该指标能够更直接反映方法对细粒度未知类的识别难度。

然后，使用 Far-OOD AUROC 衡量模型对 `D_Far_test` 中明显非目标样本的区分能力。相较于 Near-OOD，Far-OOD 通常与 ID 类别差异更大，因此该指标主要用于评估方法在更宽松 OOD 场景下的整体拒识能力。

最后，为刻画 `ACT-FedViM` 相对于 fixed-k `FedViM` 在后处理阶段的压缩效果，本文定义主子空间压缩率为

$$
\text{Compression} = 1 - \frac{k_{\text{ACT}}}{k_{\text{fixed}}}.
$$

其中，$k_{\text{fixed}}$ 表示 `FedViM` 中采用的固定主子空间维度，$k_{\text{ACT}}$ 表示 `ACT-FedViM` 中通过 ACT 自适应确定的主子空间维度。该指标用于衡量自适应选维在降低后处理存储与计算开销方面的潜在收益。

综合而言，Accuracy 用于衡量联邦分类模型的已知类识别能力，Near-OOD AUROC 与 Far-OOD AUROC 分别对应相近未知类与明显非目标样本的拒识能力，而 Compression 则用于刻画 `ACT-FedViM` 在部署效率方面的附加价值。四类指标共同构成了本文实验结果分析的基本依据。

---

## 第4章 实验结果与分析

### 4.1 总体结果与主要发现

表 4-1 给出了五个 CNN 模型上 `FedViM` 与 `ACT-FedViM` 的主要结果。

**表 4-1 五个 CNN 模型上的主要结果**

| 模型              | ID Acc (%) | FedViM k  | ACT k     | 压缩率    | FedViM Near (%) | ACT Near (%) | FedViM Far (%) | ACT Far (%) |
| ----------------- | ---------- | --------- | --------- | --------- | --------------- | ------------ | -------------- | ----------- |
| DenseNet169       | 96.50      | 1000      | 75        | 92.5%     | 82.05           | 96.73        | 86.00          | 97.24       |
| EfficientNetV2-S  | 97.04      | 512       | 63        | 87.7%     | 96.40           | 96.45        | 97.52          | 97.32       |
| MobileNetV3-Large | 95.13      | 512       | 93        | 81.8%     | 95.70           | 96.11        | 97.04          | 97.39       |
| ResNet101         | 96.22      | 1000      | 140       | 86.0%     | 96.50           | 96.74        | 97.43          | 97.41       |
| ResNet50          | 96.53      | 1000      | 138       | 86.2%     | 95.68           | 95.69        | 97.19          | 97.24       |
| **平均**          | **96.28**  | **804.8** | **101.8** | **86.8%** | **93.27**       | **96.34**    | **95.03**      | **97.32**   |



![图 4-1 主子空间压缩效果](../../paper_tools/figures/figure_2_subspace_compression.png)

*图 4-1 五个 CNN 模型上 fixed-k `FedViM` 与 `ACT-FedViM` 的主子空间维度比较。*

从表 4-1 和图 4-1 可以发现，首先，`ACT-FedViM` 在五款模型上均显著降低了主子空间维度。相较于 fixed-k `FedViM`，ACT 选择的 `k` 落在 `63` 至 `140` 的范围内，平均维度由 `804.8` 降至 `101.8`，平均压缩率达到 `86.8%`。这表明，ACT 在当前实验范围内稳定地缩小了 ViM 后处理所需的主子空间规模。

其次，从平均结果看，`ACT-FedViM` 的 Near-OOD AUROC 和 Far-OOD AUROC 分别由 `93.27%` 提升至 `96.34%`、由 `95.03%` 提升至 `97.32%`。但这一平均增益并不完全均匀：`MobileNetV3-Large`、`ResNet101` 和 `ResNet50` 上表现为小幅提升或基本持平，`EfficientNetV2-S` 在 Far-OOD 上出现轻微波动，而平均增益的主要来源是 `DenseNet169` 上对 fixed-k 失配的显著纠偏。

除检测性能外，主子空间维度的降低还具有明确的部署含义。ViM 后处理需要保存主子空间矩阵 $P \in \mathbb{R}^{D \times k}$，并在打分阶段完成投影与残差计算，因此更小的 $k$ 直接对应更低的存储开销与计算开销。因此，`ACT-FedViM` 的稳定收益主要体现在统计驱动的选维方式与一致的子空间压缩，而其 AUROC 增益则体现出一定的模型依赖性。

### 4.2 与 MSP、Energy 基线的整体比较

为了说明联邦 ViM 系方法的整体有效性，表 4-2 给出了四类主方法在五个模型上的平均表现。这里不将 pooled 参考方法并入主基线比较，而在下一节单独讨论联邦统计量聚合与 pooled 统计量之间的一致性。

**表 4-2 不同方法的平均 OOD 检测表现**

| 方法       | 平均 Near-OOD AUROC (%) | 平均 Far-OOD AUROC (%) |
| ---------- | ----------------------- | ---------------------- |
| MSP        | 90.42                   | 87.27                  |
| Energy     | 83.34                   | 80.89                  |
| FedViM     | 93.27                   | 95.03                  |
| ACT-FedViM | 96.34                   | 97.32                  |

![图 4-2 五模型方法比较](../../paper_tools/figures/figure_1_method_comparison.png)

*图 4-2 五个 CNN 模型上 `FedViM`、`ACT-FedViM`、`MSP` 与 `Energy` 的 Near/Far-OOD AUROC 对比。*

从表 4-2 和图 4-2 可以看出，基于特征空间结构的 ViM 系方法整体上优于纯输出空间基线。以平均结果计，`FedViM` 相比 `MSP` 的 Near-OOD 与 Far-OOD AUROC 分别提高 `2.85` 和 `7.76` 个百分点，相比 `Energy` 分别提高 `9.93` 和 `14.14` 个百分点；`ACT-FedViM` 相比 `MSP` 的 Near-OOD 与 Far-OOD AUROC 分别提高 `5.92` 和 `10.05` 个百分点，相比 `Energy` 分别提高 `13.00` 和 `16.43` 个百分点。

值得注意的是，在当前细粒度浮游生物任务中，`Energy` 的平均表现甚至低于 `MSP`。这一现象表明，单纯依赖输出空间置信度或能量信息的方法在该任务上存在明显局限；相比之下，利用特征空间结构信息的 ViM 系方法更适合当前 OOD 检测场景。由此可见，联邦化 ViM 本身已经构成一条有效的后处理路线。

### 4.3 与 pooled ViM 的一致性验证

为验证联邦统计量聚合没有改变 ViM 的基本判别行为，本文进一步将 `FedViM` 与 `Pooled-ViM`、`ACT-FedViM` 与 `Pooled-ACT-ViM` 进行对照。四种方法均基于同一 `best_model` 和相同的五个客户端 ID 训练分片；区别仅在于统计量是按客户端分片计算后聚合，还是先集中汇聚训练分片再统一计算。

**表 4-3 联邦统计聚合与 pooled 统计的一致性验证**

| 比较组 | 联邦 Near-OOD AUROC (%) | Pooled Near-OOD AUROC (%) | 平均绝对差 Near (%) | 联邦 Far-OOD AUROC (%) | Pooled Far-OOD AUROC (%) | 平均绝对差 Far (%) |
| ------ | ----------------------- | ------------------------- | ------------------- | ---------------------- | ------------------------ | ------------------ |
| FedViM vs Pooled-ViM | 93.27 | 93.24 | 0.02 | 95.03 | 94.85 | 0.19 |
| ACT-FedViM vs Pooled-ACT-ViM | 96.34 | 96.32 | 0.03 | 97.32 | 97.30 | 0.02 |

从表 4-3 可以看出，联邦统计聚合与 pooled 统计得到的 ViM 结果整体上保持了高度一致。对 `ResNet101`、`EfficientNetV2-S` 和 `ResNet50` 而言，联邦化 `ViM` 与 pooled `ViM` 在两位小数上基本一致；`MobileNetV3-Large` 只出现极小波动。`DenseNet169` 上 fixed-k `ViM` 的 Far-OOD 结果存在略大的差值，但其 `ACT-FedViM` 与 `Pooled-ACT-ViM` 仍然保持接近，且二者的自适应维度仅相差 `1`。这些结果说明，将 ViM 所需统计量改写为联邦充分统计量聚合后，并不会实质改变 ViM 的判别特性。

### 4.4 代表模型分析

为覆盖不同模型上 ACT 收益的代表性结果形态，本文选取 `MobileNetV3-Large`、`ResNet101` 和 `DenseNet169` 三个模型进行进一步分析。三者分别对应以下三种典型情形：`MobileNetV3-Large` 在大幅压缩主子空间的同时取得小幅性能提升；`ResNet101` 在显著压缩维度的同时保持与 fixed-k 方法接近的检测能力；`DenseNet169` 则表现出 fixed-k 与自适应选维之间较大的结果差异。

![图 4-3 正文代表模型对比](../../paper_tools/figures/figure_3_selected_models.png)

*图 4-3 `MobileNetV3-Large`、`ResNet101` 与 `DenseNet169` 三个代表模型上的 Near/Far-OOD AUROC 对比。*

图 4-3 展示了三类代表模型的差异模式。`MobileNetV3-Large` 对应“高性能前提下的大幅压缩”，`ResNet101` 对应“压缩与性能的相对平衡”，而 `DenseNet169` 则体现了 fixed-k 与自适应选维结果之间的显著差异。

`MobileNetV3-Large` 的 fixed-k 为 `512`，而 ACT 选择的主子空间维度为 `93`，压缩率达到 `81.8%`。在此基础上，Near-OOD AUROC 由 `95.70%` 提升至 `96.11%`，Far-OOD AUROC 由 `97.04%` 提升至 `97.39%`。这一结果表明，在较轻量的 CNN 模型上（`MobileNetV3-Large` 的参数量仅为 `3.02M`，`ResNet101` 的参数量为 `42.61M`，`DenseNet169` 的参数量为 `12.57M`），ACT 可以在显著压缩主子空间的同时维持甚至小幅提升检测性能。

`ResNet101` 的 fixed-k 为 `1000`，ACT 选择 `140` 维主子空间，压缩率为 `86.0%`。在更小主子空间下，Near-OOD AUROC 由 `96.50%` 提升至 `96.74%`，Far-OOD AUROC 由 `97.43%` 变化至 `97.41%`。这一结果说明，在高维 CNN 模型上，`ACT-FedViM` 可以在显著缩小子空间规模的同时保持与 fixed-k 方法几乎一致的检测能力。

`DenseNet169` 的 fixed-k 为 `1000`，而 ACT 仅保留 `75` 维主子空间，压缩率达到 `92.5%`。在此设定下，Near-OOD AUROC 由 `82.05%` 提升至 `96.73%`，Far-OOD AUROC 由 `86.00%` 提升至 `97.24%`。与其他模型相比，该模型上 fixed-k 与自适应选维之间的差异最为显著。

从当前结果分析，`DenseNet169` 的网络架构对主子空间维度的选择较为敏感。fixed-k 方法保留了较高的主子空间维度，使其残差空间被大幅压缩，从而导致 OOD 检测性能下降。综合三类代表模型可以看出，ACT 的价值主要体现在两方面：一是为 ViM 提供统计驱动的、更紧凑的选维方式；二是在特定模型架构上缓解 fixed-k 可能带来的失配风险。

## 第5章 结论与展望

本文围绕多中心海洋浮游生物监测场景下“原始图像难以跨中心共享，而监测系统又需要具备 OOD 拒识能力”的问题，研究了联邦后处理 OOD 检测方法。针对这一问题，本文提出了 `FedViM`，并在此基础上进一步构造了其自适应选维扩展 `ACT-FedViM`。其中，`FedViM` 通过联邦聚合一阶与二阶充分统计量，实现了 ViM 所需全局特征均值与协方差的重构；`ACT-FedViM` 则在利用 ACT 自动选择主子空间维度 `k`，从而形成一套统计驱动的联邦 ViM 后处理方法。

在基于 `DYB-PlanktonNet` 构建的 OOD 划分上，本文对 `5` 个 CNN 模型进行了实验评估。结果表明，联邦化 ViM 路线在该任务上是可行的：五模型平均 ID 分类准确率达到 `96.28%`，说明联邦分类模型具备较好的已知类识别能力；与 `MSP` 和 `Energy` 两类输出空间基线相比，`FedViM` 与 `ACT-FedViM` 在 Near-OOD 和 Far-OOD 检测上整体表现更优，说明利用特征空间结构信息的后处理方法更适合当前细粒度浮游生物 OOD 场景。此外，`FedViM` 与 `Pooled-ViM`、`ACT-FedViM` 与 `Pooled-ACT-ViM` 的结果基本一致，说明联邦充分统计量聚合没有改变 ViM 的核心判别特性。

综合全文的理论分析与实验结果，本文得到以下几点主要结论：

1. ViM 所需的全局特征统计量可以通过联邦充分统计量聚合完成重构，从而形成可实现的联邦后处理 OOD 检测流程；与对应的 pooled ViM 结果相比，联邦化实现保持了基本一致的判别表现。
2. ACT 可以作为联邦 ViM 的自适应选维模块接入已经训练好的分类模型的后处理评估，为主子空间维度选择提供数据驱动的统计依据。
3. `ACT-FedViM` 的主要价值并不完全体现在 AUROC 提升上，而更多体现在统计驱动的选维方式、平均 `86.8%` 的主子空间压缩以及部署友好性的提升等方面。


总体而言，`FedViM` 给出了联邦场景下实现 ViM 后处理 OOD 检测的可行方案，`ACT-FedViM` 则在此基础上进一步提高了后处理模块的紧凑性与对不同模型的适应性。

尽管本文在联邦后处理 OOD 检测方面进行了初步探索，但当前工作仍存在一定局限。

（1）本文对 ViM 的改进主要体现在主子空间维度 `k` 的选择上，没有改变主子空间与残差空间的构造方式。因此，当 OOD 判别信息没有落在样本协方差矩阵的残差空间时，自适应选维所能带来的性能改善仍然是有限的。未来可在保持联邦统计量可聚合的前提下，进一步研究 OOD 判别信息的结构，设计出能够涵盖更多 OOD 判别信息的选维机制。

（2）本文的方法虽然避免了原始图像和逐样本特征的跨中心传输，但在传输统计量时并没有引入隐私机制来给出数学上的隐私保证。因此，后续研究可在统计量聚合流程中引入安全聚合、差分隐私扰动或其他隐私保护机制，进一步提升多中心海洋浮游生物监测系统的隐私保护能力。

---

## 参考文献

[1] Naselli-Flores L, Padisak J. Ecosystem services provided by marine and freshwater phytoplankton[J]. Hydrobiologia, 2023, 850(12-13): 2691-2706.
[2] Eerola T, Kareinen J, Pitois S G, et al. Survey of automatic plankton image recognition: challenges, existing solutions and future perspectives[J]. Artificial Intelligence Review, 2024, 57(5): 114.
[3] Pitois S G, Schmid M S, Eerola T, et al. RAPID: real-time automated plankton identification dashboard using Edge AI at sea[J]. Frontiers in Marine Science, 2025, 11: 1513463.
[4] Schmid M S, Eerola T, Pitois S G, et al. Edge computing at sea: high-throughput classification of in-situ plankton imagery for adaptive sampling[J]. Frontiers in Marine Science, 2023, 10: 1187771.
[5] McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Singh A, Zhu J, eds. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics. Fort Lauderdale, FL, USA: PMLR, 2017: 1273-1282.
[6] Kairouz P, McMahan H B, Avent B, et al. Advances and open problems in federated learning[J]. Foundations and Trends in Machine Learning, 2021, 14(1-2): 1-210.
[7] Yang Q, Liu Y, Chen T, et al. Federated machine learning: Concept and applications[J]. ACM Transactions on Intelligent Systems and Technology, 2019, 10(2): 12:1-12:19.
[8] Chen C, Kyathanahally S P, Reyes M, et al. Producing plankton classifiers that are robust to dataset shift[J]. Limnology and Oceanography: Methods, 2025, 23: 39-66.
[9] Kareinen J, Skyttä A, Eerola T, et al. Open-set plankton recognition[C]//Del Bue A, Leal-Taixe L, eds. Computer Vision - ECCV 2024 Workshops. Cham: Springer, 2025: 168-184.
[10] Han Y, He J, Xie C, et al. Benchmarking out-of-distribution detection for plankton recognition: a systematic evaluation of advanced methods in marine ecological monitoring[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops. Piscataway, NJ, USA: IEEE, 2025: 2142-2152.
[11] Liao X, Liu W, Zhou P, et al. FOOGD: Federated collaboration for both out-of-distribution generalization and detection[C]//Advances in Neural Information Processing Systems. 2024, 37.
[12] Hendrycks D, Gimpel K. A baseline for detecting misclassified and out-of-distribution examples in neural networks[C/OL]//International Conference on Learning Representations. Toulon, France, 2017[2026-04-01]. https://openreview.net/forum?id=Hkg4TI9xl.
[13] Liu W, Wang X, Owens J, et al. Energy-based out-of-distribution detection[C]//Advances in Neural Information Processing Systems. Red Hook, NY, USA: Curran Associates, Inc., 2020, 33: 21464-21475.
[14] Wang H Q, Li Z, Feng L, et al. ViM: Out-of-distribution with virtual-logit matching[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. New Orleans, LA, USA: IEEE, 2022: 4911-4920.
[15] Fan J, Ke Y, Wang K, et al. Estimating number of factors by adjusted eigenvalues thresholding[J]. Journal of the American Statistical Association, 2022, 117(538): 852-861.
[16] Li J, Yang Z, Chen T. DYB-PlanktonNet[DB/OL]. IEEE Dataport, 2021[2026-04-01]. https://dx.doi.org/10.21227/875n-f104.

---

## 附录

### 全局协方差重构公式

设客户端 `i` 的本地特征为 `\{z_j^{(i)}\}_{j=1}^{n_i}`，则有

$$
\begin{aligned}
S^{(1)} &= \sum_{i=1}^{N}\sum_{j=1}^{n_i} z_j^{(i)}, \\
S^{(2)} &= \sum_{i=1}^{N}\sum_{j=1}^{n_i} z_j^{(i)}(z_j^{(i)})^\top.
\end{aligned}
$$

由此得到

$$
\mu_{\text{global}} = \frac{S^{(1)}}{\sum_i n_i},
\qquad
\Sigma_{\text{global}} = \frac{S^{(2)}}{\sum_i n_i} - \mu_{\text{global}}\mu_{\text{global}}^\top.
$$

因此，ViM 所需的全局统计量可以在联邦场景下通过充分统计量无损重构。

### 五个模型完整结果

**表 A-1 五个模型完整结果**

| 模型 | ID Acc (%) | FedViM k | ACT k | 压缩率 | FedViM Near (%) | ACT Near (%) | MSP Near (%) | Energy Near (%) | FedViM Far (%) | ACT Far (%) | MSP Far (%) | Energy Far (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DenseNet169 | 96.50 | 1000 | 75 | 92.5% | 82.05 | 96.73 | 88.60 | 77.11 | 86.00 | 97.24 | 79.22 | 66.15 |
| EfficientNetV2-S | 97.04 | 512 | 63 | 87.7% | 96.40 | 96.45 | 87.83 | 81.36 | 97.52 | 97.32 | 89.62 | 85.24 |
| MobileNetV3-Large | 95.13 | 512 | 93 | 81.8% | 95.70 | 96.11 | 93.05 | 87.23 | 97.04 | 97.39 | 92.56 | 89.42 |
| ResNet101 | 96.22 | 1000 | 140 | 86.0% | 96.50 | 96.74 | 91.59 | 87.06 | 97.43 | 97.41 | 90.12 | 86.65 |
| ResNet50 | 96.53 | 1000 | 138 | 86.2% | 95.68 | 95.69 | 91.02 | 83.97 | 97.19 | 97.24 | 84.82 | 76.97 |

### 核心算法实现代码

#### `evaluate_fedvim.py`

`evaluate_fedvim.py` 为固定维度 `FedViM` 的评估脚本，在冻结 `best_model` 的基础上聚合各客户端 ID 训练分片上的联邦统计量、执行经验 `alpha` 校准，并输出 Near-OOD 与 Far-OOD 的评测结果。正式排版版在学校 LaTeX 模板附录中以代码清单形式给出完整实现。

#### `advanced_fedvim.py`

`advanced_fedvim.py` 为 `ACT-FedViM` 的后处理评估脚本，在相同联邦统计量基础上利用 ACT 自适应确定主子空间维度 `k`，并完成经验 `alpha` 校准与最终 OOD 打分。正式排版版在学校 LaTeX 模板附录中以代码清单形式给出完整实现。

## 致谢

时光荏苒，本科阶段的学习即将结束，本论文也在反复修改与完善中接近尾声。在论文完成之际，我想向所有在本科阶段给予我帮助、指导和支持的老师、同学与家人致以诚挚的感谢。

首先，我要衷心感谢我的指导教师谢传龙副教授。大二上学期，在学习谢老师开设的《统计学习》课程之后，我第一次真切地感受到统计学在人工智能时代的重要价值。随后，在学习《深度学习》课程的过程中，我又进一步对深度学习领域产生了浓厚兴趣。毕业论文撰写期间，谢老师在论文选题、研究思路、方法设计、实验推进以及论文修改等方面都给予了我耐心而细致的指导。老师严谨认真的治学态度使我受益匪浅，也让我更加深刻地体会到学术研究需要踏实、细致与规范。

同时，我要感谢文理学院统计系各位老师在本科阶段对我的培养。作为统计系第一届本科生，我们始终感受到老师们对班级同学的关心与重视。无论是在课堂上对知识点的细致讲解，还是在课后对问题的耐心解答，老师们都给予了我们很多帮助。在一门门课程的学习过程中，我逐渐体会到钻研问题本身的乐趣，也慢慢发现，自己不再只是为了应对考试而学习，而是真正对统计学产生了兴趣。

此外，我还要感谢大学四年中结识的同学和朋友们。无论是在图书馆里一起学习、讨论问题，还是在健身房里彼此督促、相互鼓励，抑或是在紧张的学习之余一起打几局游戏放松心情，这些真实而具体的陪伴都构成了我大学生活中十分珍贵的记忆。正是因为有你们一路同行，我的大学生活才更加丰富而充实，也让我在学习、生活和心态上都获得了成长与进步。

最后，我要感谢我的家人。感谢你们一直以来对我学习和生活的理解、支持与陪伴。进入大学以后，面对陌生的环境和全新的学习节奏，我在适应过程中经历过压力、迷茫与不安，也体会过成长、收获与喜悦。课程内容的难度、与家乡不同的饮食习惯以及南方潮湿多雨的气候，都曾让我一时感到无所适从；而在学习逐渐步入正轨之后，拿到奖学金时的开心、日常生活中的点滴进步，以及大学生活里那些平凡却有趣的小事，也同样成为我想第一时间与你们分享的内容。无论是在我倾诉压力和烦恼的时候，还是在我分享收获与喜悦的时候，你们始终都耐心倾听，给予我理解、安慰与鼓励。正是在你们始终如一的陪伴和支持下，我逐渐适应了大学生活，也学会了以更加平和而坚定的心态面对成长中的困难与挑战。

谨以此文，向所有关心、帮助和支持我的人表示衷心感谢。
