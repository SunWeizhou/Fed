## FedViM 与 ACT-FedViM 说明

当前论文主线只保留两条 ViM 系方法：

- `FedViM`：联邦化 ViM，采用原版 ViM fixed-k 口径
- `ACT-FedViM`：在同一 `FedViM` checkpoint 上，用 ACT 自动选择主子空间维度 `k`

这里不再把 `variance-0.95`、`ACT-corr`、`WDiscOOD` 等探索分支纳入主流程。

### 1. FedViM 的 fixed-k 口径

`FedViM` 复现 ViM 原文中的 fixed-k heuristic：

- 若特征维度 `N > 1500`，取 `k = 1000`
- 若特征维度 `N <= C`，取 `k = N / 2`
- 否则取 `k = 512`

在当前五模型主线中，对应的 `k` 主要是 `512` 或 `1000`。

评估命令：

```bash
python3 evaluate_fedvim.py \
  --checkpoint path/to/best_model.pth \
  --data_root ./Plankton_OOD_Dataset
```

### 2. ACT-FedViM 的角色

ACT 在本项目中的作用是：

- 不改变主方向的来源
- 仍然在全局协方差矩阵上做 PCA
- 只负责自动选择更合适的主子空间维度 `k`

也就是说，`ACT-FedViM` 不是一个新的训练框架，而是 `FedViM` 的后处理自适应选维扩展。

评估命令：

```bash
python3 evaluate_act_fedvim.py \
  --checkpoint path/to/best_model.pth \
  --data_root ./Plankton_OOD_Dataset
```

### 3. alpha 口径

论文默认使用经验 alpha：

- `alpha_method = empirical`
- 通过各客户端 ID 训练集上的能量与残差统计量聚合得到

这也是当前脚本默认行为。

### 4. 论文叙述建议

当前实验结论应表述为：

- `FedViM` 解决了 ViM 在多中心敏感图像场景下无法直接共享原始数据的问题
- `ACT-FedViM` 为联邦 ViM 提供了统计驱动的自适应选维机制
- 该方法在保持竞争性 OOD 检测性能的同时显著压缩了主子空间规模

不建议写成：

- `ACT-FedViM` 稳定全面优于 `FedViM`
- `ACT` 一定带来 AUROC 提升

### 5. 当前论文实验范围

正式实验只保留 5 个 CNN backbone：

- `resnet101`
- `efficientnet_v2_s`
- `mobilenetv3_large`
- `densenet169`
- `resnet50`
