# FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究

## 五模型完整结果

| Model | ID Acc | FedViM k | ACT k | ACT Compress | FedViM Near | ACT Near | MSP Near | Energy Near | FedViM Far | ACT Far | MSP Far | Energy Far |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DenseNet169 | 93.98 | 1000 | 77 | 92.3% | 84.86 | 96.30 | 83.27 | 72.29 | 85.98 | 97.17 | 79.39 | 71.22 |
| EfficientNetV2-S | 96.33 | 512 | 61 | 88.1% | 96.18 | 95.72 | 87.64 | 84.32 | 96.91 | 96.50 | 86.48 | 82.84 |
| MobileNetV3-Large | 95.64 | 512 | 94 | 81.6% | 96.57 | 96.18 | 90.49 | 83.86 | 96.95 | 96.97 | 88.77 | 83.83 |
| ResNet101 | 95.88 | 1000 | 138 | 86.2% | 97.01 | 97.19 | 82.14 | 77.12 | 96.27 | 96.44 | 79.69 | 76.12 |
| ResNet50 | 95.99 | 1000 | 145 | 85.5% | 96.00 | 96.44 | 87.31 | 81.36 | 95.85 | 95.91 | 78.32 | 72.08 |

## 正文代表模型

选择原则：覆盖三种最关键情形，分别是轻量化部署案例、压缩与性能平衡案例，以及 fixed-k 失配纠偏案例。

| Model | ID Acc | FedViM k | ACT k | ACT Compress | FedViM Near | ACT Near | ACT-FedViM vs FedViM | FedViM Far | ACT Far |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large | 95.64 | 512 | 94 | 81.6% | 96.57 | 96.18 | -0.39 | 96.95 | 96.97 |
| ResNet101 | 95.88 | 1000 | 138 | 86.2% | 97.01 | 97.19 | +0.18 | 96.27 | 96.44 |
| DenseNet169 | 93.98 | 1000 | 77 | 92.3% | 84.86 | 96.30 | +11.44 | 85.98 | 97.17 |

## 方法平均表现

| Method | Avg ID Acc | Avg Near AUROC | Avg Far AUROC | Avg k | Avg Compression |
| --- | --- | --- | --- | --- | --- |
| FedViM | 95.56 | 94.12 | 94.39 | 804 | - |
| ACT-FedViM | 95.56 | 96.37 | 96.60 | 103 | 86.7% |
| MSP | 95.56 | 86.17 | 82.53 | - | - |
| Energy | 95.56 | 79.79 | 77.22 | - | - |

## 摘要口径

- 实验范围：ResNet101, EfficientNetV2-S, MobileNetV3-Large, DenseNet169, ResNet50
- 正文代表模型：MobileNetV3-Large, ResNet101, DenseNet169
- ACT 相对 FedViM 的平均 Near-OOD AUROC 变化：+2.24
- ACT 相对 FedViM 的平均 Far-OOD AUROC 变化：+2.21
- ACT 平均子空间压缩率：86.7%
