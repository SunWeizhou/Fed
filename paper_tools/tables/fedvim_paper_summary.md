# FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究

## 五模型完整结果

| Model | ID Acc | FedViM k | ACT k | ACT Compress | FedViM Near | ACT Near | MSP Near | Energy Near | FedViM Far | ACT Far | MSP Far | Energy Far |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DenseNet169 | 96.50 | 1000 | 75 | 92.5% | 82.05 | 96.73 | 88.60 | 77.11 | 86.00 | 97.24 | 79.22 | 66.15 |
| EfficientNetV2-S | 97.04 | 512 | 63 | 87.7% | 96.40 | 96.45 | 87.83 | 81.36 | 97.52 | 97.32 | 89.62 | 85.24 |
| MobileNetV3-Large | 95.13 | 512 | 93 | 81.8% | 95.70 | 96.11 | 93.05 | 87.23 | 97.04 | 97.39 | 92.56 | 89.42 |
| ResNet101 | 96.22 | 1000 | 140 | 86.0% | 96.50 | 96.74 | 91.59 | 87.06 | 97.43 | 97.41 | 90.12 | 86.65 |
| ResNet50 | 96.53 | 1000 | 138 | 86.2% | 95.68 | 95.69 | 91.02 | 83.97 | 97.19 | 97.24 | 84.82 | 76.97 |

## 正文代表模型

选择原则：覆盖三种最关键情形，分别是轻量化部署案例、压缩与性能平衡案例，以及 fixed-k 失配纠偏案例。

| Model | ID Acc | FedViM k | ACT k | ACT Compress | FedViM Near | ACT Near | ACT-FedViM vs FedViM | FedViM Far | ACT Far |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large | 95.13 | 512 | 93 | 81.8% | 95.70 | 96.11 | +0.41 | 97.04 | 97.39 |
| ResNet101 | 96.22 | 1000 | 140 | 86.0% | 96.50 | 96.74 | +0.23 | 97.43 | 97.41 |
| DenseNet169 | 96.50 | 1000 | 75 | 92.5% | 82.05 | 96.73 | +14.68 | 86.00 | 97.24 |

## 方法平均表现

| Method | Avg ID Acc | Avg Near AUROC | Avg Far AUROC | Avg k | Avg Compression |
| --- | --- | --- | --- | --- | --- |
| FedViM | 96.28 | 93.27 | 95.03 | 804 | - |
| ACT-FedViM | 96.28 | 96.34 | 97.32 | 101 | 86.8% |
| MSP | 96.28 | 90.42 | 87.27 | - | - |
| Energy | 96.28 | 83.34 | 80.89 | - | - |

## 摘要口径

- 实验范围：ResNet101, EfficientNetV2-S, MobileNetV3-Large, DenseNet169, ResNet50
- 正文代表模型：MobileNetV3-Large, ResNet101, DenseNet169
- ACT 相对 FedViM 的平均 Near-OOD AUROC 变化：+3.08
- ACT 相对 FedViM 的平均 Far-OOD AUROC 变化：+2.28
- ACT 平均子空间压缩率：86.8%
