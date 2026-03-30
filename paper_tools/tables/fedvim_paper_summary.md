# FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究

## 五模型完整结果

| Model | ID Acc | FedViM k | ACT k | ACT Compress | FedViM Near | ACT Near | MSP Near | Energy Near | FedViM Far | ACT Far | MSP Far | Energy Far |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DenseNet169 | 96.50 | 1000 | 99 | 90.1% | 80.04 | 95.57 | 88.60 | 77.11 | 84.78 | 96.17 | 79.22 | 66.15 |
| EfficientNetV2-S | 97.01 | 512 | 69 | 86.5% | 96.40 | 95.77 | 87.83 | 81.36 | 97.52 | 96.51 | 89.62 | 85.24 |
| MobileNetV3-Large | 96.16 | 512 | 89 | 82.6% | 96.31 | 95.26 | 93.05 | 87.23 | 97.34 | 96.42 | 92.56 | 89.42 |
| ResNet101 | 96.22 | 1000 | 143 | 85.7% | 95.86 | 96.28 | 91.59 | 87.06 | 96.68 | 96.73 | 90.12 | 86.65 |
| ResNet50 | 96.53 | 1000 | 141 | 85.9% | 95.68 | 95.32 | 91.02 | 83.97 | 97.19 | 95.89 | 84.82 | 76.97 |

## 正文代表模型

选择原则：覆盖三种最关键情形，分别是轻量化部署案例、压缩与性能平衡案例，以及 fixed-k 失配纠偏案例。

| Model | ID Acc | FedViM k | ACT k | ACT Compress | FedViM Near | ACT Near | ACT-FedViM vs FedViM | FedViM Far | ACT Far |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large | 96.16 | 512 | 89 | 82.6% | 96.31 | 95.26 | -1.05 | 97.34 | 96.42 |
| ResNet101 | 96.22 | 1000 | 143 | 85.7% | 95.86 | 96.28 | +0.42 | 96.68 | 96.73 |
| DenseNet169 | 96.50 | 1000 | 99 | 90.1% | 80.04 | 95.57 | +15.53 | 84.78 | 96.17 |

## 方法平均表现

| Method | Avg ID Acc | Avg Near AUROC | Avg Far AUROC | Avg k | Avg Compression |
| --- | --- | --- | --- | --- | --- |
| FedViM | 96.48 | 92.86 | 94.70 | 804 | - |
| ACT-FedViM | 96.48 | 95.64 | 96.34 | 108 | 86.2% |
| MSP | 96.48 | 90.42 | 87.27 | - | - |
| Energy | 96.48 | 83.34 | 80.89 | - | - |

## 摘要口径

- 实验范围：ResNet101, EfficientNetV2-S, MobileNetV3-Large, DenseNet169, ResNet50
- 正文代表模型：MobileNetV3-Large, ResNet101, DenseNet169
- ACT 相对 FedViM 的平均 Near-OOD AUROC 变化：+2.78
- ACT 相对 FedViM 的平均 Far-OOD AUROC 变化：+1.64
- ACT 平均子空间压缩率：86.2%
