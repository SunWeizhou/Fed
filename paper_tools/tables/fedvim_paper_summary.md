# FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究

## 五模型完整结果

| Model | ID Acc | FedViM k | ACT k | ACT Compress | FedViM Near | ACT Near | MSP Near | Energy Near | FedViM Far | ACT Far | MSP Far | Energy Far |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| densenet169 | 96.50 | 1000 | 99 | 90.1% | 80.04 | 95.57 | 88.60 | 77.11 | 84.78 | 96.17 | 79.22 | 66.15 |
| efficientnet_v2_s | 97.01 | 512 | 69 | 86.5% | 96.40 | 95.77 | 87.83 | 81.36 | 97.52 | 96.51 | 89.62 | 85.24 |
| mobilenetv3_large | 96.16 | 512 | 89 | 82.6% | 96.31 | 95.26 | 93.05 | 87.23 | 97.34 | 96.42 | 92.56 | 89.42 |
| resnet101 | 96.22 | 1000 | 143 | 85.7% | 95.86 | 96.28 | 91.59 | 87.06 | 96.68 | 96.73 | 90.12 | 86.65 |
| resnet50 | 96.53 | 1000 | 141 | 85.9% | 95.68 | 95.32 | 91.02 | 83.97 | 97.19 | 95.89 | 84.82 | 76.97 |

## 正文代表模型

选择规则：按 `ACT-FedViM` 的 Near-OOD AUROC 排序，并过滤掉相对 `FedViM` 的明显 Near-OOD 退化模型。

| Model | ID Acc | FedViM k | ACT k | ACT Compress | FedViM Near | ACT Near | ACT-FedViM vs FedViM | FedViM Far | ACT Far |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet101 | 96.22 | 1000 | 143 | 85.7% | 95.86 | 96.28 | +0.42 | 96.68 | 96.73 |
| efficientnet_v2_s | 97.01 | 512 | 69 | 86.5% | 96.40 | 95.77 | -0.62 | 97.52 | 96.51 |
| densenet169 | 96.50 | 1000 | 99 | 90.1% | 80.04 | 95.57 | +15.53 | 84.78 | 96.17 |

## 方法平均表现

| Method | Avg ID Acc | Avg Near AUROC | Avg Far AUROC | Avg k | Avg Compression |
| --- | --- | --- | --- | --- | --- |
| FedViM | 96.48 | 92.86 | 94.70 | 804 | - |
| ACT-FedViM | 96.48 | 95.64 | 96.34 | 108 | 86.2% |
| MSP | 96.48 | 90.42 | 87.27 | - | - |
| Energy | 96.48 | 83.34 | 80.89 | - | - |

## 摘要口径

- 实验范围：resnet101, efficientnet_v2_s, mobilenetv3_large, densenet169, resnet50
- 正文代表模型：resnet101, efficientnet_v2_s, densenet169
- ACT 相对 FedViM 的平均 Near-OOD AUROC 变化：+2.78
- ACT 相对 FedViM 的平均 Far-OOD AUROC 变化：+1.64
- ACT 平均子空间压缩率：86.2%
