# FedViM

**论文题目**：`FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究`

本仓库当前只维护本科论文主线：

- `FedViM`：联邦化 ViM，采用原版 ViM fixed-k 口径
- `ACT-FedViM`：在同一 `FedViM` checkpoint 上用 ACT 自动选择主子空间维度 `k`
- `MSP`
- `Energy`

研究背景固定为海洋浮游生物多中心监测。各客户端对应不同数据中心，原始图像及其采样相关敏感信息不直接共享；服务器只聚合一阶与二阶特征统计量。

## 当前实验范围

论文主实验只保留 5 个 CNN 家族 backbone：

- `resnet101`
- `efficientnet_v2_s`
- `mobilenetv3_large`
- `densenet169`
- `resnet50`

`ViT`、`DeiT`、`convnext_base` 和其他探索方法不再进入论文主结果。

## 安装

```bash
pip install -r requirements.txt
```

## 主流程

### 1. 训练 FedViM

```bash
python3 train_federated.py \
  --model_type resnet101 \
  --use_fedvim \
  --data_root ./Plankton_OOD_Dataset
```

训练脚本现在只负责：

- 联邦训练 backbone
- 保存 `best_model.pth` / `final_model.pth`
- 保存 `training_history.json`
- 保存 ViM 所需聚合统计量

训练阶段不再直接运行 OOD 评估。

### 2. 评估 FedViM

```bash
python3 evaluate_fedvim.py \
  --checkpoint ./experiments/experiments_rerun_v1/resnet101/experiment_xxx/best_model.pth \
  --data_root ./Plankton_OOD_Dataset
```

### 3. 评估 ACT-FedViM

```bash
python3 evaluate_act_fedvim.py \
  --checkpoint ./experiments/experiments_rerun_v1/resnet101/experiment_xxx/best_model.pth \
  --data_root ./Plankton_OOD_Dataset
```

### 4. 评估 MSP / Energy

```bash
python3 evaluate_baselines.py \
  --checkpoint ./experiments/experiments_rerun_v1/resnet101/experiment_xxx/best_model.pth \
  --data_root ./Plankton_OOD_Dataset \
  --methods msp energy
```

四种方法都会输出结构化 JSON，供论文汇总脚本直接读取。

## 五模型结果汇总

```bash
python3 paper_tools/collect_paper_results.py \
  --experiments-root experiments/experiments_rerun_v1 \
  --output-prefix paper_tools/rerun_v1_results

python3 paper_tools/generate_paper_tables.py
```

生成物包括：

- 五模型完整对比表
- 正文 2 到 3 个代表模型表
- 方法平均表现表
- 论文摘要口径 markdown

## 多机脚本

五模型一机一卡训练按以下映射固定：

- `dell@10.24.1.131` `cuda:1` -> `resnet101`
- 本机 `cuda:0` -> `densenet169`
- 本机 `cuda:1` -> `efficientnet_v2_s`
- `dell7960@10.4.47.203` `cuda:0` -> `resnet50`
- `dell7960@10.4.47.203` `cuda:1` -> `mobilenetv3_large`

脚本入口：

- `sync_fedvim_5models.sh`：同步代码和完整 `Plankton_OOD_Dataset`
- `launch_fedvim_5models.sh`：在 5 张卡上一键启动训练+四方法评估
- `fetch_fedvim_results.sh`：把远程实验结果回收到主机
- `run_fedvim_model_pipeline.sh`：单模型训练+评估流水线

## 目录说明

- `train_federated.py`：纯训练入口
- `evaluate_fedvim.py`：`FedViM` fixed-k 评估
- `evaluate_act_fedvim.py`：`ACT-FedViM` 评估
- `evaluate_baselines.py`：`MSP` / `Energy`
- `paper_tools/collect_paper_results.py`：五模型结果收集
- `paper_tools/generate_paper_tables.py`：markdown 表格生成
- `docs/paper/ACT_FedViM_论文草稿.md`：当前论文草稿
- `experiments/experiments_rerun_v1/`：实验输出

## 说明

- `wdiscood/`、旧 handover 文档和历史结果仍保留，但不属于论文主线。
- 当前正式论文口径是 `FedViM + empirical alpha`，`ACT-FedViM` 作为后处理自适应选维扩展。
