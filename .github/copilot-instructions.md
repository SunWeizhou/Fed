## 快速目标（对 AI 编码代理）

这份文件给出使你能够立刻在此仓库中高效工作的关键信息：项目整体架构、常用运行命令、代码风格约定、以及常见的实现模式与注意点。优先保持最小破坏：修改代码时优先改 `utils/`、`paper_tools/` 或新增模块；避免直接改动 `experiments/` 结果目录。

## 大局架构要点
- 主入口：`train_federated.py`（联邦训练），`advanced_fedvim.py`（ACT 后处理），`evaluate_model.py`（评估）。
- 客户端/服务端：`client.py` / `server.py`（客户端计算一阶/二阶统计量，本地训练；服务端聚合并做 PCA 提取子空间）。
- 模型与配置：`models.py`（模型集合），`config.py`（集中式默认参数）。
- 快速原型：`examples/fed_vim_cifar.py`（单文件用于快速验证）。

## 关键数据流与设计意图
- 客户端仅共享统计量（E[z], E[zz^T]），不共享原始特征以保护隐私。
- 服务端聚合后计算协方差并用 `torch.linalg.eigh` 做谱分解提取主子空间 P（见 `server.py`）。
- 记忆优化示例：用 `stat_sum_zzT = torch.matmul(features.T, features)` 替代逐样本外积以节省显存（见 README 的代码注记）。

## OOD 打分核心公式（务必保持一致）
- Energy(x) = logsumexp(logits)
- Residual(x) = || (I - P P^T) (f(x) - μ_global) ||_2
- Score(x) = Energy(x) - α × Residual(x)
  - α 的经验估计在代码中以训练集统计量计算（参见 `advanced_fedvim.py` 与 `evaluate_model.py`）。
  - 注意：实现里分数越高越偏向 ID；评估指标（AUROC 等）时常对其取反为 OOD score。

## 常用开发 / 运行命令（必记）
- 安装依赖： `pip install -r requirements.txt`。
- 语法检查（最小测试）： `python3 -B -m py_compile train_federated.py advanced_fedvim.py server.py`。
- 训练示例： `python train_federated.py --model_type resnet50 --use_fedvim --data_root ./Plankton_OOD_Dataset`。
- ACT 后处理示例：
  `python advanced_fedvim.py --checkpoint path/to/best_model.pth --data_root ./Plankton_OOD_Dataset --subspace_method act --alpha_method empirical`

## 项目特有约定与风格
- Python 风格：4 空格缩进，snake_case。
- 提交信息：短动词开头（例如：`Add experiment analysis scripts`）。
- 无完整 pytest 套件：验证通常通过 `py_compile`、运行脚本和检查生成文件（`paper_tools/` 输出、`experiments/` 下的结果）。

## 常见修改点与安全边界
- 可以改进的安全位置：`utils/`、`paper_tools/`、`docs/`、`examples/`。优先在这些目录新增工具或测试。
- 小心直接改 `experiments/`（这是实验输出）；修改 `models.py`、`config.py`、`client.py`、`server.py` 会影响实验复现。

## 对 AI 代理的具体提示（行动项）
- 读这几处文件以迅速定位逻辑：`train_federated.py`, `server.py`, `client.py`, `advanced_fedvim.py`, `models.py`, `config.py`, `data_utils.py`。
- 若要变更子空间选维或 α 的计算：修改 `server.py` 的聚合逻辑或 `advanced_fedvim.py` 的后处理实现，并在 `paper_tools/` 添加对比脚本。
- 若需添加单元/集成验证：新增小脚本在 `examples/`（保持单文件、易运行），并在 README 或 AGENTS.md 中记录运行命令。

## 发现示例（快速查阅）
- 协方差低内存实现：`stat_sum_zzT = torch.matmul(features.T, features)`（README 中示例）。
- 子空间提取：`eig_vals, eig_vecs = torch.linalg.eigh(Cov_global)` → `P_global = eig_vecs[:, -k:]`（README/`server.py`）。

## 文档与论文产物
- 任何改动若影响评估流程或指标（例如 α 的估计、子空间截断策略、ViM 公式实现）必须同步更新 `docs/paper/` 与 `paper_tools/` 中的生成脚本和表格。

## 反馈与迭代
在你审阅后告诉我：哪些段落需要更详细的代码示例（或要合并 `AGENTS.md` 中其他段落）。我会根据你的反馈迭代这份指引。
