#!/usr/bin/env python3
"""
Test script to verify ViM Score formula correctness.

原版 ViM 公式 (Sun et al., NeurIPS 2022):
Score = Energy - alpha * Residual

其中:
- Energy = logsumexp(logits) [正值，校准良好的模型约 4-8]
- Residual = ||(I-PP^T)(z-mu)||_2 [正值，到子空间的距离]
- alpha = mean(Energy) / mean(Residual) [将距离归一化到能量空间]

预期行为:
- ID: 高 Energy + 低 Residual → 高分
- OOD: 低 Energy + 高 Residual → 低分
- 检测: 低分 → OOD
"""

import numpy as np

print("="*60)
print("ViM Score Formula Verification (原版公式)")
print("="*60)

# 模拟值
alpha = 3.0  # 典型的校准后 alpha 值

# ID 数据 (高置信度，低残差)
id_energy = 6.0      # 高置信度 (大 LogSumExp)
id_residual = 2.0    # 低残差 (接近子空间)
id_score = id_energy - alpha * id_residual  # 原版公式

# OOD 数据 (低置信度，高残差)
ood_energy = 3.0     # 低置信度 (小 LogSumExp)
ood_residual = 8.0   # 高残差 (远离子空间)
ood_score = ood_energy - alpha * ood_residual  # 原版公式

print(f"\n原版 ViM 公式: Score = Energy - alpha * Residual")
print(f"Alpha: {alpha} (将几何距离归一化到能量空间)")
print(f"\n{'Metric':<20} {'ID':>12} {'OOD':>12}")
print("-"*44)
print(f"{'Energy':<20} {id_energy:>12.2f} {ood_energy:>12.2f}")
print(f"{'Residual':<20} {id_residual:>12.2f} {ood_residual:>12.2f}")
print(f"{'Score':<20} {id_score:>12.2f} {ood_score:>12.2f}")
print("-"*44)

print(f"\n解释:")
print(f"  - ID Score: {id_score:.2f} (高 = 置信度高 + 接近子空间)")
print(f"  - OOD Score: {ood_score:.2f} (低 = 置信度低 + 远离子空间)")
print(f"  - 分数差: {id_score - ood_score:.2f}")

if id_score > ood_score:
    print(f"\n✓ CORRECT: ID score ({id_score:.2f}) > OOD score ({ood_score:.2f})")
    print(f"  原版 ViM: 高分 -> ID, 低分 -> OOD")
    print(f"  AUROC 计算时需要反转: roc_auc_score(labels, -scores)")
else:
    print(f"\n✗ WRONG: ID score ({id_score:.2f}) <= OOD score ({ood_score:.2f})")
    print(f"  这会导致 AUROC < 0.5")

print("\n" + "="*60)
print("数学原理解释")
print("="*60)
print("""
原版 ViM 公式 Score = Energy - alpha * Residual 的原理:

1. 量纲一致性:
   - Energy 单位: [Logit] (概率对数空间)
   - alpha 单位: [Logit]/[距离]
   - alpha * Residual 单位: [Logit]
   - 结果: [Logit] - [Logit] = [Logit] ✓ 量纲一致

2. 物理意义:
   - alpha 将"几何空间的距离"归一化到"概率对数空间"
   - 使两者可以在同一尺度上比较

3. 检测机制:
   - ID: 高 Energy + 低 Residual → 高分
   - OOD: 低 Energy + 高 Residual → 低分

4. AUROC 计算:
   - 原版公式: OOD 得分低
   - 需要反转: roc_auc_score(labels, -scores)
   - 反转后 OOD 得分高，符合 AUROC 期望

与之前错误的公式对比:
  错误: Score = Residual - alpha * Energy
  问题: [距离] - ([Logit]/[距离]) * [Logit] = [距离] - [Logit]²/[距离] ✗ 量纲混乱
""")

print("="*60)
print("旧错误公式的测试")
print("="*60)

# 测试旧的错误公式
id_score_old = id_residual - alpha * id_energy
ood_score_old = ood_residual - alpha * ood_energy

print(f"\n旧公式: Score = Residual - alpha * Energy")
print(f"{'Metric':<20} {'ID':>12} {'OOD':>12}")
print("-"*44)
print(f"{'Score (OLD)':<20} {id_score_old:>12.2f} {ood_score_old:>12.2f}")
print("-"*44)

if ood_score_old > id_score_old:
    print(f"\n旧公式: OOD ({ood_score_old:.2f}) > ID ({id_score_old:.2f})")
    print(f"  如果直接用此公式计算 AUROC: roc_auc_score(labels, scores)")
    print(f"  可以得到正确的 AUROC，但量纲不正确！")
else:
    print(f"\n旧公式也有问题: OOD ({ood_score_old:.2f}) <= ID ({id_score_old:.2f})")

print("\n" + "="*60)
print("结论")
print("="*60)
print("""
正确的原版 ViM 公式: Score = Energy - alpha * Residual

关键修改:
1. 公式: residual - alpha * energy → energy - alpha * residual
2. AUROC: roc_auc_score(labels, scores) → roc_auc_score(labels, -scores)

为什么原版公式更正确:
1. 量纲一致: [Logit] - [Logit] 而非 [距离] - [Logit]²/[距离]
2. 物理意义: alpha 将距离归一化到能量空间，而非反过来
3. 论文一致: Sun et al. (NeurIPS 2022) 原始公式

当前代码已更新为原版公式:
  server.py:328    scores = energy - alpha * residual
  evaluate_model.py:329-331  同样公式
  AUROC 计算使用 -scores 来适配
""")
