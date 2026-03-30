#!/usr/bin/env python3
"""
遗留评估可视化工具。

说明:
- 当前 ACT-FedViM 论文主线优先使用 `server.py`、`evaluate_model.py` 和 `paper_tools/`。
- 本文件只保留历史实验仍可能复用的绘图函数，避免继续维护第二套评估实现。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_ood_detection_results(id_scores, ood_scores, output_path):
    """
    绘制 OOD 分数分布、ROC 曲线和 PR 曲线。
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(id_scores, bins=50, alpha=0.7, label="ID", density=True)
    plt.hist(ood_scores, bins=50, alpha=0.7, label="OOD", density=True)
    plt.xlabel("OOD Score")
    plt.ylabel("Density")
    plt.legend()
    plt.title("OOD Score Distribution")

    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])

    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    plt.plot(fpr, tpr, "b-", label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], "r--", label="Random Classifier")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.title("ROC Curve")

    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = average_precision_score(labels, scores)
    plt.plot(recall, precision, "g-", label=f"AUPR = {aupr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curve")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_long_tail_performance(class_accs, class_sample_counts, class_names, output_path):
    """
    绘制类别频次和分类准确率的长尾关系图。
    """
    data = []
    for class_id, accuracy in class_accs.items():
        data.append(
            {
                "id": class_id,
                "name": class_names[class_id],
                "acc": accuracy,
                "count": class_sample_counts.get(class_id, 0),
            }
        )

    data.sort(key=lambda item: item["count"], reverse=True)

    sorted_accs = [item["acc"] for item in data]
    sorted_counts = [item["count"] for item in data]
    indices = np.arange(len(data))

    fig, ax1 = plt.subplots(figsize=(15, 6))

    ax1.set_xlabel("Classes (Sorted by Sample Frequency)", fontsize=12)
    ax1.set_ylabel("Number of Samples (Log Scale)", color="tab:gray", fontsize=12)
    ax1.bar(indices, sorted_counts, color="tab:gray", alpha=0.3, label="Sample Count")
    ax1.tick_params(axis="y", labelcolor="tab:gray")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Test Accuracy", color="tab:blue", fontsize=12)
    ax2.plot(
        indices,
        sorted_accs,
        color="tab:blue",
        marker="o",
        linewidth=2,
        markersize=4,
        label="Accuracy",
    )
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0, 1.05)

    mid_point = int(len(data) * 0.5)
    ax1.axvline(x=mid_point, color="red", linestyle="--", alpha=0.5)
    ax1.text(mid_point + 1, 0.1, "Tail Classes ->", color="red", fontsize=12)
    ax1.text(mid_point - 5, 0.1, "<- Head Classes", color="red", fontsize=12)

    ax1.set_title("Class-wise Performance over Long-tailed Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


__all__ = [
    "plot_long_tail_performance",
    "plot_ood_detection_results",
]
