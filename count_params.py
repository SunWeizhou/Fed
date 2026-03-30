#!/usr/bin/env python3
"""
计算9款模型的参数量
"""

import torch
import sys
sys.path.insert(0, '.')

from models import Backbone, FedAvg_Model

# 9款模型配置
MODELS = [
    # 经典 CNN
    ('ResNet50', 'resnet50'),
    ('ResNet101', 'resnet101'),
    ('DenseNet169', 'densenet169'),
    # 现代 CNN
    ('EfficientNetV2-S', 'efficientnet_v2_s'),
    ('MobileNetV3-L', 'mobilenetv3_large'),
    ('ConvNeXt-Base', 'convnext_base'),
    # Transformer
    ('ViT-B/16', 'vit_b_16'),
    ('ViT-B/32', 'vit_b_32'),
    ('DeiT-Base', 'deit_base'),
]

NUM_CLASSES = 54

print("="*70)
print("9款模型参数量统计")
print("="*70)
print(f"{'模型':<20} {'Backbone参数':<15} {'分类器参数':<15} {'总参数':<15}")
print("-"*70)

total_backbone = 0
total_classifier = 0
total_all = 0

results = []

for model_name, model_type in MODELS:
    try:
        # 创建模型
        backbone = Backbone(model_type=model_type, pretrained=False)
        model = FedAvg_Model(backbone, num_classes=NUM_CLASSES)

        # 计算参数量
        backbone_params = sum(p.numel() for p in backbone.parameters())
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
        total_params = backbone_params + classifier_params

        results.append((model_name, backbone_params, classifier_params, total_params))

        total_backbone += backbone_params
        total_classifier += classifier_params
        total_all += total_params

        print(f"{model_name:<20} {backbone_params:>14,}  {classifier_params:>14,}  {total_params:>14,}")

    except Exception as e:
        print(f"{model_name:<20} Error: {e}")

print("-"*70)
print(f"{'总计':<20} {total_backbone:>14,}  {total_classifier:>14,}  {total_all:>14,}")

print("\n" + "="*70)
print("参数量换算 (M = Million)")
print("="*70)

for model_name, backbone, classifier, total in results:
    print(f"{model_name:<20} {backbone/1e6:.2f}M  {classifier/1e6:.2f}M  {total/1e6:.2f}M")

# 排序
print("\n按总参数量排序:")
results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
print(f"{'排名':<6} {'模型':<20} {'总参数':<15}")
print("-"*50)
for i, (name, _, _, total) in enumerate(results_sorted, 1):
    print(f"  {i:<4}  {name:<20} {total:>14,} ({total/1e6:.2f}M)")
