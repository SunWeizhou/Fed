#!/usr/bin/env python3
"""
模型定义模块。

当前仓库仅保留 Fed-ViM 主线所需的骨干网络和单头分类器。
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    DenseNet169_Weights,
    EfficientNet_V2_S_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)


class Backbone(nn.Module):
    """
    通用骨干网络包装器。

    当前论文版仅支持 5 个 CNN backbone:
    DenseNet169, ResNet50, ResNet101, EfficientNetV2-S, MobileNetV3-Large
    """

    def __init__(self, model_type="densenet169", pretrained=True):
        super().__init__()
        self.model_type = model_type.lower()

        if self.model_type == "densenet169":
            weights = DenseNet169_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet169(weights=weights)
            self.feature_dim = 1664
            self.backbone.classifier = nn.Identity()

        elif self.model_type == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
            self.feature_dim = 1280
            self.backbone.classifier[1] = nn.Identity()

        elif self.model_type == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()

        elif self.model_type == "mobilenetv3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            self.feature_dim = 960
            self.backbone.classifier = nn.Identity()

        elif self.model_type == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def forward(self, x):
        return self.backbone(x)


class FedAvg_Model(nn.Module):
    """Fed-ViM 主训练模型。"""

    def __init__(self, backbone, num_classes=54, hidden_dim=512):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = backbone.feature_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


def create_model(model_type="densenet169", num_classes=54):
    """创建 Fed-ViM 单头分类模型。"""
    backbone = Backbone(model_type=model_type, pretrained=True)
    return FedAvg_Model(backbone, num_classes=num_classes)


if __name__ == "__main__":
    print("测试 Fed-ViM 模型...")
    model = create_model()
    dummy_input = torch.randn(4, 3, 224, 224)
    logits, features = model(dummy_input)
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"分类输出尺寸: {logits.shape}")
    print(f"特征向量尺寸: {features.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")
