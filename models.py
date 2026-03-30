#!/usr/bin/env python3
"""
模型定义模块。

当前仓库仅保留 Fed-ViM 主线所需的骨干网络和单头分类器。
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
)

try:
    from torchvision.models import ConvNeXt_Tiny_Weights, EfficientNet_V2_S_Weights
except ImportError:
    ConvNeXt_Tiny_Weights = None
    EfficientNet_V2_S_Weights = None

try:
    from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights
except ImportError:
    ViT_B_16_Weights = None
    ViT_B_32_Weights = None

try:
    from torchvision.models import (
        MobileNet_V3_Large_Weights,
        ConvNeXt_Base_Weights,
        ResNet101_Weights,
    )
except ImportError:
    MobileNet_V3_Large_Weights = None
    ConvNeXt_Base_Weights = None
    ResNet101_Weights = None

try:
    import timm

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm not installed. DeiT models will not be available.")
    print("Install with: pip install timm")


class Backbone(nn.Module):
    """
    通用骨干网络包装器。

    支持: DenseNet, ResNet, ConvNeXt, EfficientNetV2, Swin, ViT, MobileNet, DeiT
    """

    def __init__(self, model_type="densenet121", pretrained=True):
        super().__init__()
        self.model_type = model_type.lower()

        if self.model_type.startswith("densenet"):
            if self.model_type == "densenet121":
                weights = DenseNet121_Weights.DEFAULT if pretrained else None
                self.backbone = models.densenet121(weights=weights)
                self.feature_dim = 1024
            elif self.model_type == "densenet169":
                weights = DenseNet169_Weights.DEFAULT if pretrained else None
                self.backbone = models.densenet169(weights=weights)
                self.feature_dim = 1664
            elif self.model_type == "densenet201":
                weights = DenseNet201_Weights.DEFAULT if pretrained else None
                self.backbone = models.densenet201(weights=weights)
                self.feature_dim = 1920
            else:
                raise ValueError(f"不支持的 DenseNet 类型: {model_type}")
            self.backbone.classifier = nn.Identity()

        elif self.model_type == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if (pretrained and ConvNeXt_Tiny_Weights) else None
            if ConvNeXt_Tiny_Weights:
                self.backbone = models.convnext_tiny(weights=weights)
            else:
                self.backbone = models.convnext_tiny(pretrained=pretrained)
            self.feature_dim = 768
            self.backbone.classifier[2] = nn.Identity()

        elif self.model_type == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.DEFAULT if (pretrained and EfficientNet_V2_S_Weights) else None
            if EfficientNet_V2_S_Weights:
                self.backbone = models.efficientnet_v2_s(weights=weights)
            else:
                self.backbone = models.efficientnet_v2_s(pretrained=pretrained)
            self.feature_dim = 1280
            self.backbone.classifier[1] = nn.Identity()

        elif self.model_type == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()

        elif self.model_type == "swin_t":
            weights = Swin_T_Weights.DEFAULT if pretrained else None
            self.backbone = models.swin_t(weights=weights)
            self.feature_dim = 768
            self.backbone.head = nn.Identity()

        elif self.model_type == "vit_b_16":
            weights = ViT_B_16_Weights.DEFAULT if (pretrained and ViT_B_16_Weights) else None
            if ViT_B_16_Weights:
                self.backbone = models.vit_b_16(weights=weights)
            else:
                self.backbone = models.vit_b_16(pretrained=pretrained)
            self.feature_dim = 768
            self.backbone.heads.head = nn.Identity()

        elif self.model_type == "vit_b_32":
            weights = ViT_B_32_Weights.DEFAULT if (pretrained and ViT_B_32_Weights) else None
            if ViT_B_32_Weights:
                self.backbone = models.vit_b_32(weights=weights)
            else:
                self.backbone = models.vit_b_32(pretrained=pretrained)
            self.feature_dim = 768
            self.backbone.heads.head = nn.Identity()

        elif self.model_type == "mobilenetv3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if (pretrained and MobileNet_V3_Large_Weights) else None
            if MobileNet_V3_Large_Weights:
                self.backbone = models.mobilenet_v3_large(weights=weights)
            else:
                self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            self.feature_dim = 960
            self.backbone.classifier = nn.Identity()

        elif self.model_type == "convnext_base":
            weights = ConvNeXt_Base_Weights.DEFAULT if (pretrained and ConvNeXt_Base_Weights) else None
            if ConvNeXt_Base_Weights:
                self.backbone = models.convnext_base(weights=weights)
            else:
                self.backbone = models.convnext_base(pretrained=pretrained)
            self.feature_dim = 1024
            self.backbone.classifier[2] = nn.Identity()

        elif self.model_type == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()

        elif self.model_type == "deit_base":
            if not HAS_TIMM:
                raise ImportError("DeiT requires timm library. Install with: pip install timm")
            self.backbone = timm.create_model("deit_base_patch16_224", pretrained=pretrained)
            self.feature_dim = 768
            self.backbone.head = nn.Identity()

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
