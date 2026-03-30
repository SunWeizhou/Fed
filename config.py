"""
Centralized configuration for Fed-ViM training

This module consolidates all hyperparameters and model-specific configurations
to eliminate magic numbers and improve maintainability.
"""

class TrainingConfig:
    """Training hyperparameters"""

    # Learning rate settings
    BASE_LR = 0.001
    MIN_LR_FACTOR = 0.1  # Don't decay below this fraction of base LR
    WARMUP_ROUNDS = 5

    # SGD-specific (for ResNet/DenseNet/EfficientNet)
    SGD_LR_MULTIPLIER = 10.0  # Scale up LR for SGD
    SGD_MOMENTUM = 0.9

    # Regularization
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP_NORM = 5.0

    # Early stopping
    EARLY_STOP_PATIENCE = 10
    EARLY_STOP_MIN_DELTA = 0.001
    OOD_TOLERANCE = 0.05  # 5% degradation tolerance

    # Data augmentation
    MIXUP_ALPHA = 1.0
    MIXUP_PROB = 0.5
    FOURIER_BETA = 0.4
    FOURIER_PROB = 0.9

    # Training duration
    DEFAULT_COMMUNICATION_ROUNDS = 50
    DEFAULT_LOCAL_EPOCHS = 4
    DEFAULT_BATCH_SIZE = 32


class ModelConfig:
    """Model-specific configurations"""

    # Feature dimensions
    FEATURE_DIMS = {
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet201': 1920,
        'resnet50': 2048,
        'resnet101': 2048,
        'convnext_tiny': 768,
        'convnext_base': 1024,
        'swin_t': 768,
        'vit_b_16': 768,
        'vit_b_32': 768,
        'deit_base': 768,
        'mobilenetv3_large': 960,
        'efficientnet_v2_s': 1280
    }

    # BatchNorm freezing recommendations
    # Note: ViT/DeiT use LayerNorm, not BatchNorm, so freeze_bn has no effect
    FREEZE_BN_DEFAULTS = {
        'convnext_tiny': 1,      # CRITICAL for OOD detection
        'convnext_base': 1,      # CRITICAL for OOD detection
        'swin_t': 1,             # CRITICAL for OOD detection
        'vit_b_16': 0,           # LayerNorm only, no effect
        'vit_b_32': 0,           # LayerNorm only, no effect
        'deit_base': 0,          # LayerNorm only, no effect
        'mobilenetv3_large': 0,  # Has LayerNorm variants
        'densenet121': 0,
        'densenet169': 0,
        'resnet50': 0,
        'resnet101': 0,
        'efficientnet_v2_s': 0
    }

    # Optimizer type selection
    # Transformers (ViT, DeiT, Swin) and modern CNNs (ConvNeXt) use AdamW
    USE_ADAMW_MODELS = {'convnext_tiny', 'convnext_base', 'swin_t', 'vit_b_16', 'vit_b_32', 'deit_base'}

    # Batch size adjustments (for memory-constrained training)
    BATCH_SIZE_OVERRIDES = {
        'convnext_tiny': 16,      # Requires smaller batch + accumulation
        'convnext_base': 8,       # Larger model, needs even smaller batch
        'swin_t': 16,             # Requires smaller batch + accumulation
        'vit_b_16': 16,           # ViT requires smaller batch + accumulation
        'vit_b_32': 16,           # ViT requires smaller batch + accumulation
        'deit_base': 16,          # DeiT requires smaller batch + accumulation
        'resnet101': 16,          # Deeper model, use smaller batch
        'mobilenetv3_large': 64   # Lightweight, can use larger batch
    }

    # Accumulation steps for effective batch size
    ACCUMULATION_STEPS = {
        'convnext_tiny': 6,
        'convnext_base': 8,       # Larger model needs more accumulation
        'swin_t': 6,
        'vit_b_16': 6,
        'vit_b_32': 6,
        'deit_base': 6,
        'resnet101': 4
    }


class ViMConfig:
    """ViM-specific configurations"""

    # PCA subspace
    TARGET_VARIANCE_RATIO = 0.95  # Keep 95% variance
    MIN_SUBSPACE_DIM = 50
    MAX_SUBSPACE_DIM = None  # No limit by default

    # Alpha calibration
    ALPHA_STABILITY_EPS = 1e-8

    # Covariance regularization
    COVARIANCE_EPS = 1e-6  # For numerical stability in eigendecomposition


class OptimizerConfig:
    """Optimizer configurations"""

    # AdamW (for transformers and modern CNNs)
    ADAMW_BETAS = (0.9, 0.999)
    ADAMW_EPS = 1e-8


# Model-specific recommended configs
RECOMMENDED_CONFIGS = {
    'resnet50': {
        'base_lr': 0.001,  # Will be scaled to 0.01 for SGD
        'batch_size': 32,
        'freeze_bn': 0,
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10
    },
    'densenet121': {
        'base_lr': 0.001,
        'batch_size': 32,
        'freeze_bn': 0,
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10
    },
    'densenet169': {
        'base_lr': 0.001,
        'batch_size': 32,
        'freeze_bn': 0,
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10
    },
    'convnext_tiny': {
        'base_lr': 0.001,  # AdamW
        'batch_size': 16,
        'accumulation_steps': 6,
        'freeze_bn': 1,  # CRITICAL
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10
    },
    'swin_t': {
        'base_lr': 0.001,  # AdamW
        'batch_size': 16,
        'accumulation_steps': 6,
        'freeze_bn': 1,  # CRITICAL
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10,
        'warmup_rounds': 10  # Transformers need more warmup
    },
    'vit_b_16': {
        'base_lr': 0.001,  # AdamW
        'batch_size': 16,
        'accumulation_steps': 6,
        'freeze_bn': 0,  # LayerNorm only, no effect
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10,
        'warmup_rounds': 10  # Transformers need more warmup
    },
    'vit_b_32': {
        'base_lr': 0.001,  # AdamW
        'batch_size': 16,
        'accumulation_steps': 6,
        'freeze_bn': 0,  # LayerNorm only, no effect
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10,
        'warmup_rounds': 10  # Transformers need more warmup
    },
    'efficientnet_v2_s': {
        'base_lr': 0.001,  # AdamW
        'batch_size': 32,
        'freeze_bn': 0,
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10
    },
    # ==================== 新增模型配置 (方案B) ====================
    'mobilenetv3_large': {
        'base_lr': 0.001,  # SGD
        'batch_size': 64,  # Lightweight, larger batch
        'freeze_bn': 0,
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10
    },
    'convnext_base': {
        'base_lr': 0.001,  # AdamW
        'batch_size': 8,   # Larger model, smaller batch
        'accumulation_steps': 8,
        'freeze_bn': 1,    # CRITICAL for OOD
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10,
        'warmup_rounds': 10
    },
    'resnet101': {
        'base_lr': 0.001,  # SGD
        'batch_size': 16,  # Deeper model, smaller batch
        'accumulation_steps': 4,
        'freeze_bn': 0,
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10
    },
    'deit_base': {
        'base_lr': 0.001,  # AdamW
        'batch_size': 16,
        'accumulation_steps': 6,
        'freeze_bn': 0,    # LayerNorm only
        'weight_decay': 1e-4,
        'communication_rounds': 50,
        'local_epochs': 4,
        'early_stop_patience': 10,
        'warmup_rounds': 10
    }
}


def get_model_config(model_type):
    """
    Get recommended configuration for a model type

    Args:
        model_type: Model type string (e.g., 'resnet50', 'convnext_tiny')

    Returns:
        dict: Configuration dictionary
    """
    model_type = model_type.lower().replace('-', '_')

    # Start with defaults
    config = {
        'base_lr': TrainingConfig.BASE_LR,
        'batch_size': TrainingConfig.DEFAULT_BATCH_SIZE,
        'freeze_bn': 0,
        'weight_decay': TrainingConfig.WEIGHT_DECAY,
        'communication_rounds': TrainingConfig.DEFAULT_COMMUNICATION_ROUNDS,
        'local_epochs': TrainingConfig.DEFAULT_LOCAL_EPOCHS,
        'early_stop_patience': TrainingConfig.EARLY_STOP_PATIENCE,
        'accumulation_steps': 1
    }

    # Update with model-specific recommendations
    if model_type in RECOMMENDED_CONFIGS:
        config.update(RECOMMENDED_CONFIGS[model_type])

    # Apply BatchNorm freezing defaults
    if model_type in ModelConfig.FREEZE_BN_DEFAULTS:
        config['freeze_bn'] = ModelConfig.FREEZE_BN_DEFAULTS[model_type]

    return config


def should_use_adamw(model_type):
    """Check if model should use AdamW optimizer"""
    return model_type.lower() in ModelConfig.USE_ADAMW_MODELS


def get_feature_dim(model_type):
    """Get feature dimension for a model type"""
    model_type = model_type.lower().replace('-', '_')
    return ModelConfig.FEATURE_DIMS.get(model_type, None)


if __name__ == "__main__":
    # Test configuration retrieval
    for model in ['resnet50', 'densenet121', 'convnext_tiny', 'swin_t', 'efficientnet_v2_s']:
        config = get_model_config(model)
        print(f"\n{model.upper()} Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
