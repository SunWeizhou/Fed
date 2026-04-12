#!/usr/bin/env python3
"""联邦分类 backbone 训练主脚本。

当前脚本只负责联邦分类训练与 best_model 选择。
FedViM / ACT-FedViM / Pooled-ViM / Pooled-ACT-ViM 均在训练完成后
通过专门评估脚本进行后处理，不在训练阶段累计 OOD 统计量。
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import random  # 确保导入 random

from models import create_model
from data_utils import create_federated_loaders, get_recommended_num_workers, get_split_manifest_path
from client import FLClient
from server import FLServer

# 导入配置和早停模块
from config import TrainingConfig, get_model_config
from early_stopping import EarlyStoppingMonitor

def set_seed(seed):
    """固定所有随机种子，确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")


def setup_experiment(args):
    """设置实验环境"""
    # 如果指定了 resume_from，直接使用该目录
    if hasattr(args, 'resume_from') and args.resume_from:
        experiment_dir = args.resume_from
        print(f"[Resume] Using existing experiment directory: {experiment_dir}")
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
        return experiment_dir

    # 创建输出目录 - 按模型类型分组
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 如果 output_dir 已经是模型专属目录（如 experiments_large_v2/resnet50）
    # 则直接在其下创建 experiment_* 子目录
    # 否则假设 output_dir 是根目录，需要添加模型类型子目录
    if args.model_type and args.model_type in os.path.basename(args.output_dir):
        # output_dir 已经包含模型名（如 experiments_large_v2/resnet50）
        experiment_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    else:
        # output_dir 是根目录，需要添加模型类型子目录
        model_subdir = os.path.join(args.output_dir, args.model_type) if args.model_type else args.output_dir
        experiment_dir = os.path.join(model_subdir, f"experiment_{timestamp}")

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)

    # 保存实验配置
    config_path = os.path.join(experiment_dir, "config.json")
    config_payload = dict(vars(args))
    config_payload["split_manifest"] = args.split_manifest
    with open(config_path, 'w') as f:
        json.dump(config_payload, f, indent=2)

    return experiment_dir


def create_clients(n_clients, model_template, client_loaders, device, model_type='densenet169', freeze_bn=True, base_lr=0.001, use_fedvim=False, weight_decay=1e-5):
    """创建客户端，仅用于联邦分类训练。"""
    clients = []

    for client_id in range(n_clients):
        client_model = create_model(model_type=model_type, num_classes=54)
        client_model.load_state_dict(model_template.state_dict())
        client_model = client_model.to(device)

        client = FLClient(
            client_id=client_id,
            model=client_model,
            train_loader=client_loaders[client_id],
            device=device,
            alpha_loader=None,
            freeze_bn=freeze_bn,
            base_lr=base_lr,
            use_fedvim=use_fedvim,
            weight_decay=weight_decay,
        )
        clients.append(client)

    return clients


def _evaluate_accuracy(model, data_loader, device):
    """
    简单的准确率评估函数 (用于验证集)

    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 计算设备

    Returns:
        accuracy: 准确率 (0-1)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()  # 恢复训练模式
    return correct / total


def federated_training(args):
    """联邦学习训练主函数 (增强版: 集成 Early Stopping 和配置优化)"""
    print("=" * 70)
    print("联邦分类 Backbone 训练".center(70))
    print("=" * 70)
    print()

    # 设置设备
    device = torch.device(args.device)
    print(f"[设备] 使用设备: {device}")
    if args.use_fedvim:
        print("[说明] --use_fedvim 仅作为实验配置记录保留；训练阶段不计算任何 OOD 统计量。")

    # 设置实验
    experiment_dir = setup_experiment(args)
    print(f"[输出] 实验目录: {experiment_dir}")
    print(f"[分片] 固定分片文件: {args.split_manifest}")

    # 获取模型推荐配置
    model_config = get_model_config(args.model_type)
    print(f"\n[模型配置] {args.model_type.upper()} 推荐参数:")
    for key, value in model_config.items():
        print(f"  - {key}: {value}")

    # 创建数据加载器
    print(f"\n[数据加载] 创建数据加载器...")
    client_loaders, _, _, _, val_loader = create_federated_loaders(
        data_root=args.data_root,
        n_clients=args.n_clients,
        alpha=args.alpha,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        partition_seed=args.seed,
    )
    print(f"[数据加载] 完成 - 训练客户端: {len(client_loaders)}, 验证集: OK")
    print("[数据加载] 说明 - 客户端训练分片仅用于分类训练；OOD 统计量将在后处理评估脚本中重新计算")

    # 创建全局模型
    print(f"\n[模型] 创建全局模型: {args.model_type.upper()}")
    global_model = create_model(model_type=args.model_type, num_classes=54)

    # 将模型移动到设备
    global_model = global_model.to(device)

    # 创建服务端
    server = FLServer(global_model, device)
    print(f"[模型] 模型已创建并移动到 {device}")

    # -------------------------------------------------
    # 1. 增强的检查点加载逻辑 (断点续训/恢复)
    # -------------------------------------------------
    start_round = 0
    best_acc = 0.0

    # 初始化 Early Stopping Monitor
    early_stopping = EarlyStoppingMonitor(
        patience=model_config.get('early_stop_patience', TrainingConfig.EARLY_STOP_PATIENCE),
        min_delta=TrainingConfig.EARLY_STOP_MIN_DELTA,
        save_dir=experiment_dir
    )

    # 尝试自动寻找最新的检查点 (防止意外中断)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        # 如果指定了 --resume_from 或 --resume，则恢复训练
        if checkpoints and (args.resume or hasattr(args, 'resume_from') and args.resume_from):
            latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"\n[恢复训练] 发现检查点: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, weights_only=False)

            server.global_model.load_state_dict(ckpt['global_model_state_dict'])

            # 恢复训练进度
            start_round = ckpt['round']
            training_history = ckpt['training_history']
            best_acc = training_history.get('best_acc', 0.0) # 如果历史里没存，就默认0

            # 恢复 Early Stopping 状态（如果存在）
            if 'early_stopping' in ckpt:
                early_stopping.history = ckpt['early_stopping'].get('history', early_stopping.history)
                early_stopping.best_val_acc = ckpt['early_stopping'].get('best_val_acc', 0.0)
                early_stopping.best_round = ckpt['early_stopping'].get('best_round', 0)
                early_stopping.wait = ckpt['early_stopping'].get('wait', 0)
                print(f"[恢复训练] Early Stopping 状态已恢复 (best_val_acc={early_stopping.best_val_acc:.4f})")

            # 注意：Client 的状态将在创建 Client 后恢复
            saved_client_states = ckpt.get('client_states', None)
        else:
            saved_client_states = None
    else:
        saved_client_states = None

    # =================================================================
    # 【优化】使用配置模块设置训练参数
    # =================================================================
    # 从配置获取推荐参数（优先使用配置，其次使用命令行参数）
    weight_decay = model_config.get('weight_decay', TrainingConfig.WEIGHT_DECAY)
    base_lr = model_config.get('base_lr', TrainingConfig.BASE_LR)

    print(f"\n[优化器配置]")
    print(f"  - 基础学习率: {base_lr}")
    print(f"  - 权重衰减: {weight_decay}")
    print(f"  - 冻结 BatchNorm: {args.freeze_bn}")

    # 创建客户端
    print(f"\n[客户端] 创建 {args.n_clients} 个联邦学习客户端...")
    clients = create_clients(
        args.n_clients, global_model, client_loaders, device, args.model_type,
        freeze_bn=args.freeze_bn,
        base_lr=base_lr,
        use_fedvim=args.use_fedvim,
        weight_decay=weight_decay,
    )
    print(f"[客户端] 客户端创建完成")

    if saved_client_states is not None:
        for i, client in enumerate(clients):
            if i in saved_client_states:
                client.model.load_state_dict(saved_client_states[i], strict=False)

    # 如果没有恢复历史，初始化历史记录
    if 'training_history' not in locals():
        training_history = {
            'rounds': [],
            'train_losses': [],
            'val_accuracies': [],      # 验证集准确率 (用于模型选择)
            'best_acc': 0.0
        }

    # 联邦学习训练循环
    print(f"\n{'='*70}")
    print(f"开始联邦学习训练".center(70))
    print(f"{'='*70}")
    print(f"配置: {args.communication_rounds} 轮 × {args.local_epochs} 本地轮次")
    print(f"Early Stopping: 耐心值={early_stopping.patience}, 监控指标=验证集准确率")
    print(f"{'='*70}\n")

    for round_num in range(start_round, args.communication_rounds):
        print(f"\n=== 通信轮次 {round_num + 1}/{args.communication_rounds} ===")

        # 选择参与本轮训练的客户端
        if args.client_fraction < 1.0:
            n_selected = max(1, int(args.n_clients * args.client_fraction))
            selected_clients = np.random.choice(
                args.n_clients, n_selected, replace=False
            )
        else:
            selected_clients = range(args.n_clients)

        print(f"选择的客户端: {list(selected_clients)}")

        # 客户端本地训练
        client_updates = []
        client_sample_sizes = []
        round_train_loss = 0.0

        for client_id in selected_clients:
            client = clients[client_id]

            # 设置客户端模型参数
            client.set_generic_parameters(server.get_global_parameters())

            client_update, client_loss, _ = client.train_step(
                local_epochs=args.local_epochs,
                current_round=round_num,  # 当前轮次
                global_stats=None,
                total_rounds=args.communication_rounds,  # [新增] 传入总轮数参数
                warmup_rounds=args.warmup_rounds,  # [新增] 传入 warmup 轮数
                accumulation_steps=args.accumulation_steps  # [新增] 传入梯度累积步数
            )

            client_updates.append(client_update)
            client_sample_sizes.append(len(client.train_loader.dataset))
            round_train_loss += client_loss
            print(f"  客户端 {client_id}: 本地损失 = {client_loss:.4f}")

        # 服务器聚合
        aggregated_params = server.aggregate(client_updates, client_sample_sizes)
        server.set_global_parameters(aggregated_params)

        # 更新客户端模型 (为下一轮做准备)
        for client in clients:
            client.set_generic_parameters(server.get_global_parameters())

        # === 评估阶段：仅使用验证集做模型选择 ===
        print(f"\n{'='*70}")
        print(f"第 {round_num + 1} 轮评估".center(70))
        print(f"{'='*70}")

        print(f"[验证集] 评估中... (用于 Early Stopping 和模型选择)")
        val_acc = _evaluate_accuracy(server.global_model, val_loader, device)
        print(f"  验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"[本轮结果] 验证集准确率: {val_acc:.4f}")

        # =========================================================
        # Early Stopping 检查
        # =========================================================
        checkpoint_path = os.path.join(experiment_dir, "checkpoints", f"round_{round_num + 1}.pth")
        should_stop, stop_reason = early_stopping.check(
            round_num=round_num + 1,
            val_acc=val_acc,
            checkpoint_path=checkpoint_path
        )

        if should_stop:
            print(f"\n{'='*70}")
            print(f"Early Stopping 触发！训练提前终止。".center(70))
            print(f"{'='*70}")
            print(f"原因: {stop_reason}")
            print(f"\n正在恢复最佳模型...")

            best_ckpt_path = early_stopping.get_best_checkpoint()
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                ckpt = torch.load(best_ckpt_path, weights_only=False, map_location=device)
                server.global_model.load_state_dict(ckpt['global_model_state_dict'])
                print(f"已恢复最佳模型: {best_ckpt_path}")
                print(f"最佳验证集准确率: {early_stopping.best_val_acc:.4f}")

                training_history['early_stopped'] = True
                training_history['early_stop_round'] = round_num + 1
                training_history['early_stop_reason'] = stop_reason
                break
            else:
                print(f"[警告] 未找到最佳检查点: {best_ckpt_path}")
                print(f"继续训练...")

        # 3. 记录日志
        training_history['rounds'].append(round_num + 1)
        training_history['train_losses'].append(round_train_loss / len(selected_clients))
        training_history['val_accuracies'].append(val_acc)

        # 4. 保存 Best Model (基于验证集准确率)
        if val_acc > best_acc:
            best_acc = val_acc
            training_history['best_acc'] = best_acc

            client_states = {i: client.model.state_dict() for i, client in enumerate(clients)}

            torch.save({
                'round': round_num + 1,
                'global_model_state_dict': server.global_model.state_dict(),
                'client_states': client_states,
                'vim_stats': None,
                'best_acc': best_acc,
                'config': vars(args)
            }, os.path.join(experiment_dir, "best_model.pth"))
            print(f"  [Best] 新最优模型 (Val Acc: {best_acc:.4f}) 已保存")

        # 5. 保存定期 Checkpoint
        if (round_num + 1) % args.save_frequency == 0:
            client_states = {i: client.model.state_dict() for i, client in enumerate(clients)}

            torch.save({
                'round': round_num + 1,
                'global_model_state_dict': server.global_model.state_dict(),
                'client_states': client_states,
                'vim_stats': None,
                'training_history': training_history,
                'early_stopping': {
                    'history': early_stopping.history,
                    'best_val_acc': early_stopping.best_val_acc,
                    'best_round': early_stopping.best_round,
                    'wait': early_stopping.wait
                },
                'config': vars(args)
            }, os.path.join(experiment_dir, "checkpoints", f"round_{round_num + 1}.pth"))
            print(f"[检查点] 已保存: round_{round_num + 1}.pth")

    # 训练完成
    print(f"\n{'='*70}")
    print(f"联邦学习训练完成！".center(70))
    print(f"{'='*70}")

    # 保存 Early Stopping 历史和摘要
    early_stopping.save_history()
    early_stop_summary = early_stopping.get_summary()

    # 打印 Early Stopping 摘要
    print(f"\n[Early Stopping 摘要]")
    print(f"  最佳轮次: {early_stop_summary['best_round']}")
    print(f"  最佳验证集准确率: {early_stop_summary['best_val_acc']:.4f}")
    if early_stop_summary['stopped_epoch']:
        print(f"  停止轮次: {early_stop_summary['stopped_epoch']}")
        print(f"  停止原因: {early_stop_summary['reason']}")
    else:
        print(f"  完成所有计划的训练轮次")

    # 保存最终模型 (包含所有客户端状态)
    final_model_path = os.path.join(experiment_dir, "final_model.pth")
    client_states = {i: client.model.state_dict() for i, client in enumerate(clients)}

    torch.save({
        'round': args.communication_rounds,
        'global_model_state_dict': server.global_model.state_dict(),
        'client_states': client_states, # [关键] 保存客户端状态
        'vim_stats': None,
        'training_history': training_history,
        'early_stopping_summary': early_stop_summary,  # [新增] 保存 Early Stopping 摘要
        'config': vars(args)
    }, final_model_path)
    print(f"[保存] 最终模型: {final_model_path}")

    # 保存训练历史
    history_path = os.path.join(experiment_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"[保存] 训练历史: {history_path}")

    # 绘制训练曲线
    plot_training_curves(training_history, experiment_dir)
    print("\n" + "=" * 60)
    print("联邦分类训练流程完成")
    print("=" * 60)
    print(f"  - 实验目录: {experiment_dir}")
    print("  - 下一步: 固定 best_model 后使用 evaluate_fedvim.py / advanced_fedvim.py / evaluate_pooled_vim.py / evaluate_pooled_act_vim.py / evaluate_baselines.py 进行后处理评估")
    print("=" * 60 + "\n")

    return training_history


def plot_training_curves(history, output_dir):
    """绘制训练曲线"""
    if not history['rounds']:
        return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['rounds'], history['train_losses'], 'b-', label='Training Loss')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['rounds'], history['val_accuracies'], 'g-', label='Validation Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"训练曲线图已保存: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='FedViM 联邦训练脚本（仅训练，不做 OOD 评估）')

    # 数据参数
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset',
                       help='数据集根目录路径')
    parser.add_argument('--n_clients', type=int, default=5,
                       help='客户端数量')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='狄利克雷分布参数（控制数据异质性）')

    # 训练参数
    parser.add_argument('--communication_rounds', type=int, default=None,
                       help='通信轮次（默认根据模型自动设置，通常为50）')
    parser.add_argument('--local_epochs', type=int, default=4,
                       help='每个客户端的本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--client_fraction', type=float, default=1.0,
                       help='每轮选择的客户端比例')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='densenet169',
                       choices=['densenet169', 'resnet50', 'resnet101', 'mobilenetv3_large', 'efficientnet_v2_s'],
                       help='五模型主线骨干网络类型')
    parser.add_argument('--use_fedvim', action='store_true', default=False,
                       help='保留兼容标记；训练阶段仅训练分类 backbone，FedViM/ACT-FedViM 在后处理脚本中评估')
    parser.add_argument('--image_size', type=int, default=320,
                       help='输入图像尺寸')

    # 保存
    parser.add_argument('--save_frequency', type=int, default=10,
                       help='保存检查点频率（轮次）。')

    # 系统参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')
    parser.add_argument('--output_dir', type=str, default='./experiments/experiments_rerun_v1',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')  # [新增]
    parser.add_argument('--resume', action='store_true', default=False,
                       help='是否从最新检查点恢复训练')  # [新增]
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从指定实验目录恢复训练（直接使用该目录）')  # [新增]
    parser.add_argument('--freeze_bn', type=int, default=None,
                       help='是否冻结BN统计量 (1=True, 0=False)。默认根据模型自动设置')
    parser.add_argument('--base_lr', type=float, default=None,
                       help='基础学习率（默认根据模型自动设置）')
    parser.add_argument('--accumulation_steps', type=int, default=None,
                       help='梯度累积步数（默认根据模型自动设置）')
    parser.add_argument('--warmup_rounds', type=int, default=None,
                       help='学习率预热轮数（默认根据模型自动设置）')
    parser.add_argument('--num_workers', type=int, default=TrainingConfig.DEFAULT_NUM_WORKERS,
                       help='DataLoader worker 数。默认使用论文正式设定。')

    args = parser.parse_args()
    args.split_manifest = get_split_manifest_path(args.n_clients, args.alpha, args.seed)

    # [优化] 根据模型类型自动设置默认参数（使用 config.py）
    model_config = get_model_config(args.model_type)

    if args.freeze_bn is None:
        args.freeze_bn = model_config.get('freeze_bn', 0)
        print(f"[自动配置] freeze_bn 设置为 {args.freeze_bn} (基于模型 {args.model_type})")

    if args.base_lr is None:
        args.base_lr = model_config.get('base_lr', TrainingConfig.BASE_LR)
        print(f"[自动配置] base_lr 设置为 {args.base_lr} (基于模型 {args.model_type})")

    if args.accumulation_steps is None:
        args.accumulation_steps = model_config.get('accumulation_steps', 1)
        if args.accumulation_steps > 1:
            print(f"[自动配置] accumulation_steps 设置为 {args.accumulation_steps} (基于模型 {args.model_type})")

    if args.num_workers is None:
        args.num_workers = get_recommended_num_workers(max_workers=TrainingConfig.DEFAULT_NUM_WORKERS)
        print(f"[自动配置] num_workers 设置为 {args.num_workers} (论文正式设定)")

    if args.warmup_rounds is None:
        args.warmup_rounds = model_config.get('warmup_rounds', TrainingConfig.WARMUP_ROUNDS)
        print(f"[自动配置] warmup_rounds 设置为 {args.warmup_rounds} (基于模型 {args.model_type})")

    if args.communication_rounds is None:
        args.communication_rounds = model_config.get('communication_rounds', TrainingConfig.DEFAULT_COMMUNICATION_ROUNDS)
        print(f"[自动配置] communication_rounds 设置为 {args.communication_rounds}")

    # [新增] 在一切开始之前，先设置种子
    set_seed(args.seed)

    # 打印配置
    print("联邦学习训练配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # 开始训练
    training_history = federated_training(args)

    # 打印最终结果
    print("\n=== 训练完成 ===")

    # 确保 training_history 存在且包含必要数据
    if not training_history or 'val_accuracies' not in training_history:
        print("警告: 训练历史数据不完整")
        return

    if training_history['val_accuracies']:
        final_acc = training_history['val_accuracies'][-1]
        print(f"最终验证集准确率: {final_acc:.4f}")
    else:
        print("警告: 未找到验证集准确率数据")
        final_acc = 0.0

    # 数据点统计
    print(f"\n数据点统计:")
    print(f"  总通信轮次: {len(training_history['rounds']) if 'rounds' in training_history else 0}")
    print(f"  保存频率: 每 {args.save_frequency} 轮保存一次检查点")
    print(f"  曲线数据点: {len(training_history['val_accuracies']) if training_history['val_accuracies'] else 0} 个")


if __name__ == "__main__":
    # [新增] 开启 cudnn benchmark
    torch.backends.cudnn.benchmark = True

    main()
