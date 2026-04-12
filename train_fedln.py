#!/usr/bin/env python3
"""Independent thesis-oriented FedLN training entrypoint."""

from __future__ import annotations

import argparse

import torch

from config import TrainingConfig, get_model_config
from data_utils import create_federated_loaders, get_recommended_num_workers, get_split_manifest_path
from early_stopping import EarlyStoppingMonitor
from methods.fedln import FedLNClient, FedLNServer
from methods.foster.foster_utils import (
    build_checkpoint_payload,
    evaluate_accuracy,
    save_json,
    set_seed,
    setup_experiment_dir,
)
from models import create_model


THESIS_TITLE = "FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究"
SUPPORTED_MODELS = {
    "densenet169",
    "efficientnet_v2_s",
    "mobilenetv3_large",
    "resnet50",
    "resnet101",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the thesis-oriented independent FedLN baseline.")
    parser.add_argument("--data_root", type=str, default="./Plankton_OOD_Dataset")
    parser.add_argument("--model_type", type=str, required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--n_clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--communication_rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments/fedln_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=TrainingConfig.DEFAULT_NUM_WORKERS)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--freeze_bn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = get_model_config(args.model_type)
    if args.num_workers is None:
        args.num_workers = get_recommended_num_workers(max_workers=TrainingConfig.DEFAULT_NUM_WORKERS)

    experiment_dir = setup_experiment_dir(args.output_dir, args.model_type)
    config_payload = {
        **vars(args),
        "thesis_title": THESIS_TITLE,
        "method": "FedLN",
        "weight_decay": model_cfg.get("weight_decay", TrainingConfig.WEIGHT_DECAY),
        "base_lr": model_cfg.get("base_lr", TrainingConfig.BASE_LR) * TrainingConfig.SGD_LR_MULTIPLIER,
        "momentum": 0.9,
        "result_scope": "supplemental_fedln_baseline",
        "notes": "Thesis-oriented federated LogitNorm baseline under the FedViM protocol.",
        "evaluation_score_default": "msp",
        "split_manifest": get_split_manifest_path(args.n_clients, args.alpha, args.seed),
    }
    save_json(experiment_dir / "config.json", config_payload)

    print("=" * 72)
    print("Independent FedLN Baseline Training".center(72))
    print("=" * 72)
    print(f"[Device] {device}")
    print(f"[Output] {experiment_dir}")
    print(f"[Split] {config_payload['split_manifest']}")

    global_model = create_model(model_type=args.model_type, num_classes=54).to(device)
    base_lr = config_payload["base_lr"]
    weight_decay = config_payload["weight_decay"]

    client_loaders, _, _, _, val_loader = create_federated_loaders(
        data_root=args.data_root,
        n_clients=args.n_clients,
        alpha=args.alpha,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        partition_seed=args.seed,
    )

    clients = []
    initial_state = {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}
    for client_id, loader in enumerate(client_loaders):
        client_model = create_model(model_type=args.model_type, num_classes=54).to(device)
        client_model.load_state_dict({k: v.to(device) for k, v in initial_state.items()}, strict=True)
        clients.append(
            FedLNClient(
                client_id=client_id,
                model=client_model,
                train_loader=loader,
                device=device,
                base_lr=base_lr,
                weight_decay=weight_decay,
                momentum=0.9,
                temperature=args.temperature,
                freeze_bn=args.freeze_bn,
            )
        )
        print(f"[Client {client_id}] samples={clients[-1].sample_count}")

    server = FedLNServer(global_model=global_model, device=device)
    early_stopping = EarlyStoppingMonitor(
        patience=model_cfg.get("early_stop_patience", TrainingConfig.EARLY_STOP_PATIENCE),
        min_delta=TrainingConfig.EARLY_STOP_MIN_DELTA,
        save_dir=str(experiment_dir),
    )
    training_history = {
        "rounds": [],
        "train_loss": [],
        "val_accuracy": [],
        "best_val_acc": 0.0,
        "best_round": 0,
        "method": "FedLN",
        "temperature": args.temperature,
    }

    global_state = server.global_state()
    best_model_path = experiment_dir / "best_model.pth"
    final_model_path = experiment_dir / "final_model.pth"
    for round_idx in range(args.communication_rounds):
        print(f"\n[Round {round_idx + 1}/{args.communication_rounds}]")
        server.load_global_state(global_state)

        updates = []
        sample_sizes = []
        round_losses = []
        for client in clients:
            client.load_global_state(global_state)
            updated_state, metrics = client.train_round(
                current_round=round_idx,
                total_rounds=args.communication_rounds,
                local_epochs=args.local_epochs,
                warmup_rounds=TrainingConfig.WARMUP_ROUNDS,
                min_lr_factor=TrainingConfig.MIN_LR_FACTOR,
            )
            updates.append(updated_state)
            sample_sizes.append(client.sample_count)
            round_losses.append(metrics["loss"])
            print(f"  [Client {client.client_id}] loss={metrics['loss']:.4f}")

        global_state = server.aggregate(updates, sample_sizes)
        server.load_global_state(global_state)
        val_acc = evaluate_accuracy(server.global_model, val_loader, device)
        print(f"  [Validation] acc={val_acc:.4f}")

        training_history["rounds"].append(round_idx + 1)
        training_history["train_loss"].append(sum(round_losses) / len(round_losses))
        training_history["val_accuracy"].append(val_acc)

        should_stop, _ = early_stopping.check(round_idx + 1, val_acc, checkpoint_path=str(best_model_path))
        if val_acc > training_history["best_val_acc"]:
            training_history["best_val_acc"] = val_acc
            training_history["best_round"] = round_idx + 1
            best_payload = build_checkpoint_payload(server.global_model, round_idx + 1, config_payload, training_history)
            torch.save(best_payload, best_model_path)
            print(f"  [Checkpoint] saved best model -> {best_model_path}")
        if should_stop:
            break

    final_payload = build_checkpoint_payload(
        server.global_model,
        training_history["rounds"][-1] if training_history["rounds"] else 0,
        config_payload,
        training_history,
    )
    torch.save(final_payload, final_model_path)
    save_json(experiment_dir / "training_history.json", training_history)
    early_stopping.save_history()

    print("\nTraining complete.")
    print(f"Best model:  {best_model_path}")
    print(f"Final model: {final_model_path}")


if __name__ == "__main__":
    main()
