#!/usr/bin/env python3
"""Independent thesis-oriented FOOGD training entrypoint."""

from __future__ import annotations

import argparse

import torch

from config import TrainingConfig, get_model_config
from data_utils import get_split_manifest_path
from early_stopping import EarlyStoppingMonitor
from methods.foster.foster_utils import (
    evaluate_accuracy,
    save_json,
    set_seed,
    setup_experiment_dir,
)
from methods.foogd import (
    FOOGDClient,
    FOOGDScoreModel,
    FOOGDServer,
    create_foogd_federated_loaders,
    get_foogd_defaults,
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


def build_checkpoint_payload(
    model: torch.nn.Module,
    score_model: torch.nn.Module,
    round_idx: int,
    config: dict,
    training_history: dict,
) -> dict:
    return {
        "global_model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "score_model_state_dict": {key: value.detach().cpu().clone() for key, value in score_model.state_dict().items()},
        "round": int(round_idx),
        "config": dict(config),
        "training_history": dict(training_history),
    }


def parse_args() -> argparse.Namespace:
    defaults = get_foogd_defaults()
    parser = argparse.ArgumentParser(description="Train the thesis-oriented independent FOOGD baseline.")
    parser.add_argument("--data_root", type=str, default="./Plankton_OOD_Dataset")
    parser.add_argument("--model_type", type=str, required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--n_clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--communication_rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments/foogd_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--freeze_bn", action="store_true")
    parser.add_argument("--fourier_mix_alpha", type=float, default=defaults["fourier_mix_alpha"])
    parser.add_argument("--lambda_sag", type=float, default=defaults["lambda_sag"])
    parser.add_argument("--lambda_sm3d", type=float, default=defaults["lambda_sm3d"])
    parser.add_argument("--score_learning_rate", type=float, default=defaults["score_learning_rate"])
    parser.add_argument("--score_momentum", type=float, default=defaults["score_momentum"])
    parser.add_argument("--score_weight_decay", type=float, default=defaults["score_weight_decay"])
    parser.add_argument("--score_hidden_dim", type=int, default=defaults["score_hidden_dim"])
    parser.add_argument("--sample_steps", type=int, default=defaults["sample_steps"])
    parser.add_argument("--sample_eps", type=float, default=defaults["sample_eps"])
    parser.add_argument("--sigma_begin", type=float, default=defaults["sigma_begin"])
    parser.add_argument("--sigma_end", type=float, default=defaults["sigma_end"])
    parser.add_argument("--anneal_power", type=float, default=defaults["anneal_power"])
    parser.add_argument("--noise_type", type=str, default=defaults["noise_type"])
    parser.add_argument("--loss_type", type=str, default=defaults["loss_type"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = get_model_config(args.model_type)
    if args.num_workers is None:
        from data_utils import get_recommended_num_workers

        args.num_workers = get_recommended_num_workers()

    experiment_dir = setup_experiment_dir(args.output_dir, args.model_type)
    config_payload = {
        **vars(args),
        "thesis_title": THESIS_TITLE,
        "method": "FOOGD",
        "base_lr": model_cfg.get("base_lr", TrainingConfig.BASE_LR) * TrainingConfig.SGD_LR_MULTIPLIER,
        "weight_decay": model_cfg.get("weight_decay", TrainingConfig.WEIGHT_DECAY),
        "momentum": 0.9,
        "num_classes": 54,
        "result_scope": "supplemental_foogd_baseline",
        "notes": "Thesis-oriented FOOGD adaptation with official score-model and Fourier-pair logic.",
        "evaluation_score_default": "sm",
        "split_manifest": get_split_manifest_path(args.n_clients, args.alpha, args.seed),
    }
    save_json(experiment_dir / "config.json", config_payload)

    print("=" * 72)
    print("Independent FOOGD Baseline Training".center(72))
    print("=" * 72)
    print(f"[Device] {device}")
    print(f"[Output] {experiment_dir}")
    print(f"[Split] {config_payload['split_manifest']}")

    global_model = create_model(model_type=args.model_type, num_classes=54).to(device)
    feature_dim = int(global_model.feature_dim)
    global_score_model = FOOGDScoreModel(feature_dim=feature_dim, hidden_dim=args.score_hidden_dim).to(device)

    client_loaders, _, _, _, val_loader = create_foogd_federated_loaders(
        data_root=args.data_root,
        n_clients=args.n_clients,
        alpha=args.alpha,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        partition_seed=args.seed,
        fourier_mix_alpha=args.fourier_mix_alpha,
    )

    initial_model_state = {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}
    initial_score_state = {key: value.detach().cpu().clone() for key, value in global_score_model.state_dict().items()}

    clients = []
    for client_id, loader in enumerate(client_loaders):
        client_model = create_model(model_type=args.model_type, num_classes=54).to(device)
        client_model.load_state_dict({k: v.to(device) for k, v in initial_model_state.items()}, strict=True)
        client_score_model = FOOGDScoreModel(feature_dim=feature_dim, hidden_dim=args.score_hidden_dim).to(device)
        client_score_model.load_state_dict({k: v.to(device) for k, v in initial_score_state.items()}, strict=True)
        clients.append(
            FOOGDClient(
                client_id=client_id,
                model=client_model,
                score_model=client_score_model,
                train_loader=loader,
                device=device,
                base_lr=config_payload["base_lr"],
                weight_decay=config_payload["weight_decay"],
                momentum=config_payload["momentum"],
                num_classes=54,
                lambda_sag=args.lambda_sag,
                lambda_sm3d=args.lambda_sm3d,
                score_learning_rate=args.score_learning_rate,
                score_momentum=args.score_momentum,
                score_weight_decay=args.score_weight_decay,
                sample_steps=args.sample_steps,
                sample_eps=args.sample_eps,
                sigma_begin=args.sigma_begin,
                sigma_end=args.sigma_end,
                anneal_power=args.anneal_power,
                noise_type=args.noise_type,
                loss_type=args.loss_type,
                freeze_bn=args.freeze_bn,
            )
        )
        print(f"[Client {client_id}] samples={clients[-1].sample_count}")

    server = FOOGDServer(global_model=global_model, global_score_model=global_score_model, device=device)
    early_stopping = EarlyStoppingMonitor(
        patience=model_cfg.get("early_stop_patience", TrainingConfig.EARLY_STOP_PATIENCE),
        min_delta=TrainingConfig.EARLY_STOP_MIN_DELTA,
        save_dir=str(experiment_dir),
    )

    training_history = {
        "rounds": [],
        "backbone_loss": [],
        "score_loss": [],
        "ce_loss": [],
        "ksd_loss": [],
        "dsm_loss": [],
        "mmd_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "best_val_acc": 0.0,
        "best_round": 0,
        "method": "FOOGD",
    }

    global_model_state, global_score_state = server.global_states()
    best_model_path = experiment_dir / "best_model.pth"
    final_model_path = experiment_dir / "final_model.pth"

    for round_idx in range(args.communication_rounds):
        print(f"\n[Round {round_idx + 1}/{args.communication_rounds}]")
        server.load_global_states(global_model_state, global_score_state)

        model_updates = []
        score_updates = []
        sample_sizes = []
        metric_keys = ["backbone_loss", "score_loss", "ce_loss", "ksd_loss", "dsm_loss", "mmd_loss", "train_accuracy"]
        round_metrics = {key: [] for key in metric_keys}

        for client in clients:
            client.load_global_state(global_model_state, global_score_state)
            updated_model_state, updated_score_state, metrics = client.train_round(
                current_round=round_idx,
                total_rounds=args.communication_rounds,
                local_epochs=args.local_epochs,
                warmup_rounds=TrainingConfig.WARMUP_ROUNDS,
                min_lr_factor=get_foogd_defaults()["min_lr_factor"],
            )
            model_updates.append(updated_model_state)
            score_updates.append(updated_score_state)
            sample_sizes.append(client.sample_count)
            for key in metric_keys:
                round_metrics[key].append(metrics[key])
            print(
                f"  [Client {client.client_id}] backbone={metrics['backbone_loss']:.4f} "
                f"score={metrics['score_loss']:.4f} acc={metrics['train_accuracy']:.4f}"
            )

        global_model_state, global_score_state = server.aggregate(model_updates, score_updates, sample_sizes)
        server.load_global_states(global_model_state, global_score_state)
        val_acc = evaluate_accuracy(server.global_model, val_loader, device)
        print(f"  [Validation] acc={val_acc:.4f}")

        training_history["rounds"].append(round_idx + 1)
        for key in metric_keys:
            training_history[key].append(sum(round_metrics[key]) / max(1, len(round_metrics[key])))
        training_history["val_accuracy"].append(val_acc)

        should_stop, _ = early_stopping.check(round_idx + 1, val_acc, checkpoint_path=str(best_model_path))
        if val_acc > training_history["best_val_acc"]:
            training_history["best_val_acc"] = val_acc
            training_history["best_round"] = round_idx + 1
            best_payload = build_checkpoint_payload(
                server.global_model,
                server.global_score_model,
                round_idx + 1,
                config_payload,
                training_history,
            )
            torch.save(best_payload, best_model_path)
            print(f"  [Checkpoint] saved best model -> {best_model_path}")
        if should_stop:
            break

    final_payload = build_checkpoint_payload(
        server.global_model,
        server.global_score_model,
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
