#!/usr/bin/env python3
"""Automate the remaining FOSTER follow-up steps.

This daemon waits for:
1. local `resnet50` evaluation to finish, then launches local `mobilenetv3_large`
   training followed by evaluation on GPU1;
2. remote `efficientnet_v2_s` and `resnet101` training to finish, then runs their
   remote evaluations and syncs the resulting experiment folders back locally.
"""

from __future__ import annotations

import glob
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/dell7960/桌面/FedOOD")
REMOTE = "10.4.47.203"
REMOTE_ROOT = "/home/dell7960/桌面/FedOOD"
POLL_SECONDS = 60


def run_local(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def run_remote(shell_cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_local(["ssh", REMOTE, shell_cmd], check=check)


def local_train_running(model_type: str) -> bool:
    result = run_local(["bash", "-lc", f"pgrep -f 'train_foster.py --model_type {model_type}' >/dev/null"])
    return result.returncode == 0


def remote_train_running(model_type: str) -> bool:
    result = run_remote(f"pgrep -f 'train_foster.py --model_type {model_type}' >/dev/null", check=False)
    return result.returncode == 0


def latest_local_experiment(model_type: str) -> Path | None:
    candidates = sorted(
        glob.glob(str(ROOT / "experiments" / "foster_v1" / model_type / "experiment_*")),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    return Path(candidates[0]) if candidates else None


def latest_remote_experiment(model_type: str) -> str | None:
    cmd = (
        f"cd {REMOTE_ROOT} && "
        f"ls -td experiments/foster_v1/{model_type}/experiment_* 2>/dev/null | head -1"
    )
    result = run_remote(cmd, check=False)
    latest = result.stdout.strip()
    return latest or None


def file_exists_remote(path: str) -> bool:
    result = run_remote(f"test -f '{path}'", check=False)
    return result.returncode == 0


def sync_remote_experiment(model_type: str, experiment_rel: str) -> None:
    local_parent = ROOT / "experiments" / "foster_v1" / model_type
    local_parent.mkdir(parents=True, exist_ok=True)
    run_local(
        [
            "rsync",
            "-az",
            f"{REMOTE}:{REMOTE_ROOT}/{experiment_rel}/",
            str(local_parent / Path(experiment_rel).name),
        ]
    )


def evaluate_remote(model_type: str, gpu: int) -> Path:
    latest = latest_remote_experiment(model_type)
    if latest is None:
        raise RuntimeError(f"No remote experiment directory found for {model_type}")
    result_path = f"{REMOTE_ROOT}/{latest}/foster_results.json"
    if not file_exists_remote(result_path):
        eval_cmd = (
            f"cd {REMOTE_ROOT} && "
            f"CUDA_VISIBLE_DEVICES={gpu} python3 evaluate_foster.py "
            f"--checkpoint '{latest}/best_model.pth' "
            f"--data_root ./Plankton_OOD_Dataset "
            f"--device cuda:0 --evaluation_score msp"
        )
        print(f"[remote-eval] {model_type}: {latest}", flush=True)
        run_remote(eval_cmd)
    sync_remote_experiment(model_type, latest)
    return ROOT / latest / "foster_results.json"


def train_and_eval_local_mobilenet() -> Path:
    latest = latest_local_experiment("mobilenetv3_large")
    if latest and (latest / "foster_results.json").exists():
        return latest / "foster_results.json"

    train_cmd = [
        "bash",
        "-lc",
        (
            "cd /home/dell7960/桌面/FedOOD && "
            "CUDA_VISIBLE_DEVICES=1 python3 train_foster.py "
            "--model_type mobilenetv3_large --batch_size 32 "
            "--data_root ./Plankton_OOD_Dataset "
            "--device cuda:0 --output_dir experiments/foster_v1"
        ),
    ]
    print("[local-train] mobilenetv3_large", flush=True)
    subprocess.run(train_cmd, check=True)

    latest = latest_local_experiment("mobilenetv3_large")
    if latest is None:
        raise RuntimeError("mobilenetv3_large experiment directory missing after training")
    eval_cmd = [
        "bash",
        "-lc",
        (
            f"cd /home/dell7960/桌面/FedOOD && "
            f"CUDA_VISIBLE_DEVICES=1 python3 evaluate_foster.py "
            f"--checkpoint '{latest}/best_model.pth' "
            f"--data_root ./Plankton_OOD_Dataset "
            f"--device cuda:0 --evaluation_score msp"
        ),
    ]
    print(f"[local-eval] mobilenetv3_large: {latest}", flush=True)
    subprocess.run(eval_cmd, check=True)
    return latest / "foster_results.json"


def wait_for_local_resnet50_result() -> Path:
    while True:
        latest = latest_local_experiment("resnet50")
        if latest and (latest / "foster_results.json").exists():
            return latest / "foster_results.json"
        print("[wait] local resnet50 foster_results.json", flush=True)
        time.sleep(POLL_SECONDS)


def main() -> None:
    remote_done = {"efficientnet_v2_s": False, "resnet101": False}
    mobilenet_done = False

    while True:
        if not mobilenet_done:
            resnet50_result = latest_local_experiment("resnet50")
            if resnet50_result and (resnet50_result / "foster_results.json").exists():
                train_and_eval_local_mobilenet()
                mobilenet_done = True

        for model_type, gpu in [("efficientnet_v2_s", 1), ("resnet101", 0)]:
            if remote_done[model_type]:
                continue
            if not remote_train_running(model_type):
                latest = latest_remote_experiment(model_type)
                if latest:
                    result_path = f"{REMOTE_ROOT}/{latest}/foster_results.json"
                    if not file_exists_remote(result_path):
                        evaluate_remote(model_type, gpu)
                    else:
                        sync_remote_experiment(model_type, latest)
                    remote_done[model_type] = True

        if all(remote_done.values()) and mobilenet_done:
            print("[done] All queued FOSTER follow-up steps completed.", flush=True)
            return

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
