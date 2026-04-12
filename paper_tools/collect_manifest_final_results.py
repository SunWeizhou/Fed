#!/usr/bin/env python3
"""Collect final manifest-v1 results for thesis comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


DEFAULT_MODELS = [
    "mobilenetv3_large",
    "resnet50",
    "resnet101",
    "densenet169",
    "efficientnet_v2_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect manifest final thesis comparison results.")
    parser.add_argument("--main-root", type=str, default="experiments/fedavg_manifest_final_v1")
    parser.add_argument("--foster-root", type=str, default="experiments/foster_manifest_final_v1")
    parser.add_argument("--fedln-root", type=str, default="experiments/fedln_manifest_final_v1")
    parser.add_argument("--output-prefix", type=str, default="paper_tools/manifest_final_v1")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_experiment(root: Path, model_name: str) -> Path:
    model_root = root / model_name
    candidates = sorted(model_root.glob("experiment_*"))
    if not candidates:
        raise FileNotFoundError(f"No experiment directory found under {model_root}")
    return candidates[-1]


def require_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return load_json(path)


def collect_model_record(main_root: Path, foster_root: Path, fedln_root: Path, model_name: str) -> dict:
    main_exp = latest_experiment(main_root, model_name)
    foster_exp = latest_experiment(foster_root, model_name)
    fedln_exp = latest_experiment(fedln_root, model_name)

    return {
        "model_name": model_name,
        "main_experiment_dir": str(main_exp.resolve()),
        "foster_experiment_dir": str(foster_exp.resolve()),
        "fedln_experiment_dir": str(fedln_exp.resolve()),
        "FedViM": require_json(main_exp / "fedvim_results.json"),
        "ACT-FedViM": require_json(main_exp / "act_fedvim_results.json"),
        "MSP": require_json(main_exp / "baselines_eval" / "msp_results.json"),
        "Energy": require_json(main_exp / "baselines_eval" / "energy_results.json"),
        "Pooled-ViM": require_json(main_exp / "pooled_vim_results.json"),
        "Pooled-ACT-ViM": require_json(main_exp / "pooled_act_vim_results.json"),
        "FOSTER": require_json(foster_exp / "foster_results.json"),
        "FedLN": require_json(fedln_exp / "fedln_results.json"),
    }


def build_main_vs_baselines(records: list[dict]) -> dict:
    rows = []
    for record in records:
        rows.append(
            {
                "model_name": record["model_name"],
                "fedvim_near_auroc": record["FedViM"]["near_auroc"],
                "fedvim_far_auroc": record["FedViM"]["far_auroc"],
                "act_near_auroc": record["ACT-FedViM"]["near_auroc"],
                "act_far_auroc": record["ACT-FedViM"]["far_auroc"],
                "msp_near_auroc": record["MSP"]["near_auroc"],
                "msp_far_auroc": record["MSP"]["far_auroc"],
                "foster_near_auroc": record["FOSTER"]["near_auroc"],
                "foster_far_auroc": record["FOSTER"]["far_auroc"],
                "fedln_near_auroc": record["FedLN"]["near_auroc"],
                "fedln_far_auroc": record["FedLN"]["far_auroc"],
            }
        )

    averages = {}
    for key in [
        "fedvim_near_auroc",
        "fedvim_far_auroc",
        "act_near_auroc",
        "act_far_auroc",
        "msp_near_auroc",
        "msp_far_auroc",
        "foster_near_auroc",
        "foster_far_auroc",
        "fedln_near_auroc",
        "fedln_far_auroc",
    ]:
        averages[key] = mean(row[key] for row in rows)

    return {"rows": rows, "averages": averages}


def build_pooled_consistency(records: list[dict]) -> dict:
    rows = []
    for record in records:
        fedvim = record["FedViM"]
        pooled = record["Pooled-ViM"]
        act = record["ACT-FedViM"]
        pact = record["Pooled-ACT-ViM"]
        rows.append(
            {
                "model_name": record["model_name"],
                "fedvim_near_auroc": fedvim["near_auroc"],
                "pooled_vim_near_auroc": pooled["near_auroc"],
                "delta_vim_near": pooled["near_auroc"] - fedvim["near_auroc"],
                "fedvim_far_auroc": fedvim["far_auroc"],
                "pooled_vim_far_auroc": pooled["far_auroc"],
                "delta_vim_far": pooled["far_auroc"] - fedvim["far_auroc"],
                "act_near_auroc": act["near_auroc"],
                "pooled_act_near_auroc": pact["near_auroc"],
                "delta_act_near": pact["near_auroc"] - act["near_auroc"],
                "act_far_auroc": act["far_auroc"],
                "pooled_act_far_auroc": pact["far_auroc"],
                "delta_act_far": pact["far_auroc"] - act["far_auroc"],
            }
        )

    averages = {}
    for key in [
        "delta_vim_near",
        "delta_vim_far",
        "delta_act_near",
        "delta_act_far",
    ]:
        averages[key] = mean(row[key] for row in rows)

    return {"rows": rows, "averages": averages}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    main_root = Path(args.main_root).expanduser().resolve()
    foster_root = Path(args.foster_root).expanduser().resolve()
    fedln_root = Path(args.fedln_root).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()

    records = [collect_model_record(main_root, foster_root, fedln_root, model_name) for model_name in args.models]
    main_vs_baselines = build_main_vs_baselines(records)
    pooled_consistency = build_pooled_consistency(records)

    write_json(output_prefix.with_name(f"{output_prefix.name}_records.json"), {"records": records})
    write_json(output_prefix.with_name(f"{output_prefix.name}_main_vs_baselines.json"), main_vs_baselines)
    write_csv(output_prefix.with_name(f"{output_prefix.name}_main_vs_baselines.csv"), main_vs_baselines["rows"])
    write_json(output_prefix.with_name(f"{output_prefix.name}_pooled_consistency.json"), pooled_consistency)
    write_csv(output_prefix.with_name(f"{output_prefix.name}_pooled_consistency.csv"), pooled_consistency["rows"])

    print(f"Collected {len(records)} model records.")
    print(f"Wrote: {output_prefix.with_name(f'{output_prefix.name}_main_vs_baselines.json')}")
    print(f"Wrote: {output_prefix.with_name(f'{output_prefix.name}_pooled_consistency.json')}")


if __name__ == "__main__":
    main()
