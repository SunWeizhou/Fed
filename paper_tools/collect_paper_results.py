#!/usr/bin/env python3
"""Collect five-model FedViM paper results into structured artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_common import DEFAULT_FIVE_MODELS


THESIS_TITLE = "FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect five-model FedViM paper results.")
    parser.add_argument(
        "--experiments-root",
        type=str,
        default="experiments/experiments_rerun_v1",
        help="Root directory containing per-model experiment folders.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="paper_tools/rerun_v1_results",
        help="Output prefix for generated JSON/CSV files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_FIVE_MODELS,
        help="Model names to include in the five-model mainline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of paper models to keep after ACT Near-OOD ranking/filtering.",
    )
    parser.add_argument(
        "--max-near-drop",
        type=float,
        default=0.01,
        help="Maximum allowed Near-OOD AUROC drop for ACT relative to FedViM.",
    )
    return parser.parse_args()


def latest_experiment_dir(model_root: Path) -> Path:
    """Pick the latest experiment_* directory for a model."""
    candidates = sorted(model_root.glob("experiment_*"))
    if not candidates:
        raise FileNotFoundError(f"No experiment_* directory found under {model_root}")
    return candidates[-1]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def newest_matching_file(directory: Path, pattern: str) -> Path | None:
    """Return the lexicographically newest file matching a regex."""
    matches = [path for path in directory.iterdir() if path.is_file() and re.fullmatch(pattern, path.name)]
    if not matches:
        return None
    return sorted(matches)[-1]


def load_legacy_baseline_txt(path: Path, method_name: str) -> dict:
    """Parse the legacy two-line baseline txt format."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    near_metrics = ast.literal_eval(lines[0])
    far_metrics = ast.literal_eval(lines[1])
    return {
        "method": method_name,
        "checkpoint": None,
        "model_type": None,
        "image_size": None,
        "num_classes": None,
        "feature_dim": None,
        "id_accuracy": None,
        "near_auroc": float(near_metrics["AUROC"]),
        "near_aupr": float(near_metrics["AUPR"]),
        "near_fpr95": float(near_metrics["FPR95"]),
        "far_auroc": float(far_metrics["AUROC"]),
        "far_aupr": float(far_metrics["AUPR"]),
        "far_fpr95": float(far_metrics["FPR95"]),
        "alpha": None,
        "alpha_source": "legacy_txt",
        "fixed_k": None,
        "act_k": None,
        "compression_rate": None,
        "feature_compression_rate": None,
        "source_file": str(path),
    }


def normalize_fedvim_record(path: Path) -> dict:
    """Normalize either the new or legacy FedViM JSON schema."""
    data = load_json(path)
    if "method" in data and isinstance(data["method"], str):
        record = dict(data)
        record["source_file"] = str(path)
        return record

    method = data.get("method", {})
    model = data.get("model", {})
    perf = data.get("performance", {})
    diagnostics = data.get("diagnostics", {})
    return {
        "method": "FedViM",
        "thesis_title": THESIS_TITLE,
        "checkpoint": data.get("checkpoint"),
        "model_type": model.get("model_type"),
        "image_size": model.get("image_size"),
        "num_classes": model.get("num_classes"),
        "feature_dim": model.get("feature_dim"),
        "id_accuracy": perf.get("id_accuracy"),
        "near_auroc": perf.get("near_auroc"),
        "near_aupr": perf.get("near_aupr"),
        "near_fpr95": perf.get("near_fpr95"),
        "far_auroc": perf.get("far_auroc"),
        "far_aupr": perf.get("far_aupr"),
        "far_fpr95": perf.get("far_fpr95"),
        "alpha": diagnostics.get("alpha"),
        "alpha_source": "legacy_record",
        "fixed_k": method.get("optimal_k"),
        "act_k": None,
        "compression_rate": None,
        "feature_compression_rate": None if not model.get("feature_dim") else 1.0 - (method.get("optimal_k", 0) / model["feature_dim"]),
        "source_file": str(path),
    }


def normalize_act_record(path: Path) -> dict:
    """Normalize either the new or legacy ACT-FedViM JSON schema."""
    data = load_json(path)
    if "method" in data and isinstance(data["method"], str):
        record = dict(data)
        record["source_file"] = str(path)
        return record

    act_config = data.get("act_config", {})
    perf = data.get("performance", {})
    fixed_k = act_config.get("original_k")
    act_k = act_config.get("optimal_k")
    return {
        "method": "ACT-FedViM",
        "thesis_title": THESIS_TITLE,
        "checkpoint": data.get("checkpoint"),
        "model_type": None,
        "image_size": None,
        "num_classes": None,
        "feature_dim": act_config.get("feature_dim"),
        "id_accuracy": perf.get("id_accuracy"),
        "near_auroc": perf.get("near_auroc"),
        "near_aupr": perf.get("near_aupr"),
        "near_fpr95": perf.get("near_fpr95"),
        "far_auroc": perf.get("far_auroc"),
        "far_aupr": perf.get("far_aupr"),
        "far_fpr95": perf.get("far_fpr95"),
        "alpha": act_config.get("alpha"),
        "alpha_source": act_config.get("alpha_source", "legacy_record"),
        "fixed_k": fixed_k,
        "act_k": act_k,
        "compression_rate": None if not fixed_k else 1.0 - (act_k / fixed_k),
        "feature_compression_rate": None if not act_config.get("feature_dim") else 1.0 - (act_k / act_config["feature_dim"]),
        "n_train_samples": act_config.get("n_samples_train"),
        "act_threshold_s": act_config.get("threshold_s"),
        "act_rho": act_config.get("rho"),
        "source_file": str(path),
    }


def normalize_baseline_record(path: Path, method_key: str) -> dict:
    """Normalize baseline JSON or txt records."""
    if path.suffix == ".json":
        record = load_json(path)
        record["source_file"] = str(path)
        return record
    return load_legacy_baseline_txt(path, method_name=method_key.upper() if method_key == "msp" else "Energy")


def method_record_for_experiment(experiment_dir: Path, method_key: str) -> dict:
    """Load the preferred record for one method within an experiment dir."""
    if method_key == "FedViM":
        for candidate in ("fedvim_results.json", "vim_vim_paper.json"):
            path = experiment_dir / candidate
            if path.exists():
                return normalize_fedvim_record(path)
        fallback = newest_matching_file(experiment_dir, r"subspace_results_vim_paper_\d{8}_\d{6}\.json")
        if fallback:
            return normalize_fedvim_record(fallback)
        raise FileNotFoundError(f"FedViM result not found in {experiment_dir}")

    if method_key == "ACT-FedViM":
        path = experiment_dir / "act_fedvim_results.json"
        if path.exists():
            return normalize_act_record(path)
        fallback = newest_matching_file(experiment_dir, r"subspace_results_act_\d{8}_\d{6}\.json")
        if fallback:
            return normalize_act_record(fallback)
        raise FileNotFoundError(f"ACT-FedViM result not found in {experiment_dir}")

    if method_key in ("MSP", "Energy"):
        stem = method_key.lower()
        json_path = experiment_dir / "baselines_eval" / f"{stem}_results.json"
        txt_path = experiment_dir / "baselines_eval" / f"{stem}_results.txt"
        if json_path.exists():
            return normalize_baseline_record(json_path, stem)
        if txt_path.exists():
            return normalize_baseline_record(txt_path, stem)
        raise FileNotFoundError(f"{method_key} result not found in {experiment_dir}")

    raise ValueError(f"Unsupported method key: {method_key}")


def collect_records(experiments_root: Path, model_names: list[str]) -> list[dict]:
    """Collect all four method records for each model."""
    records = []
    for model_name in model_names:
        experiment_dir = latest_experiment_dir(experiments_root / model_name)
        model_records = []
        for method_key in ("FedViM", "ACT-FedViM", "MSP", "Energy"):
            record = method_record_for_experiment(experiment_dir, method_key)
            record["model_name"] = model_name
            record["experiment_dir"] = str(experiment_dir)
            if record.get("model_type") is None:
                record["model_type"] = model_name
            model_records.append(record)

        reference = next(record for record in model_records if record["method"] == "FedViM")
        for record in model_records:
            for shared_key in ("checkpoint", "model_type", "image_size", "num_classes", "feature_dim", "id_accuracy"):
                if record.get(shared_key) is None:
                    record[shared_key] = reference.get(shared_key)

        records.extend(model_records)
    return records


def build_comparison_rows(records: list[dict]) -> list[dict]:
    """Build one wide comparison row per model."""
    by_model: dict[str, dict[str, dict]] = {}
    for record in records:
        by_model.setdefault(record["model_name"], {})[record["method"]] = record

    rows = []
    for model_name, methods in by_model.items():
        fedvim = methods["FedViM"]
        act = methods["ACT-FedViM"]
        msp = methods["MSP"]
        energy = methods["Energy"]
        fedvim_fixed_k = fedvim.get("fixed_k")
        act_k = act.get("act_k")
        act_compression = None
        if fedvim_fixed_k not in (None, 0) and act_k is not None:
            act_compression = 1.0 - (act_k / fedvim_fixed_k)
        rows.append(
            {
                "model_name": model_name,
                "feature_dim": fedvim.get("feature_dim") or act.get("feature_dim"),
                "id_accuracy": fedvim.get("id_accuracy"),
                "fedvim_fixed_k": fedvim_fixed_k,
                "act_k": act_k,
                "act_compression_rate": act_compression,
                "fedvim_near_auroc": fedvim.get("near_auroc"),
                "act_near_auroc": act.get("near_auroc"),
                "act_minus_fedvim_near": None if fedvim.get("near_auroc") is None else act.get("near_auroc") - fedvim.get("near_auroc"),
                "msp_near_auroc": msp.get("near_auroc"),
                "energy_near_auroc": energy.get("near_auroc"),
                "fedvim_far_auroc": fedvim.get("far_auroc"),
                "act_far_auroc": act.get("far_auroc"),
                "act_minus_fedvim_far": None if fedvim.get("far_auroc") is None else act.get("far_auroc") - fedvim.get("far_auroc"),
                "msp_far_auroc": msp.get("far_auroc"),
                "energy_far_auroc": energy.get("far_auroc"),
            }
        )
    return sorted(rows, key=lambda row: row["model_name"])


def average_metrics(records: list[dict], method_name: str) -> dict:
    """Average common metrics for a method across models."""
    method_records = [record for record in records if record["method"] == method_name]

    def avg(key):
        values = [record[key] for record in method_records if record.get(key) is not None]
        return None if not values else mean(values)

    return {
        "method": method_name,
        "id_accuracy": avg("id_accuracy"),
        "near_auroc": avg("near_auroc"),
        "far_auroc": avg("far_auroc"),
        "fixed_k": avg("fixed_k"),
        "act_k": avg("act_k"),
        "compression_rate": avg("compression_rate"),
    }


def select_paper_models(comparison_rows: list[dict], top_k: int, max_near_drop: float) -> list[dict]:
    """Select the 2-3 paper models by ACT Near-OOD performance with a regression filter."""
    filtered = [
        row
        for row in comparison_rows
        if row["act_minus_fedvim_near"] is None or row["act_minus_fedvim_near"] >= -max_near_drop
    ]
    ranked = sorted(filtered, key=lambda row: row["act_near_auroc"], reverse=True)
    return ranked[:top_k]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser()

    records = collect_records(experiments_root, args.models)
    comparison_rows = build_comparison_rows(records)
    selected_rows = select_paper_models(comparison_rows, top_k=args.top_k, max_near_drop=args.max_near_drop)

    method_averages = {
        method: average_metrics(records, method)
        for method in ("FedViM", "ACT-FedViM", "MSP", "Energy")
    }
    method_averages["ACT-FedViM"]["compression_rate"] = mean(row["act_compression_rate"] for row in comparison_rows)

    summary = {
        "thesis_title": THESIS_TITLE,
        "experiment_scope": "five_cnn_mainline",
        "models": args.models,
        "selected_models": [row["model_name"] for row in selected_rows],
        "selection_rule": {
            "rank_metric": "ACT Near-OOD AUROC",
            "max_near_drop": args.max_near_drop,
            "top_k": args.top_k,
        },
        "method_averages": method_averages,
        "act_vs_fedvim": {
            "avg_near_delta": mean(row["act_minus_fedvim_near"] for row in comparison_rows),
            "avg_far_delta": mean(row["act_minus_fedvim_far"] for row in comparison_rows),
            "avg_fixed_k": mean(row["fedvim_fixed_k"] for row in comparison_rows),
            "avg_act_k": mean(row["act_k"] for row in comparison_rows),
            "avg_compression_rate": mean(row["act_compression_rate"] for row in comparison_rows),
        },
    }

    write_json(output_prefix.with_name(output_prefix.name + "_records.json"), records)
    write_json(output_prefix.with_name(output_prefix.name + "_full_comparison.json"), comparison_rows)
    write_json(output_prefix.with_name(output_prefix.name + "_selected_models.json"), selected_rows)
    write_json(output_prefix.with_name(output_prefix.name + "_summary.json"), summary)
    write_csv(output_prefix.with_name(output_prefix.name + "_full_comparison.csv"), comparison_rows)

    print(f"Collected {len(records)} method records across {len(args.models)} models.")
    print(f"Selected paper models: {', '.join(summary['selected_models'])}")
    print(f"Summary written to: {output_prefix.with_name(output_prefix.name + '_summary.json')}")


if __name__ == "__main__":
    main()
