#!/usr/bin/env python3
"""Collect independent FOOGD baseline results into standalone artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect FOOGD baseline results.")
    parser.add_argument("--experiments-root", type=str, default="experiments/foogd_v1")
    parser.add_argument("--output-prefix", type=str, default="paper_tools/foogd")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_records(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.glob("*/experiment_*/foogd_results*.json")):
        record = load_json(path)
        record["source_file"] = str(path.resolve())
        records.append(record)
    return records


def build_summary(records: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = {}
    for record in records:
        by_model.setdefault(record["model_type"], []).append(record)

    per_model = {}
    for model_name, model_records in by_model.items():
        preferred = sorted(
            model_records,
            key=lambda item: (item.get("evaluation_score") != "sm", item.get("timestamp", "")),
        )[0]
        per_model[model_name] = preferred

    return {
        "experiment_count": len(records),
        "models": sorted(per_model.keys()),
        "per_model": per_model,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_type",
        "evaluation_score",
        "id_accuracy",
        "near_auroc",
        "near_aupr",
        "near_fpr95",
        "far_auroc",
        "far_aupr",
        "far_fpr95",
        "checkpoint",
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name in summary["models"]:
            record = summary["per_model"][model_name]
            writer.writerow({field: record.get(field) for field in fields})


def main() -> None:
    args = parse_args()
    root = Path(args.experiments_root).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    records = collect_records(root)
    summary = build_summary(records)

    write_json(output_prefix.with_name(f"{output_prefix.name}_records.json"), {"records": records})
    write_json(output_prefix.with_name(f"{output_prefix.name}_summary.json"), summary)
    write_csv(output_prefix.with_name(f"{output_prefix.name}_summary.csv"), summary)

    print(f"Collected {len(records)} FOOGD result files from {root}")
    print(f"Summary models: {summary['models']}")


if __name__ == "__main__":
    main()
