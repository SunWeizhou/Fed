# Repository Guidelines

## Project Structure & Module Organization

Core training and evaluation code lives at the repository root:

- `train_federated.py`: main federated training entrypoint
- `client.py`, `server.py`: client/server FL logic
- `models.py`, `config.py`, `data_utils.py`: model definitions, defaults, data loading
- `advanced_fedvim.py`: ACT-FedViM post-hoc evaluation
- `evaluate_baselines.py`: federated-compatible `MSP` and `Energy` baselines

Supporting directories:

- `paper_tools/`: result collection, table generation, post-hoc pipelines
- `docs/`, `docs/paper/`, `Reference/`: paper notes, methodology, handover docs
- `examples/`: historical CIFAR prototype
- `experiments/experiments_rerun_v1/`: current official rerun outputs
- `experiments/experiments_v6/`: older experiment set

## Build, Test, and Development Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run federated training:

```bash
python3 train_federated.py --model_type resnet50 --use_fedvim --data_root ./Plankton_OOD_Dataset
```

Run ACT post-hoc on a checkpoint:

```bash
python3 advanced_fedvim.py --checkpoint path/to/best_model.pth --data_root ./Plankton_OOD_Dataset --subspace_method act --alpha_method empirical
```

Refresh paper artifacts:

```bash
python3 paper_tools/collect_paper_results.py --experiments-root experiments/experiments_rerun_v1 --output-prefix paper_tools/rerun_v1_results
python3 paper_tools/generate_paper_tables.py
```

Quick syntax validation:

```bash
python3 -B -m py_compile train_federated.py advanced_fedvim.py server.py
```

## Coding Style & Naming Conventions

Use Python with 4-space indentation and ASCII by default. Prefer `snake_case` for functions, variables, CLI flags, and filenames; use clear experiment names such as `experiment_YYYYMMDD_HHMMSS`. Keep comments short and technical. Reuse shared helpers in `utils/` or `paper_tools/` instead of duplicating scoring logic.

## Testing Guidelines

There is no dedicated `pytest` suite yet. Treat `py_compile`, targeted script runs, and artifact checks as the minimum validation bar. For new evaluation code, verify both console output and generated files such as `subspace_results_*.json`, `final_report.txt`, or `paper_tools/*.md`.

## Commit & Pull Request Guidelines

Recent history uses short imperative summaries, e.g. `Add experiment analysis scripts and project documentation`, `Fix alpha recalibration...`. Follow the same pattern: verb-first, specific scope, one logical change per commit.

PRs should include:

- purpose and affected pipeline stage
- exact commands run
- changed output paths or tables
- representative metrics if results changed

## Experiment & Documentation Notes

The current official protocol is `Fed-ViM + empirical alpha`, with `ACT-FedViM` as post-hoc refinement. Baseline comparisons in the federated setting should stay limited to methods that do not require server-side access to labeled training features. Keep paper-facing docs aligned with code; if evaluation logic changes, update `docs/paper/` and `Reference/` in the same change.
