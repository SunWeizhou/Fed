# ACT-FedViM Paper Execution Checklist

## Goal

Build a complete paper pipeline for `ACT-FedViM` around the application story defined in:

- `docs/paper/ACT-FedViM.pdf`
- `docs/paper/孙伟洲本科生开题报告.docx`

Core claim:

`ACT-FedViM` enables privacy-friendly, lightweight, and adaptive federated OOD detection for distributed marine plankton monitoring.

## Paper Storyline

### Problem setting

- Distributed marine monitoring nodes cannot upload massive raw images.
- Edge devices cannot afford high-cost generative OOD methods.
- High-dimensional backbone features contain noisy long-tail eigen-directions.

### Method structure

1. Federated sufficient-statistics aggregation reconstructs global covariance.
2. ACT selects an adaptive and denoised subspace dimension.
3. ViM uses energy plus residual for lightweight OOD scoring.

### Main message to prove experimentally

- `ACT-FedViM` improves CNN backbones clearly.
- `ACT-FedViM` keeps Transformer backbones stable while compressing subspace size.
- The method is practical for marine edge deployment because it is privacy-friendly and lightweight.

## Required Experiments

### Main results

- 9 backbones on the unified DYB-PlanktonNet federated setup
- Metrics: ID Accuracy, Near-OOD AUROC, Far-OOD AUROC
- Original ViM vs ACT-FedViM

### Baseline comparison

- MSP
- Energy
- MSP / Energy
- Original ViM
- ACT-FedViM

### Ablations

- ACT vs fixed variance thresholds: 90%, 95%, 99%
- Client count: 3, 5, 10
- Dirichlet alpha: 0.1, 0.5, 1.0
- `freeze_bn` on/off for ConvNeXt and Swin-family models
- Alpha recalibration on/off after ACT

### Efficiency analysis

- Original `k` vs ACT `k`
- Compression rate
- Inference latency
- Communication payload for uploaded statistics

## Required Tables

1. Experiment setup table
2. Main 9-backbone result table
3. Baseline comparison table
4. ACT ablation table
5. Federated sensitivity table
6. Efficiency table
7. CNN vs Transformer summary table

## Required Figures

1. Backbone performance bar chart
2. ROC comparison figure
3. Eigenvalue spectrum before/after ACT
4. t-SNE feature visualization
5. Compression rate vs AUROC gain scatter plot

## Code Tasks

### Phase 1: paper infrastructure

- [ ] Standardize paper experiment configuration
- [x] Add result collection script
- [x] Add baseline collection script
- [x] Add ablation command generator

### Phase 2: evaluation closure

- [x] Export all metrics to unified CSV/JSON
- [x] Remove any test-set-dependent training decisions
- [ ] Support multi-seed aggregation
- [x] Generate markdown-ready tables automatically

### Phase 3: paper figures

- [ ] Add figure generation script for all core plots
- [ ] Save figures to a dedicated paper output directory

## Current Immediate Next Steps

1. Use `paper_tools/collect_paper_results.py` to collect current `experiments_v6` results.
2. Freeze a single source of truth CSV for all reported model results.
3. Build the main result table from that CSV.
4. Use `paper_tools/generate_ablation_commands.py` to launch the next ablation suite.
