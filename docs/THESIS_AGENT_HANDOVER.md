# Thesis Agent Handover

## Purpose

This document is for the next agent who will continue the thesis writing work on top of the current codebase and finalized experiment outputs.

The current repo state is aligned to commit:

- `3e274d52dfcab12b923ff91ac70d7309af6367c1`

Repository:

- `origin = git@github.com:SunWeizhou/Fed.git`
- branch used for the current synced state: `main`

## Thesis Scope

Current thesis title:

- `FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究`

Current formal scope is restricted to:

- main methods: `FedViM`, `ACT-FedViM`
- output-space baselines inside the mainline experiment: `MSP`, `Energy`
- supplemental federated OOD baselines: `FOSTER (MSP)`, `FedLN (MSP)`
- pooled controls: `Pooled-ViM`, `Pooled-ACT-ViM`

Current backbone set is fixed to five CNNs:

- `mobilenetv3_large`
- `resnet50`
- `resnet101`
- `densenet169`
- `efficientnet_v2_s`

Items that are no longer part of the thesis mainline:

- `FOOGD`
- `FedATOL`
- transformer backbones
- ConvNeXt and other exploratory models

## Formal Experimental Protocol

The current formal protocol is:

1. Train a federated classifier backbone with `FedAvg`.
2. Use validation accuracy only to select `best_model.pth`.
3. Freeze `best_model`.
4. Run post-hoc evaluation scripts separately.

Current split protocol:

- `n_clients = 5`
- `Dirichlet alpha = 0.1`
- `seed = 42`
- `image_size = 320`
- `batch_size = 32`
- `communication_rounds = 50`
- `local_epochs = 4`
- validation set = `10%` split from `D_ID_train`

Canonical split file:

- [canonical_split_seed42_alpha0.1_nclients5.json](/home/dell7960/桌面/FedOOD/splits/canonical_split_seed42_alpha0.1_nclients5.json)

This canonical manifest is important. The repo was refactored so all formal training and formal post-hoc evaluation use fixed deterministic splits instead of regenerating partitions ad hoc on different machines.

## Method Definitions Used in the Thesis

### Mainline

- `FedViM`: clients compute first- and second-order feature sufficient statistics on their local ID-train splits; the server aggregates them into global `mu` and `cov`, then calibrates empirical `alpha` from federated residual/energy statistics.
- `ACT-FedViM`: same federated post-hoc pipeline, but replaces fixed-k with ACT-based adaptive subspace selection.
- `MSP`, `Energy`: standard output-space baselines computed directly on the frozen classifier.

### Supplemental baselines

- `FedLN (MSP)`: thesis-oriented independent federated LogitNorm baseline.
- `FOSTER (MSP)`: thesis-oriented independent FOSTER baseline.

### Pooled controls

- `Pooled-ViM`: same frozen `best_model`, but ViM statistics are computed centrally on the pooled union of the five client ID-train splits.
- `Pooled-ACT-ViM`: same pooled control for ACT.

## What the Thesis Should Claim

There are two distinct evidence chains in the current final results.

### 1. Why FedViM is useful

Use:

- `FedViM`
- `ACT-FedViM`
- `FOSTER`
- `FedLN`
- `MSP`
- `Energy`

Interpretation:

- `FedViM` and especially `ACT-FedViM` outperform the more generic federated OOD baselines.
- This supports the thesis claim that a ViM-based federated post-hoc route is effective for multi-center plankton monitoring.

### 2. Why federating ViM is reasonable

Use:

- `FedViM` vs `Pooled-ViM`
- `ACT-FedViM` vs `Pooled-ACT-ViM`

Interpretation:

- federating the sufficient statistics does not materially damage ViM
- pooled and federated ViM are effectively equivalent under the finalized protocol

### 3. Why ViM was chosen

Do not reframe this as a new ablation.

Use Han et al. as the direct task-specific evidence that ViM is a strong score on plankton OOD. That belongs in related work / method motivation, not as a fresh experimental claim.

## Final Result Files

These are the current authoritative result artifacts for thesis writing.

### Mainline summary

- [manifest_final_mainline_summary.json](/home/dell7960/桌面/FedOOD/paper_tools/manifest_final_mainline_summary.json)
- [manifest_final_mainline_full_comparison.json](/home/dell7960/桌面/FedOOD/paper_tools/manifest_final_mainline_full_comparison.json)
- [manifest_final_mainline_selected_models.json](/home/dell7960/桌面/FedOOD/paper_tools/manifest_final_mainline_selected_models.json)

### Mainline vs federated baselines

- [manifest_final_v1_main_vs_baselines.json](/home/dell7960/桌面/FedOOD/paper_tools/manifest_final_v1_main_vs_baselines.json)
- [manifest_final_v1_main_vs_baselines.csv](/home/dell7960/桌面/FedOOD/paper_tools/manifest_final_v1_main_vs_baselines.csv)

### Federated vs pooled consistency

- [manifest_final_v1_pooled_consistency.json](/home/dell7960/桌面/FedOOD/paper_tools/manifest_final_v1_pooled_consistency.json)
- [manifest_final_v1_pooled_consistency.csv](/home/dell7960/桌面/FedOOD/paper_tools/manifest_final_v1_pooled_consistency.csv)

### Supplemental baseline summaries

- [manifest_final_foster_summary.json](/home/dell7960/桌面/FedOOD/paper_tools/manifest_final_foster_summary.json)
- [manifest_final_fedln_summary.json](/home/dell7960/桌面/FedOOD/paper_tools/manifest_final_fedln_summary.json)

## Final Average Results

Five-model averages:

- `FedViM`: Near AUROC `0.9252`, Far AUROC `0.9388`
- `ACT-FedViM`: Near AUROC `0.9637`, Far AUROC `0.9660`
- `MSP`: Near AUROC `0.8617`, Far AUROC `0.8253`
- `Energy`: Near AUROC `0.7979`, Far AUROC `0.7722`
- `FedLN (MSP)`: Near AUROC `0.9212`, Far AUROC `0.8738`
- `FOSTER (MSP)`: Near AUROC `0.8485`, Far AUROC `0.7780`

ACT compression summary:

- average fixed-k = `804.8`
- average ACT k = `103`
- average compression rate = `86.7453%`

Pooled consistency summary:

- average `Pooled-ViM - FedViM` delta on Near AUROC = `+0.000994`
- average `Pooled-ViM - FedViM` delta on Far AUROC = `-0.000505`
- average `Pooled-ACT-ViM - ACT-FedViM` delta on Near AUROC = approximately `0`
- average `Pooled-ACT-ViM - ACT-FedViM` delta on Far AUROC = approximately `0`

## Per-model Interpretive Notes

### mobilenetv3_large

- lightweight representative case
- `FedViM` is already very strong
- `ACT-FedViM` is close to `FedViM`
- pooled control is effectively identical

### resnet101

- balanced representative case
- `FedViM` and `ACT-FedViM` are both very strong
- pooled control is effectively identical
- good candidate when the thesis needs one clean, stable flagship example

### densenet169

- this is the fixed-k mismatch case
- `FedViM` fixed-k result is much lower than the other four models
- `ACT-FedViM` corrects that mismatch and jumps back to the top tier
- pooled fixed-k ViM is still poor, so this is not a federated aggregation failure
- the correct interpretation is: `ACT` repairs an unsuitable fixed-k setting on this backbone

### resnet50

- `FedViM` is strong
- `ACT-FedViM` gives a modest positive gain
- pooled control remains effectively identical

### efficientnet_v2_s

- `FedViM` is strong
- `ACT-FedViM` is close to `FedViM`
- pooled control remains effectively identical
- `FOSTER` is particularly weak on this model under the current thesis-oriented protocol

## Current Writing Guidance

The next agent should keep the following narrative stable.

### Main contribution hierarchy

- `FedViM` is the principal contribution.
- `ACT-FedViM` is an extension on top of `FedViM`, not the main thesis contribution.

### Introduction logic

The introduction should progress in this order:

1. marine plankton monitoring need
2. OOD / open-environment risk
3. why closed-set accuracy is insufficient
4. why multi-center data cannot simply be pooled
5. why federated learning is needed
6. why post-hoc OOD is preferred here
7. why `ViM` is chosen
8. why `ACT` is only an extension

### Related work logic

The related-work chapter should explicitly cover:

- plankton monitoring and open-world recognition
- post-hoc OOD detection
- Han et al. benchmark as the direct reason to choose ViM
- federated learning basics
- generic federated OOD baselines, especially `FedLN` and `FOSTER`

### Results chapter logic

The results chapter should be split into four blocks:

1. five-backbone mainline results: `FedViM / ACT-FedViM / MSP / Energy`
2. pooled consistency: `FedViM ≈ Pooled-ViM`, `ACT-FedViM ≈ Pooled-ACT-ViM`
3. supplemental federated baselines: `FOSTER / FedLN`
4. representative-model analysis and ACT compression discussion

## Draft File to Continue

Current thesis draft:

- [FedViM_论文草稿.md](/home/dell7960/桌面/FedOOD/docs/paper/FedViM_论文草稿.md)

The next agent should treat the result JSON files as the source of truth and update the thesis draft to match them exactly.

## Commands to Rebuild Result Summaries

Mainline:

```bash
python3 paper_tools/collect_paper_results.py \
  --experiments-root experiments/fedavg_manifest_final_v1 \
  --output-prefix paper_tools/manifest_final_mainline
```

Baseline summaries:

```bash
python3 paper_tools/collect_foster_results.py \
  --experiments-root experiments/foster_manifest_final_v1 \
  --output-prefix paper_tools/manifest_final_foster

python3 paper_tools/collect_fedln_results.py \
  --experiments-root experiments/fedln_manifest_final_v1 \
  --output-prefix paper_tools/manifest_final_fedln
```

Final thesis comparison artifacts:

```bash
python3 paper_tools/collect_manifest_final_results.py \
  --main-root experiments/fedavg_manifest_final_v1 \
  --foster-root experiments/foster_manifest_final_v1 \
  --fedln-root experiments/fedln_manifest_final_v1 \
  --output-prefix paper_tools/manifest_final_v1
```

## Important Boundaries

- Do not reintroduce `FOOGD` into the thesis main baseline set.
- Do not treat `FOSTER` or `FedLN` as pooled controls. They are generic federated OOD baselines.
- Do not claim the thesis proves ViM should be chosen on plankton from scratch. Use Han et al. for that justification.
- Do not describe the `densenet169` fixed-k failure as a federated failure. It is a fixed-k mismatch case.
- Do not use old pre-manifest experiment directories for formal thesis numbers.

## Local-Only Items Not Synced to GitHub

The following currently exist locally but were intentionally not included in the synced GitHub state:

- `Reference/baselines/`
- `splits/canonical_split_seed42_alpha0.1_nclients1.json`

They should not be used as thesis source-of-truth artifacts.
