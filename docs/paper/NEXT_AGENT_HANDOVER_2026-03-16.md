# Next Agent Handover

Date: 2026-03-18

## Project Definition

Current paper title remains `ACT-FedViM`.

The implementation and paper logic are aligned to this definition:

- `Original Fed-ViM`: federated ViM baseline with heuristic subspace selection and empirical alpha
- `ACT-FedViM`: post-hoc ACT-based subspace refinement on the same trained checkpoint

Important clarification:

- This is **not** ACT integrated into federated training.
- This is **Fed-ViM training first, ACT post-processing second**.

## Official Experimental Protocol

- Data split: `5 clients`, `Dirichlet alpha = 0.1`
- Training: `Fed-ViM`
- Official score:
  - `Score(x) = Energy(x) - alpha * Residual(x)`
- Official alpha:
  - empirical alpha from client-side ID-train statistics
  - `alpha = |mean(Energy)| / mean(Residual)`

Alpha is a **post-hoc evaluation parameter**, not a training parameter.

## Current Main Experimental Reality

The most important recent finding is that the original ACT post-hoc variant is **not** the strongest ACT-style method in this repo.

### Methods now tested on rerun checkpoints

- `Variance-0.90`
- `Variance-0.95`
- `Variance-0.99`
- `ViM-paper`
- `ACT-FedViM`
- `ACT-cov-reordered`
- `ACT-corr-reordered`
- `MSP`
- `Energy`

Main comparison table:

- [grouped_ood_table.md](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/grouped_ood_table.md)

### Current conclusions

1. `ACT-cov-reordered` is not useful.
2. `ACT-corr-reordered` is the most promising ACT direction tested so far.
3. `ViM-paper` is still a very strong competitor and often wins.
4. `MSP` and `Energy` are clearly weaker than ViM-style methods.

Do **not** claim that ACT uniformly beats all heuristics.

## Removed Exploratory Variants

The following exploratory branches were tested and then removed from the working tree:

- additional score-term branch
- foreground-cropping branch
- correlation-shrinkage branch

They should not be treated as active evaluation paths.

## Dataset/Feature Interpretation Already Established

This dataset is not natural-image-like. Important properties already observed:

- black/dark background
- sparse foreground
- strong long-tail class distribution
- color channels highly correlated
- morphology matters more than color for OOD separation

This explains why:

- ACT often selects very small `k`
- aggressive truncation can hurt
- correlation-space methods are more promising than covariance-space ACT

## Key Files

Training:

- [train_federated.py](/home/dell7960/桌面/FedRoD/Fed-ViM/train_federated.py)

Main evaluation:

- [server.py](/home/dell7960/桌面/FedRoD/Fed-ViM/server.py)

Shared OOD logic:

- [utils/ood_utils.py](/home/dell7960/桌面/FedRoD/Fed-ViM/utils/ood_utils.py)

Main ACT post-hoc:

- [advanced_fedvim.py](/home/dell7960/桌面/FedRoD/Fed-ViM/advanced_fedvim.py)

Exploratory ACT space script:

- [evaluate_act_reordered_spaces.py](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/evaluate_act_reordered_spaces.py)

Current result tables:

- [grouped_ood_table.md](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/grouped_ood_table.md)
- [nine_way_summary.md](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/nine_way_summary.md)

## Practical Guidance For The Next Agent

1. Treat `ACT-corr-reordered` as the strongest ACT-style baseline so far.
2. Do **not** invest more time in:
   - `ACT-cov-reordered`
   - deleted exploratory branches
3. If continuing ACT rescue attempts, prioritize:
   - architecture-specific ACT analysis
4. Keep paper wording conservative:
   - ACT is competitive and statistically motivated
   - but it does not yet consistently beat strong heuristics like `ViM-paper`
