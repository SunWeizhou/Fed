# Dialogue And Exploration Summary

Date: 2026-03-18

## Purpose

This document summarizes the recent discussion-driven exploration work after the main rerun experiments, with emphasis on:

- what was tested
- what worked
- what failed
- what should be considered the current best ACT-related direction

## 1. Main problem identified

The original concern was:

- `ACT-FedViM` did not consistently beat heuristic ViM variants
- in several models, `ViM-paper` or variance heuristics performed better

This led to a deeper diagnosis of **how ACT was being used**, not only whether ACT itself was mathematically valid.

## 2. Key conceptual finding

The original ACT implementation in this repo mainly used ACT to choose `k`, then returned to covariance-space PCA for the final directions.

This suggested a possible mismatch:

- ACT correction was computed in one spectral space
- but the final subspace directions were still taken from another ordering

That motivated testing reordered-direction variants.

## 3. Variants tested

### 3.1 `ACT-cov-reordered`

Result:

- not useful
- did not improve over the original ACT pipeline

### 3.2 `ACT-corr-reordered`

Result:

- clearly the most promising ACT-style variant tested so far
- often much better than the original `ACT-FedViM`
- competitive with `ViM-paper`, though not consistently better

This is currently the most important ACT rescue direction.

### 3.3 Dual-score ACT

Idea:

- combine:
  - `Energy`
  - orthogonal residual
  - ACT-corrected in-subspace Mahalanobis term

Result:

- not useful in current experiments
- no stable gain over simpler methods

Conclusion:

- do not continue prioritizing this branch

### 3.4 Removed exploratory branches

The following branches were explored and then deleted from the working tree:

- a foreground-cropping branch
- a correlation-shrinkage branch
- an extra score-term branch

## 4. Best current interpretation

The current best interpretation is:

1. ACT itself is not dead.
2. The original ACT usage in this repo was likely suboptimal.
3. Correlation-space reordered ACT is the strongest ACT-related direction found so far.
4. Several auxiliary rescue attempts were explored and then discarded.

## 5. Recommended next research priority

If continuing ACT-centered exploration, prioritize:

1. fully analyzing `ACT-corr-reordered`
2. architecture-specific analysis of why ACT helps some CNNs more than transformers

Do not prioritize:

- covariance-space reordered ACT

## 6. Most important files from this exploration

- [evaluate_act_reordered_spaces.py](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/evaluate_act_reordered_spaces.py)
- [grouped_ood_table.md](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/grouped_ood_table.md)
- [nine_way_summary.md](/home/dell7960/桌面/FedRoD/Fed-ViM/paper_tools/nine_way_summary.md)

## 7. One-sentence status

Current best ACT-related story is:

`ACT-corr-reordered is the only clearly promising ACT-based variant so far; other rescue attempts either failed or produced only marginal stabilization gains.`
