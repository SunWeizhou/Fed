# FOSTER Baseline Notes

## Reference

- Paper: `Turning the Curse of Heterogeneity in Federated Learning into a Blessing for Out-of-Distribution Detection`
- OpenReview: `https://openreview.net/forum?id=mMNimwRb7Gr`
- Official code: `https://github.com/illidanlab/FOSTER`

## This Repository's Adaptation

This repository keeps `FedViM` as the thesis mainline and adds `FOSTER` only as an independent supplemental baseline.

The implementation here is a thesis-oriented adaptation rather than a full reproduction of the official repository:

- It reuses the current project data split, backbones, validation protocol, and output schema.
- It uses a central class-conditional feature generator in backbone feature space instead of the original image-level or broader codebase-specific pipeline.
- It keeps a standard FedAvg backbone training loop and adds generator-assisted synthetic external-feature regularization after a warmup stage.
- It evaluates the trained checkpoint with the same `D_ID_test`, `D_Near_test`, and `D_Far_test` protocol used by the main thesis experiments.

## Supported Backbones

First release support is intentionally limited to:

- `resnet50`
- `resnet101`

## Training

```bash
python3 train_foster.py \
  --model_type resnet50 \
  --data_root ./Plankton_OOD_Dataset \
  --device cuda:0
```

## Evaluation

```bash
python3 evaluate_foster.py \
  --checkpoint ./experiments/foster_v1/resnet50/experiment_xxx/best_model.pth \
  --data_root ./Plankton_OOD_Dataset \
  --evaluation_score msp
```

## Result Collection

```bash
python3 paper_tools/collect_foster_results.py \
  --experiments-root experiments/foster_v1 \
  --output-prefix paper_tools/foster
```

## Current Simplifications

- No changes are made to `FedViM` mainline training or evaluation files.
- The first release uses one global classifier and one central feature generator.
- The generator is optimized by class-conditional cross-entropy against the current global classifier.
- The client regularizer uses OE loss on generated external features sampled from classes missing in the client's local split.
- The main thesis table should report the `msp`-based FOSTER result first; other output scores can be stored in JSON as auxiliary runs.
