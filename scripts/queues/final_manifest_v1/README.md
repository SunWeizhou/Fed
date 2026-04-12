# Final Queue Layout

Official rerun queue for the thesis baselines under the fixed manifest:

- Methods: `FedAvg`, `FOSTER`, `FedLN`
- Backbones: `mobilenetv3_large`, `resnet101`, `densenet169`, `resnet50`, `efficientnet_v2_s`
- Shared settings: `n_clients=5`, `alpha=0.1`, `communication_rounds=50`, `local_epochs=4`, `batch_size=32`, `image_size=320`, `num_workers=8`, `seed=42`
- Split manifest: `/home/dell7960/桌面/FedOOD/splits/canonical_split_seed42_alpha0.1_nclients5.json`

Queue assignment:

- `local_gpu0.sh`
  - `1/3` `FedAvg mobilenetv3_large`
  - `2/3` `FOSTER mobilenetv3_large`
  - `3/3` `FedAvg resnet50`
- `local_gpu1.sh`
  - `1/3` `FedLN mobilenetv3_large`
  - `2/3` `FedLN resnet50`
  - `3/3` `FedAvg efficientnet_v2_s`
- `linux_gpu0.sh`
  - `1/2` `FedAvg resnet101`
  - `2/2` `FedAvg densenet169`
- `linux_gpu1.sh`
  - `1/3` `FedLN resnet101`
  - `2/3` `FedLN densenet169`
  - `3/3` `FedLN efficientnet_v2_s`
- `dell_gpu0.cmd`
  - `1/2` `FOSTER resnet101`
  - `2/2` `FOSTER resnet50`
- `dell_gpu1.cmd`
  - `1/2` `FOSTER densenet169`
  - `2/2` `FOSTER efficientnet_v2_s`
