"""FOOGD-specific data wrappers that preserve the thesis split protocol."""

from __future__ import annotations

import random
from math import sqrt

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from data_utils import (
    PlanktonDataset,
    _build_loader_kwargs,
    _create_federated_train_subsets,
    get_transforms,
)


def pil_image_to_numpy(image: Image.Image) -> np.ndarray:
    """Module-level helper so Windows DataLoader workers can pickle the transform."""
    return np.asarray(image)


def colorful_spectrum_mix(img1: np.ndarray, img2: np.ndarray, alpha: float, ratio: float = 1.0):
    lam = np.random.uniform(0, alpha)
    assert img1.shape == img2.shape
    h, w, _ = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))
    img1_abs_copy = np.copy(img1_abs)
    img2_abs_copy = np.copy(img2_abs)

    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = (
        lam * img2_abs_copy[h_start:h_start + h_crop, w_start:w_start + w_crop]
        + (1.0 - lam) * img1_abs_copy[h_start:h_start + h_crop, w_start:w_start + w_crop]
    )
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = (
        lam * img1_abs_copy[h_start:h_start + h_crop, w_start:w_start + w_crop]
        + (1.0 - lam) * img2_abs_copy[h_start:h_start + h_crop, w_start:w_start + w_crop]
    )

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    return img21


class FOOGDFourierPairDataset(Dataset):
    """Return one original view and one Fourier-mixed view from the same client split."""

    def __init__(self, dataset: Subset, image_size: int, fourier_mix_alpha: float = 1.0) -> None:
        self.dataset = dataset
        self.fourier_mix_alpha = float(fourier_mix_alpha)
        self.pre_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            pil_image_to_numpy,
        ])
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        if not isinstance(image, Image.Image):
            raise TypeError("FOOGD Fourier dataset expects raw PIL images.")
        mix_index = random.randint(0, len(self.dataset) - 1)
        other_image, _ = self.dataset[mix_index]
        if not isinstance(other_image, Image.Image):
            raise TypeError("FOOGD Fourier dataset expects raw PIL images.")

        image_array = np.array(self.pre_transform(image), copy=True)
        other_array = np.array(self.pre_transform(other_image), copy=True)
        mixed_array = colorful_spectrum_mix(image_array, other_array, alpha=self.fourier_mix_alpha)
        image_tensor = self.post_transform(np.array(image_array, copy=True))
        mixed_tensor = self.post_transform(np.array(mixed_array, copy=True))
        return image_tensor, mixed_tensor, label


def create_foogd_federated_loaders(
    data_root: str,
    n_clients: int,
    alpha: float,
    batch_size: int,
    image_size: int,
    num_workers: int,
    partition_seed: int,
    fourier_mix_alpha: float,
):
    train_dataset, val_dataset, client_indices = _create_federated_train_subsets(
        data_root=data_root,
        n_clients=n_clients,
        alpha=alpha,
        image_size=image_size,
        partition_seed=partition_seed,
    )
    raw_full_train_dataset = PlanktonDataset(data_root, transform=None, mode="train")
    raw_train_dataset = Subset(raw_full_train_dataset, train_dataset.indices)
    _, test_transform = get_transforms(image_size)

    client_loaders = []
    for client_id in range(n_clients):
        client_raw_subset = Subset(raw_train_dataset, client_indices[client_id])
        client_dataset = FOOGDFourierPairDataset(
            client_raw_subset,
            image_size=image_size,
            fourier_mix_alpha=fourier_mix_alpha,
        )
        client_loader = DataLoader(
            client_dataset,
            **_build_loader_kwargs(batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False),
        )
        client_loaders.append(client_loader)
        print(f"客户端 {client_id}: {len(client_dataset)} 样本")

    test_dataset = PlanktonDataset(data_root, transform=test_transform, mode="test")
    near_dataset = PlanktonDataset(data_root, transform=test_transform, mode="near_ood")
    far_dataset = PlanktonDataset(data_root, transform=test_transform, mode="far_ood")

    test_loader = DataLoader(
        test_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
    near_loader = DataLoader(
        near_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
    far_loader = DataLoader(
        far_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )

    print("FOOGD 数据加载器创建完成:")
    print(f"  - 客户端数量: {len(client_loaders)}")
    print(f"  - 训练集 (联邦): {len(raw_train_dataset)} 样本")
    print(f"  - 验证集 (服务端): {len(val_dataset)} 样本")
    print(f"  - 测试集: {len(test_dataset)} 样本")
    print(f"  - Near-OOD: {len(near_dataset)} 样本")
    print(f"  - Far-OOD: {len(far_dataset)} 样本")
    return client_loaders, test_loader, near_loader, far_loader, val_loader
