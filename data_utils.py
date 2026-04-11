#!/usr/bin/env python3
"""
数据工具模块 - 用于联邦学习的数据划分和加载
基于项目工作文档中的严格类别定义

作者: Claude Code
日期: 2025-11-22
"""

import os
import json
import numpy as np
import torch
import pickle
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 严格类别定义 - 根据新项目工作文档
ID_CLASSES = [
    "Polychaeta_most with eggs", "Polychaeta Type A", "Polychaeta Type B", "Polychaeta Type C",
    "Polychaeta Type D", "Polychaeta Type E", "Polychaeta Type F", "Penilia avirostris",
    "Evadne tergestina", "Acartia sp.A", "Acartia sp.B", "Acartia sp.C", "Calanopia sp.",
    "Labidocera sp.", "Tortanus gracilis", "Calanoid with egg", "Calanoid Type A",
    "Calanoid Type B", "Oithona sp.B with egg", "Cyclopoid Type A with egg",
    "Harpacticoid mating", "Microsetella sp.", "Caligus sp.", "Copepod Type A", "Caprella sp.",
    "Amphipoda Type A", "Amphipoda Type B", "Amphipoda Type C", "Gammarids Type A",
    "Gammarids Type B", "Gammarids Type C", "Cymodoce sp.", "Lucifer sp.", "Macrura larvae",
    "Megalopa larva Phase 1 Type B", "Megalopa larva Phase 1 Type C",
    "Megalopa larva Phase 1 Type D", "Megalopa larva_Phase 2", "Porcrellanidae larva",
    "Shrimp-like larva Type A", "Shrimp-like larva Type B", "Shrimp-like Type A",
    "Shrimp-like Type B", "Shrimp-like Type D", "Shrimp-like Type F", "Cumacea Type A",
    "Cumacea Type B", "Chaetognatha", "Oikopleura sp. parts", "Tunicata Type A",
    "Jellyfish", "Creseis acicula", "Noctiluca scintillans", "Phaeocystis globosa"
]

NEAR_OOD_CLASSES = [
    "Polychaeta larva", "Calanoid Nauplii", "Calanoid Type C", "Calanoid Type D",
    "Oithona sp.A with egg", "Cyclopoid Type A", "Harpacticoid", "Monstrilla sp.A",
    "Monstrilla sp.B", "Megalopa larva Phase 1 Type A", "Shrimp-like Type C",
    "Shrimp-like Type E", "Ostracoda", "Oikopleura sp.", "Actiniaria larva", "Hydroid",
    "Jelly-like", "Bryozoan larva", "Gelatinous Zooplankton", "Unknown Type A",
    "Unknown Type B", "Unknown Type C", "Unknown Type D", "Balanomorpha exuviate",
    "Monstrilloid", "Fish Larvae"
]

FAR_OOD_CLASSES = [
    "Crustacean limb Type A", "Crustacean limb_Type B", "Fish egg",
    "Particle filamentous Type A", "Particle filamentous Type B", "Particle bluish",
    "Particle molts", "Particle translucent flocs", "Particle_yellowish flocs",
    "Particle_yellowish rods", "Bubbles", "Fish tail"
]


# data_utils.py
DATA_CACHE_FORMAT_VERSION = 2
SPLIT_MANIFEST_VERSION = 1
CANONICAL_TRAIN_VAL_SEED = 42


def _splits_dir():
    """Canonical split manifests stored in-repo for cross-machine reuse."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")


def _default_split_manifest_path(n_clients, alpha, partition_seed):
    seed_str = "none" if partition_seed is None else str(partition_seed)
    filename = f"canonical_split_seed{seed_str}_alpha{alpha}_nclients{n_clients}.json"
    return os.path.join(_splits_dir(), filename)


def get_split_manifest_path(n_clients, alpha, partition_seed):
    """Public helper for recording the canonical split file used by an experiment."""
    return _default_split_manifest_path(n_clients=n_clients, alpha=alpha, partition_seed=partition_seed)


def _build_client_class_distribution(client_indices_full, labels):
    distribution = []
    for client_full_indices in client_indices_full:
        per_client = [0] * len(ID_CLASSES)
        for full_idx in client_full_indices:
            label = int(labels[full_idx])
            if label != -1:
                per_client[label] += 1
        distribution.append(per_client)
    return distribution


def _save_split_manifest(manifest_path, manifest_payload):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, ensure_ascii=False, indent=2)


def _generate_split_manifest(data_root, n_clients, alpha, image_size, partition_seed, manifest_path):
    """Generate one canonical split file that every machine can reuse verbatim."""
    train_transform, _ = get_transforms(image_size)
    full_train_dataset = PlanktonDataset(data_root, transform=train_transform, mode='train')

    total_len = len(full_train_dataset)
    val_len = int(total_len * 0.1)
    train_len = total_len - val_len

    train_dataset, val_dataset_raw = torch.utils.data.random_split(
        full_train_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(CANONICAL_TRAIN_VAL_SEED)
    )

    train_indices_full = [int(idx) for idx in train_dataset.indices]
    val_indices_full = [int(idx) for idx in val_dataset_raw.indices]
    client_indices_rel = partition_data(train_dataset, n_clients=n_clients, alpha=alpha, seed=partition_seed)
    client_indices_full = [
        [train_indices_full[int(rel_idx)] for rel_idx in client_rel_indices]
        for client_rel_indices in client_indices_rel
    ]

    manifest_payload = {
        "manifest_version": SPLIT_MANIFEST_VERSION,
        "split_name": os.path.basename(manifest_path),
        "data_root_basename": os.path.basename(os.path.abspath(data_root)),
        "n_clients": int(n_clients),
        "alpha": float(alpha),
        "partition_seed": None if partition_seed is None else int(partition_seed),
        "train_val_seed": CANONICAL_TRAIN_VAL_SEED,
        "train_indices": train_indices_full,
        "val_indices": val_indices_full,
        "client_indices": client_indices_full,
        "client_sample_counts": [len(indices) for indices in client_indices_full],
        "client_class_distribution": _build_client_class_distribution(client_indices_full, full_train_dataset.labels),
        "created_at": datetime.now().isoformat(),
    }
    _save_split_manifest(manifest_path, manifest_payload)
    print(f"[split] 已生成固定分片文件: {manifest_path}")
    return manifest_payload


def _load_or_create_split_manifest(data_root, n_clients, alpha, image_size, partition_seed):
    manifest_path = _default_split_manifest_path(n_clients=n_clients, alpha=alpha, partition_seed=partition_seed)
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_payload = json.load(f)
        if (
            manifest_payload.get("manifest_version") == SPLIT_MANIFEST_VERSION
            and int(manifest_payload.get("n_clients", -1)) == int(n_clients)
            and float(manifest_payload.get("alpha", -1.0)) == float(alpha)
            and manifest_payload.get("partition_seed") == (None if partition_seed is None else int(partition_seed))
        ):
            print(f"[split] 使用固定分片文件: {manifest_path}")
            return manifest_payload
        print(f"[split] 分片文件版本或配置不匹配，重新生成: {manifest_path}")
    return _generate_split_manifest(
        data_root=data_root,
        n_clients=n_clients,
        alpha=alpha,
        image_size=image_size,
        partition_seed=partition_seed,
        manifest_path=manifest_path,
    )

class PlanktonDataset(Dataset):
    """浮游生物数据集类 - 支持缓存加速"""

    def __init__(self, root_dir, transform=None, mode='train', client_id=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.client_id = client_id
        self.image_paths = []
        self.labels = []

        # ============================================================
        # 1. 确定当前模式对应的数据目录
        # ============================================================
        if mode == 'train':
            data_dir = os.path.join(root_dir, 'D_ID_train')
        elif mode == 'test':
            data_dir = os.path.join(root_dir, 'D_ID_test')
        elif mode == 'near_ood':
            data_dir = os.path.join(root_dir, 'D_Near_test')
        elif mode == 'far_ood':
            data_dir = os.path.join(root_dir, 'D_Far_test')
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ============================================================
        # 2. 建立标签映射 (保留原有逻辑)
        # ============================================================
        base_train_dir = os.path.join(root_dir, 'D_ID_train')
        self.class_to_idx = {}  # 初始化为空字典
        if os.path.exists(base_train_dir):
            id_dirs = sorted([
                d for d in os.listdir(base_train_dir)
                if os.path.isdir(os.path.join(base_train_dir, d))
            ])
            self.class_to_idx = {dirname: idx for idx, dirname in enumerate(id_dirs)}
        else:
            print(f"Warning: Base train dir {base_train_dir} not found for label mapping")

        # ============================================================
        # 3. 加载图像数据 (改进的缓存逻辑)
        # ============================================================
        # 定义缓存文件路径，例如: ./Plankton_OOD_Dataset/cache_train.pkl
        cache_file = os.path.join(root_dir, f"cache_{mode}.pkl")
        use_cache = False

        # ============================================================
        # 改进的缓存逻辑：只有当缓存"新鲜"时才使用
        # ============================================================
        if os.path.exists(cache_file):
            print(f"[{mode}] 发现缓存文件: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            cache_version_ok = cache_data.get('cache_format_version') == DATA_CACHE_FORMAT_VERSION
            cache_paths = cache_data.get('paths', [])
            if cache_version_ok and len(cache_paths) > 0 and \
               os.path.exists(cache_paths[0]) and \
               os.path.exists(cache_paths[-1]):

                self.image_paths = cache_paths
                self.labels = cache_data['labels']
                use_cache = True
                print(f"[{mode}] 缓存校验通过，已加载 {len(self.image_paths)} 张图片。")
            else:
                print(f"[{mode}] 缓存文件版本或路径已失效，将重新扫描...")
                use_cache = False

        if not use_cache:
            # 未命中缓存：执行原来的扫描逻辑
            print(f"[{mode}] 扫描文件 (这可能需要一些时间)...")
            if os.path.exists(data_dir):
                for dir_name in sorted(os.listdir(data_dir)):
                    class_dir = os.path.join(data_dir, dir_name)
                    if not os.path.isdir(class_dir):
                        continue

                    # 确定标签
                    current_label = -1
                    if mode in ['train', 'test']:
                        if dir_name in self.class_to_idx:
                            current_label = self.class_to_idx[dir_name]
                        else:
                            continue
                    else:
                        current_label = -1

                    # 加载图片
                    for img_name in sorted(os.listdir(class_dir)):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                            img_path = os.path.join(class_dir, img_name)
                            self.image_paths.append(img_path)
                            self.labels.append(current_label)

                # 扫描完成后，保存到缓存
                print(f"[{mode}] 保存扫描结果到 {cache_file}...")
                with open(cache_file, 'wb') as f:
                    # 建议：可以在缓存里多存一些元数据，比如生成时间
                    pickle.dump({
                        'paths': self.image_paths,
                        'labels': self.labels,
                        'cache_format_version': DATA_CACHE_FORMAT_VERSION,
                        'timestamp': datetime.now().isoformat()
                    }, f)
            else:
                print(f"[{mode}] 警告: 数据目录 {data_dir} 不存在")

        print(f"[{mode}] 已加载 {len(self.image_paths)} 张图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=320):
    """
    获取图像变换

    Args:
        image_size: 目标图像尺寸

    Returns:
        train_transform: 训练集变换
        test_transform: 测试集变换
    """
    # 训练集变换 - 包含数据增强
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 测试集变换 - 不包含数据增强
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def partition_data(dataset, n_clients=10, alpha=0.1, seed=None):
    """
    使用狄利克雷分布划分数据到多个客户端

    Args:
        dataset: 数据集对象
        n_clients: 客户端数量
        alpha: 狄利克雷分布参数，控制数据异质性

    Returns:
        client_indices: 每个客户端的数据索引列表
    """
    n_classes = len(ID_CLASSES)

    # 按类别组织数据索引
    class_indices = {i: [] for i in range(n_classes)}

    # 优化：直接访问标签而不应用transform
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'labels'):
        # 处理 random_split 返回的 Subset
        base_labels = dataset.dataset.labels
        # 使用子集索引（0 到 len(dataset)-1）
        for subset_idx, base_idx in enumerate(dataset.indices):
            label = base_labels[base_idx]
            if label != -1:
                class_indices[label].append(subset_idx)
    elif hasattr(dataset, 'labels'):
        # 直接访问标签
        for idx, label in enumerate(dataset.labels):
            if label != -1:
                class_indices[label].append(idx)
    else:
        # 回退到原始方法（慢）
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label != -1:
                class_indices[label].append(idx)

    # 使用狄利克雷分布生成客户端数据分布
    client_indices = [[] for _ in range(n_clients)]

    rng = np.random.default_rng(seed)

    for class_idx in range(n_classes):
        # 为每个类别生成客户端分布
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))

        # 获取当前类别的所有索引
        class_data = np.array(class_indices[class_idx], dtype=int)
        if len(class_data) > 0:
            class_data = rng.permutation(class_data)

        # 根据比例分配数据到客户端
        proportions = (proportions * len(class_data)).astype(int)
        proportions[-1] = len(class_data) - np.sum(proportions[:-1])

        start_idx = 0
        for client_id in range(n_clients):
            end_idx = start_idx + proportions[client_id]
            client_indices[client_id].extend(class_data[start_idx:end_idx].tolist())
            start_idx = end_idx

    # 打乱每个客户端的数据
    for client_id in range(n_clients):
        if client_indices[client_id]:
            client_indices[client_id] = rng.permutation(np.array(client_indices[client_id], dtype=int)).tolist()

    return client_indices


def get_recommended_num_workers(max_workers=8):
    """基于 CPU 核心数给出保守的 DataLoader worker 建议值。"""
    cpu_count = os.cpu_count() or 4
    return max(1, min(max_workers, max(1, cpu_count // 4)))


def _build_loader_kwargs(batch_size, shuffle, num_workers, drop_last=False):
    """统一 DataLoader 配置，避免不同入口的加载器参数漂移。"""
    if num_workers is None:
        num_workers = get_recommended_num_workers()

    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'drop_last': drop_last,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    return kwargs


def _create_federated_train_subsets(data_root, n_clients=10, alpha=0.1, image_size=320, partition_seed=None):
    """Create the train split, validation split, and per-client indices once."""
    train_transform, test_transform = get_transforms(image_size)
    manifest_payload = _load_or_create_split_manifest(
        data_root=data_root,
        n_clients=n_clients,
        alpha=alpha,
        image_size=image_size,
        partition_seed=partition_seed,
    )

    train_indices_full = [int(idx) for idx in manifest_payload["train_indices"]]
    val_indices_full = [int(idx) for idx in manifest_payload["val_indices"]]

    full_train_dataset = PlanktonDataset(data_root, transform=train_transform, mode='train')
    full_train_eval_dataset = PlanktonDataset(data_root, transform=test_transform, mode='train')

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices_full)
    val_dataset = torch.utils.data.Subset(full_train_eval_dataset, val_indices_full)

    full_to_train_idx = {full_idx: rel_idx for rel_idx, full_idx in enumerate(train_indices_full)}
    client_indices = [
        [full_to_train_idx[int(full_idx)] for full_idx in client_full_indices]
        for client_full_indices in manifest_payload["client_indices"]
    ]
    return train_dataset, val_dataset, client_indices


def create_federated_loaders(data_root, n_clients=10, alpha=0.1, batch_size=32, image_size=320, num_workers=None, partition_seed=None):
    """
    创建联邦学习数据加载器

    Args:
        data_root: 数据根目录
        n_clients: 客户端数量
        alpha: 狄利克雷分布参数
        batch_size: 批次大小
        image_size: 图像尺寸

    Returns:
        client_loaders: 客户端数据加载器列表
        test_loader: 测试数据加载器
        near_ood_loader: Near-OOD数据加载器
        far_ood_loader: Far-OOD数据加载器
        val_loader: 服务端验证数据加载器 (新增)
    """
    _, test_transform = get_transforms(image_size)
    train_dataset, val_dataset, client_indices = _create_federated_train_subsets(
        data_root=data_root,
        n_clients=n_clients,
        alpha=alpha,
        image_size=image_size,
        partition_seed=partition_seed,
    )

    # 创建客户端数据加载器
    client_loaders = []
    for client_id in range(n_clients):
        client_dataset = torch.utils.data.Subset(train_dataset, client_indices[client_id])
        client_loader = DataLoader(
            client_dataset,
            **_build_loader_kwargs(
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
            )
        )
        client_loaders.append(client_loader)
        print(f"客户端 {client_id}: {len(client_dataset)} 样本")

    # 创建测试和OOD数据加载器
    test_dataset = PlanktonDataset(data_root, transform=test_transform, mode='test')
    near_ood_dataset = PlanktonDataset(data_root, transform=test_transform, mode='near_ood')
    far_ood_dataset = PlanktonDataset(data_root, transform=test_transform, mode='far_ood')

    test_loader = DataLoader(
        test_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )
    near_ood_loader = DataLoader(
        near_ood_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )
    far_ood_loader = DataLoader(
        far_ood_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

    # 5. 创建验证集 Loader (用于服务端挑选最佳模型)
    val_loader = DataLoader(
        val_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

    train_len = len(train_dataset)
    val_len = len(val_dataset)

    print(f"联邦学习数据加载器创建完成:")
    print(f"  - 客户端数量: {len(client_loaders)}")
    print(f"  - 训练集 (联邦): {train_len} 样本")
    print(f"  - 验证集 (服务端): {val_len} 样本")
    print(f"  - 测试集: {len(test_dataset)} 样本")
    print(f"  - Near-OOD: {len(near_ood_dataset)} 样本")
    print(f"  - Far-OOD: {len(far_ood_dataset)} 样本")

    return client_loaders, test_loader, near_ood_loader, far_ood_loader, val_loader


def create_id_train_client_loaders_only(data_root, n_clients=10, alpha=0.1, batch_size=32, image_size=320, num_workers=None, partition_seed=None):
    """
    Create per-client ID-train loaders with test-time transforms only.
    Used for federated empirical alpha calibration without data augmentation.
    """
    _, test_transform = get_transforms(image_size)
    train_dataset, _, client_indices = _create_federated_train_subsets(
        data_root=data_root,
        n_clients=n_clients,
        alpha=alpha,
        image_size=image_size,
        partition_seed=partition_seed,
    )

    base_dataset = PlanktonDataset(data_root, transform=test_transform, mode='train')
    train_subset_indices = train_dataset.indices
    client_loaders = []

    for client_id in range(n_clients):
        base_indices = [train_subset_indices[idx] for idx in client_indices[client_id]]
        client_dataset = torch.utils.data.Subset(base_dataset, base_indices)
        client_loader = DataLoader(
            client_dataset,
            **_build_loader_kwargs(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
            )
        )
        client_loaders.append(client_loader)

    print("客户端 ID 训练集 Alpha 加载器创建完成:")
    print(f"  - 客户端数量: {len(client_loaders)}")
    return client_loaders


def create_id_train_pooled_loader_only(data_root, n_clients=10, alpha=0.1, batch_size=32, image_size=320, num_workers=None, partition_seed=None):
    """
    Create one pooled ID-train loader using the same 90% train split as federated runs.

    This keeps the sample scope aligned with FedViM:
    - the same random 90% training subset from D_ID_train
    - test-time transforms only
    - no validation samples included
    """
    _, test_transform = get_transforms(image_size)
    train_dataset, _, _ = _create_federated_train_subsets(
        data_root=data_root,
        n_clients=n_clients,
        alpha=alpha,
        image_size=image_size,
        partition_seed=partition_seed,
    )

    base_dataset = PlanktonDataset(data_root, transform=test_transform, mode='train')
    pooled_dataset = torch.utils.data.Subset(base_dataset, train_dataset.indices)
    pooled_loader = DataLoader(
        pooled_dataset,
        **_build_loader_kwargs(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    )

    print("Pooled ID 训练集加载器创建完成:")
    print(f"  - 使用与联邦训练一致的训练子集: {len(pooled_dataset)} 样本")
    return pooled_loader


def create_test_loaders_only(data_root, batch_size=32, image_size=320, num_workers=None):
    """
    只创建测试数据加载器（不创建训练数据加载器）
    用于测试脚本，避免加载训练数据

    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        image_size: 图像尺寸

    Returns:
        test_loader: 测试数据加载器
        near_ood_loader: Near-OOD数据加载器
        far_ood_loader: Far-OOD数据加载器
    """
    _, test_transform = get_transforms(image_size)

    # 创建测试和OOD数据集
    test_dataset = PlanktonDataset(data_root, transform=test_transform, mode='test')
    near_ood_dataset = PlanktonDataset(data_root, transform=test_transform, mode='near_ood')
    far_ood_dataset = PlanktonDataset(data_root, transform=test_transform, mode='far_ood')

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

    near_ood_loader = DataLoader(
        near_ood_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

    far_ood_loader = DataLoader(
        far_ood_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

    print(f"测试数据加载器创建完成:")
    print(f"  - 测试集: {len(test_dataset)} 样本")
    print(f"  - Near-OOD: {len(near_ood_dataset)} 样本")
    print(f"  - Far-OOD: {len(far_ood_dataset)} 样本")

    return test_loader, near_ood_loader, far_ood_loader


def create_id_train_loader_only(data_root, batch_size=32, image_size=320, num_workers=None):
    """
    只创建 ID 训练集 loader，默认使用无增强的 test transform。
    用于离线评估或经验 alpha 校准。
    """
    _, test_transform = get_transforms(image_size)

    train_dataset = PlanktonDataset(data_root, transform=test_transform, mode='train')
    train_loader = DataLoader(
        train_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

    print("训练集数据加载器创建完成:")
    print(f"  - ID Train: {len(train_dataset)} 样本")

    return train_loader


if __name__ == "__main__":
    # 测试数据加载器
    data_root = "./data"

    try:
        # [修改] 接收 5 个返回值，用 _ 忽略 val_loader
        client_loaders, test_loader, near_ood_loader, far_ood_loader, _ = create_federated_loaders(
            data_root, n_clients=3, batch_size=4, image_size=320
        )

        # 测试一个批次
        for client_id, loader in enumerate(client_loaders):
            for images, labels in loader:
                print(f"客户端 {client_id} - 批次图像尺寸: {images.shape}")
                print(f"客户端 {client_id} - 批次标签: {labels}")
                break

    except Exception as e:
        print(f"数据加载测试失败: {e}")
        print("请确保数据集已正确划分并放置在指定目录")
