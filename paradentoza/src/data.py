from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import BINARY_DISEASE, BINARY_HEALTHY, IMAGE_EXTENSIONS, STAGE_DIRS

DEFAULT_IMG_HEIGHT = 224
DEFAULT_IMG_WIDTH = 448


def list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class PeriodontalDataset(Dataset):
    """Unified dataset for both binary and stage classification."""

    def __init__(
        self,
        paths: List[Path],
        labels: List[int],
        transform: Callable | None = None,
        class_names: List[str] | None = None,
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or []

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    @property
    def num_classes(self) -> int:
        return len(set(self.labels)) if self.labels else 0

    @staticmethod
    def binary_from_folders(
        healthy_dir: Path = BINARY_HEALTHY,
        disease_dir: Path = BINARY_DISEASE,
        seed: int = 42,
    ) -> PeriodontalDataset:
        rng = random.Random(seed)
        h = list_images(healthy_dir)
        d = list_images(disease_dir)
        paths = h + d
        labels = [0] * len(h) + [1] * len(d)
        combined = list(zip(paths, labels))
        rng.shuffle(combined)
        if combined:
            paths_s, labels_s = map(list, zip(*combined))
        else:
            paths_s, labels_s = [], []
        return PeriodontalDataset(
            paths_s, labels_s,
            class_names=["non_periodontal", "periodontal"],
        )

    @staticmethod
    def stage_from_folders(
        stage_dirs: List[Path] | None = None,
        seed: int = 42,
    ) -> PeriodontalDataset:
        rng = random.Random(seed)
        dirs = stage_dirs or STAGE_DIRS
        paths: List[Path] = []
        labels: List[int] = []
        for stage_idx, folder in enumerate(dirs):
            for p in list_images(folder):
                paths.append(p)
                labels.append(stage_idx)
        combined = list(zip(paths, labels))
        rng.shuffle(combined)
        if combined:
            paths_s, labels_s = map(list, zip(*combined))
        else:
            paths_s, labels_s = [], []
        names = [f"stage_{i + 1}" for i in range(len(dirs))]
        return PeriodontalDataset(paths_s, labels_s, class_names=names)


def stratified_kfold_split(
    paths: List[Path],
    labels: List[int],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Stratified k-fold indices. Returns list of (train_indices, val_indices)."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)

    for indices in by_class.values():
        rng.shuffle(indices)

    folds: list[tuple[list[int], list[int]]] = []
    for fold_idx in range(n_folds):
        train_idx, val_idx = [], []
        for _y, indices in by_class.items():
            n = len(indices)
            fold_start = int(round(n * fold_idx / n_folds))
            fold_end = int(round(n * (fold_idx + 1) / n_folds))
            for i, idx in enumerate(indices):
                if fold_start <= i < fold_end:
                    val_idx.append(idx)
                else:
                    train_idx.append(idx)
        rng_fold = random.Random(seed + fold_idx)
        rng_fold.shuffle(train_idx)
        rng_fold.shuffle(val_idx)
        folds.append((train_idx, val_idx))

    return folds


def train_val_split(
    paths: List[Path],
    labels: List[int],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Single stratified split. Returns (train_indices, val_indices)."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for _y, indices in by_class.items():
        indices = indices.copy()
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 0
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def get_transforms(
    img_height: int = DEFAULT_IMG_HEIGHT,
    img_width: int = DEFAULT_IMG_WIDTH,
    train: bool = True,
) -> transforms.Compose:
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05),
                scale=(0.9, 1.1), shear=5,
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        norm,
    ])
