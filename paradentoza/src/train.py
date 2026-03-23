"""
Train binary (healthy vs periodontal) or 4-class stage classifier.

Supports stratified k-fold cross-validation, early stopping, LR scheduling,
and full evaluation metrics (confusion matrix, precision/recall/F1, AUC-ROC).

Examples:
  python -m src.train --task binary --epochs 30 --folds 5
  python -m src.train --task stage  --epochs 40 --folds 5 --patience 8
  python -m src.train --task binary --epochs 30 --folds 0   # single split
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CHECKPOINTS_DIR, PROJECT_ROOT
from .data import (
    DEFAULT_IMG_HEIGHT,
    DEFAULT_IMG_WIDTH,
    PeriodontalDataset,
    get_transforms,
    stratified_kfold_split,
    train_val_split,
)
from .metrics import (
    EpochLogger,
    aggregate_fold_metrics,
    compute_metrics,
    print_metrics,
    save_metrics_json,
)
from .model import build_classifier

RESULTS_DIR = PROJECT_ROOT / "results"


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], np.ndarray]:
    """Run model on loader, return (targets, preds, prob_matrix)."""
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_targets.extend(y.tolist())
            all_preds.extend(preds)
            all_probs.append(probs)
    return all_targets, all_preds, np.concatenate(all_probs, axis=0)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    train: bool = True,
) -> tuple[float, float]:
    """Single training/validation epoch. Returns (loss, accuracy)."""
    model.train() if train else model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(1) == y).sum().item()
            n += bs
    return total_loss / max(n, 1), total_correct / max(n, 1)


def _train_fold(
    fold_idx: int | None,
    train_ds: PeriodontalDataset,
    val_ds: PeriodontalDataset,
    num_classes: int,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, dict]:
    """Train one fold. Returns (best_model_state_dict, fold_metrics)."""
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = build_classifier(num_classes, pretrained=not args.no_pretrained).to(device)

    class_counts = Counter(train_ds.labels)
    total = sum(class_counts.values())
    weights = torch.tensor(
        [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=max(3, args.patience // 2),
    )

    fold_label = f"fold {fold_idx}" if fold_idx is not None else "single split"
    log_name = f"fold_{fold_idx}.csv" if fold_idx is not None else "training.csv"
    logger = EpochLogger(RESULTS_DIR / args.task / log_name)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = _run_epoch(
            model, train_loader, device, criterion, optimizer, train=True,
        )
        va_loss, va_acc = _run_epoch(
            model, val_loader, device, criterion, train=False,
        )
        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]["lr"]

        logger.log({
            "epoch": epoch,
            "train_loss": f"{tr_loss:.4f}",
            "train_acc": f"{tr_acc:.4f}",
            "val_loss": f"{va_loss:.4f}",
            "val_acc": f"{va_acc:.4f}",
            "lr": f"{lr:.6f}",
        })

        print(
            f"  [{fold_label}] epoch {epoch:03d}  "
            f"train {tr_loss:.4f}/{tr_acc:.4f}  "
            f"val {va_loss:.4f}/{va_acc:.4f}  "
            f"lr {lr:.1e}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"  [{fold_label}] early stopping at epoch {epoch} "
                    f"(patience {args.patience})"
                )
                break

    logger.close()

    model.load_state_dict(best_state)
    targets, preds, probs = _collect_predictions(model, val_loader, device)
    metrics = compute_metrics(targets, preds, probs, class_names)
    print_metrics(metrics, class_names, fold=fold_idx)

    return best_state, metrics


def main():
    p = argparse.ArgumentParser(
        description="Train periodontal classifier with optional k-fold CV",
    )
    p.add_argument("--task", choices=["binary", "stage"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-height", type=int, default=DEFAULT_IMG_HEIGHT)
    p.add_argument("--img-width", type=int, default=DEFAULT_IMG_WIDTH)
    p.add_argument(
        "--folds", type=int, default=5,
        help="K-fold CV folds; 0 or 1 = single train/val split",
    )
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Val ratio when --folds 0")
    p.add_argument("--patience", type=int, default=7,
                   help="Early stopping patience (epochs without val loss improvement)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS_DIR)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # --- build dataset ---
    if args.task == "binary":
        full_ds = PeriodontalDataset.binary_from_folders(seed=args.seed)
        out_name = "binary_resnet18.pt"
    else:
        full_ds = PeriodontalDataset.stage_from_folders(seed=args.seed)
        out_name = "stage_resnet18.pt"

    num_classes = full_ds.num_classes
    class_names = full_ds.class_names
    print(f"task: {args.task}  classes: {num_classes}  images: {len(full_ds)}")

    counts = Counter(full_ds.labels)
    for idx in sorted(counts):
        name = class_names[idx] if idx < len(class_names) else str(idx)
        print(f"  {name}: {counts[idx]} images")

    train_tf = get_transforms(args.img_height, args.img_width, train=True)
    val_tf = get_transforms(args.img_height, args.img_width, train=False)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / args.task).mkdir(parents=True, exist_ok=True)

    all_fold_metrics: list[dict] = []
    best_overall_state = None
    best_overall_acc = -1.0

    if args.folds >= 2:
        # ---- k-fold cross-validation ----
        folds = stratified_kfold_split(
            full_ds.paths, full_ds.labels,
            n_folds=args.folds, seed=args.seed,
        )
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n{'=' * 60}")
            print(f"  FOLD {fold_idx + 1} / {args.folds}  "
                  f"(train {len(train_idx)}, val {len(val_idx)})")
            print(f"{'=' * 60}")

            torch.manual_seed(args.seed + fold_idx)

            train_ds = PeriodontalDataset(
                [full_ds.paths[i] for i in train_idx],
                [full_ds.labels[i] for i in train_idx],
                transform=train_tf, class_names=class_names,
            )
            val_ds = PeriodontalDataset(
                [full_ds.paths[i] for i in val_idx],
                [full_ds.labels[i] for i in val_idx],
                transform=val_tf, class_names=class_names,
            )

            state, metrics = _train_fold(
                fold_idx, train_ds, val_ds,
                num_classes, class_names, args, device,
            )
            all_fold_metrics.append(metrics)

            if metrics["accuracy"] > best_overall_acc:
                best_overall_acc = metrics["accuracy"]
                best_overall_state = state

        # ---- aggregate ----
        agg = aggregate_fold_metrics(all_fold_metrics)
        print(f"\n{'=' * 60}")
        print(f"  CROSS-VALIDATION SUMMARY ({args.folds} folds)")
        print(f"{'=' * 60}")
        print(f"  Accuracy : {agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.4f}")
        if "auc_roc_mean" in agg:
            print(f"  AUC-ROC  : {agg['auc_roc_mean']:.4f} +/- {agg['auc_roc_std']:.4f}")
        per_class = agg.get("per_class_f1", {})
        if per_class:
            print(f"\n  {'Class':<20} {'F1 mean':>8} {'F1 std':>8}")
            print(f"  {'-' * 36}")
            for cls, vals in per_class.items():
                print(f"  {cls:<20} {vals['f1_mean']:>8.4f} {vals['f1_std']:>8.4f}")
        print(f"  Per-fold acc: {[f'{a:.4f}' for a in agg['per_fold_accuracy']]}")
        print(f"{'=' * 60}")

        save_metrics_json(agg, RESULTS_DIR / args.task / "cv_summary.json")
        save_metrics_json(all_fold_metrics, RESULTS_DIR / args.task / "fold_details.json")

    else:
        # ---- single split ----
        train_idx, val_idx = train_val_split(
            full_ds.paths, full_ds.labels,
            val_ratio=args.val_ratio, seed=args.seed,
        )

        train_ds = PeriodontalDataset(
            [full_ds.paths[i] for i in train_idx],
            [full_ds.labels[i] for i in train_idx],
            transform=train_tf, class_names=class_names,
        )
        val_ds = PeriodontalDataset(
            [full_ds.paths[i] for i in val_idx],
            [full_ds.labels[i] for i in val_idx],
            transform=val_tf, class_names=class_names,
        )

        print(f"\nSingle split: train {len(train_ds)}, val {len(val_ds)}")
        best_overall_state, metrics = _train_fold(
            None, train_ds, val_ds,
            num_classes, class_names, args, device,
        )
        save_metrics_json(metrics, RESULTS_DIR / args.task / "eval_metrics.json")

    # ---- save checkpoint ----
    best_path = args.checkpoint_dir / out_name
    torch.save(
        {
            "task": args.task,
            "num_classes": num_classes,
            "class_names": class_names,
            "model_state": best_overall_state,
            "img_height": args.img_height,
            "img_width": args.img_width,
            "img_size": args.img_height,  # backward compat with old pipeline
        },
        best_path,
    )
    print(f"\nbest checkpoint saved: {best_path}")


if __name__ == "__main__":
    main()
