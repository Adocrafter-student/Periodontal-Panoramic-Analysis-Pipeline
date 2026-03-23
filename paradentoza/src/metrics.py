"""
Evaluation metrics: confusion matrix, per-class precision/recall/F1, AUC-ROC.
CSV epoch logging and JSON result serialisation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def compute_metrics(
    targets: List[int],
    preds: List[int],
    probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute full classification metrics via scikit-learn."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )

    t = np.array(targets)
    p = np.array(preds)
    num_classes = probs.shape[1]
    labels = list(range(num_classes))

    acc = accuracy_score(t, p)
    cm = confusion_matrix(t, p, labels=labels)
    report = classification_report(
        t, p,
        target_names=class_names,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    result: Dict[str, Any] = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    try:
        if num_classes == 2:
            result["auc_roc"] = float(roc_auc_score(t, probs[:, 1]))
        else:
            result["auc_roc"] = float(
                roc_auc_score(t, probs, multi_class="ovr", average="macro")
            )
    except ValueError:
        result["auc_roc"] = None

    return result


def print_metrics(
    metrics: Dict[str, Any],
    class_names: List[str],
    fold: int | None = None,
) -> None:
    header = f"Fold {fold}" if fold is not None else "Evaluation"
    print(f"\n{'=' * 60}")
    print(f"  {header} Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    if metrics.get("auc_roc") is not None:
        print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")

    report = metrics["classification_report"]
    print(f"\n  {'Class':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>6}")
    print(f"  {'-' * 44}")
    for name in class_names:
        if name in report:
            r = report[name]
            print(
                f"  {name:<20} {r['precision']:>6.3f} {r['recall']:>6.3f} "
                f"{r['f1-score']:>6.3f} {r['support']:>6.0f}"
            )
    if "macro avg" in report:
        r = report["macro avg"]
        print(f"  {'-' * 44}")
        print(
            f"  {'macro avg':<20} {r['precision']:>6.3f} {r['recall']:>6.3f} "
            f"{r['f1-score']:>6.3f} {r['support']:>6.0f}"
        )

    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print("  " + " " * 14 + "".join(f"{n[:10]:>12}" for n in class_names))
    for i, row in enumerate(cm):
        name = class_names[i] if i < len(class_names) else str(i)
        print("  " + f"{name[:12]:<14}" + "".join(f"{v:>12}" for v in row))

    print(f"{'=' * 60}")


def aggregate_fold_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    accs = [m["accuracy"] for m in fold_metrics]
    aucs = [m["auc_roc"] for m in fold_metrics if m.get("auc_roc") is not None]

    result: Dict[str, Any] = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "per_fold_accuracy": accs,
    }
    if aucs:
        result["auc_roc_mean"] = float(np.mean(aucs))
        result["auc_roc_std"] = float(np.std(aucs))
        result["per_fold_auc_roc"] = aucs

    # Aggregate per-class F1 across folds
    all_reports = [m["classification_report"] for m in fold_metrics]
    class_keys = [
        k for k in all_reports[0]
        if k not in ("accuracy", "macro avg", "weighted avg")
    ]
    per_class_f1: Dict[str, Dict[str, float]] = {}
    for cls in class_keys:
        f1s = [r[cls]["f1-score"] for r in all_reports if cls in r]
        per_class_f1[cls] = {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        }
    result["per_class_f1"] = per_class_f1

    return result


def save_metrics_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class EpochLogger:
    """Log per-epoch metrics to CSV."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer: csv.DictWriter | None = None

    def log(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
