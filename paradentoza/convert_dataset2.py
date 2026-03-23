"""
Convert dataset2 (YOLO object-detection format) into a classification dataset
compatible with the existing periodontal staging pipeline.

dataset2 annotates dental panoramic X-rays with bounding boxes for tooth-loss
patterns and pathological findings:

    0: Broken_Root              Retained root fragments
    1: PCT                      Pericoronitis
    2: Free_R_Max               Free-end edentulous area, right maxilla
    3: Free_L_Max               Free-end edentulous area, left maxilla
    4: Not_Free_Max             Bounded edentulous area, maxilla
    5: Not_Free_Center_Max      Bounded edentulous area, center maxilla
    6: Free_R_Mand              Free-end edentulous area, right mandible
    7: Free_L_Mand              Free-end edentulous area, left mandible
    8: Not_Free_Mand            Bounded edentulous area, mandible
    9: Not_Free_Center_Mand     Bounded edentulous area, center mandible

"Free-end" (Kennedy I/II) = posterior teeth lost with no distal abutment.
"Not_Free" / bounded (Kennedy III) = teeth lost between remaining teeth.

Mapping rationale (2017 periodontal staging approximation):
    - Free-end tooth loss indicates more advanced disease (posterior support lost).
    - The NUMBER of affected regions and co-occurrence of Broken_Root / PCT
      approximate clinical severity.

Binary:   non_periodontal (0) vs periodontal (1)
Staging:  stage_1 .. stage_4

Run:
    python convert_dataset2.py                         # analyse only
    python convert_dataset2.py --copy                  # analyse + copy files
    python convert_dataset2.py --copy --dest dataset   # custom output root
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

FREE_CLASSES = {2, 3, 6, 7}
NOT_FREE_CLASSES = {4, 5, 8, 9}
BROKEN_ROOT = 0
PCT = 1

CLASS_NAMES = [
    "Broken_Root",
    "PCT",
    "Free_R_Max",
    "Free_L_Max",
    "Not_Free_Max",
    "Not_Free_Center_Max",
    "Free_R_Mand",
    "Free_L_Mand",
    "Not_Free_Mand",
    "Not_Free_Center_Mand",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_yolo_label(label_path: Path) -> list[int]:
    """Return list of class IDs present in a YOLO label file."""
    classes: list[int] = []
    text = label_path.read_text().strip()
    if not text:
        return classes
    for line in text.splitlines():
        parts = line.strip().split()
        if parts:
            classes.append(int(parts[0]))
    return classes


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    """Find the image file that corresponds to a label file."""
    stem = label_path.stem
    for ext in IMAGE_EXTS:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def derive_binary_label(classes: list[int]) -> int:
    """0 = non-periodontal, 1 = periodontal."""
    if not classes:
        return 0
    return 1


def derive_stage_label(classes: list[int]) -> int:
    """
    Approximate 2017 periodontal stage (0-indexed: 0=stage1 .. 3=stage4).

    Logic:
      - No annotations or only Not_Free_Center → Stage 1 (minimal findings)
      - Not_Free regions only (bounded tooth loss, no free-end) → Stage 2
      - Any Free_* class present (free-end tooth loss) → Stage 3
      - Multiple Free_* region types OR Free_* + multiple Broken_Root → Stage 4
    """
    if not classes:
        return 0

    class_set = set(classes)
    class_counts = Counter(classes)

    free_types = class_set & FREE_CLASSES
    not_free_types = class_set & NOT_FREE_CLASSES
    has_broken_root = BROKEN_ROOT in class_set
    has_pct = PCT in class_set
    n_broken_roots = class_counts.get(BROKEN_ROOT, 0)
    n_free_types = len(free_types)

    if n_free_types >= 3 or (n_free_types >= 2 and n_broken_roots >= 2):
        return 3  # Stage 4: extensive free-end loss + broken roots
    if n_free_types >= 1:
        return 2  # Stage 3: free-end edentulous area(s) present
    if not_free_types or has_broken_root or has_pct:
        return 1  # Stage 2: bounded tooth loss / root fragments / PCT
    return 0      # Stage 1: minimal findings


def scan_split(split_dir: Path) -> list[dict]:
    """Scan one split (train/valid/test) and return per-image records."""
    labels_dir = split_dir / "labels"
    images_dir = split_dir / "images"
    records = []

    if not labels_dir.is_dir():
        return records

    for lbl in sorted(labels_dir.glob("*.txt")):
        img = find_image_for_label(lbl, images_dir)
        if img is None:
            continue
        classes = parse_yolo_label(lbl)
        records.append({
            "image": img,
            "label_file": lbl,
            "classes": classes,
            "binary": derive_binary_label(classes),
            "stage": derive_stage_label(classes),
            "split": split_dir.name,
        })

    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl_candidate = labels_dir / (img_path.stem + ".txt")
        if not lbl_candidate.exists():
            records.append({
                "image": img_path,
                "label_file": None,
                "classes": [],
                "binary": 0,
                "stage": 0,
                "split": split_dir.name,
            })

    seen = set()
    unique = []
    for r in records:
        key = r["image"]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def print_analysis(records: list[dict]) -> None:
    """Print a comprehensive analysis of the derived labels."""
    print(f"\n{'=' * 70}")
    print(f"  DATASET2 ANALYSIS — {len(records)} images total")
    print(f"{'=' * 70}")

    # Per-split counts
    by_split = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)

    for split_name in ["train", "valid", "test"]:
        split_records = by_split.get(split_name, [])
        if not split_records:
            continue
        print(f"\n  [{split_name}] {len(split_records)} images")

        binary_counts = Counter(r["binary"] for r in split_records)
        print(f"    Binary:  non-periodontal={binary_counts.get(0, 0)}  "
              f"periodontal={binary_counts.get(1, 0)}")

        stage_counts = Counter(r["stage"] for r in split_records)
        for s in range(4):
            print(f"    Stage {s + 1}: {stage_counts.get(s, 0)} images")

    # Overall
    print(f"\n  {'—' * 50}")
    print("  OVERALL:")
    binary_counts = Counter(r["binary"] for r in records)
    print(f"    Binary:  non-periodontal={binary_counts.get(0, 0)}  "
          f"periodontal={binary_counts.get(1, 0)}")

    stage_counts = Counter(r["stage"] for r in records)
    for s in range(4):
        print(f"    Stage {s + 1}: {stage_counts.get(s, 0)} images")

    # Detection class frequency across images
    print(f"\n  Detection class presence (images containing each class):")
    class_image_counts = Counter()
    for r in records:
        for c in set(r["classes"]):
            class_image_counts[c] += 1
    for cid in range(10):
        name = CLASS_NAMES[cid]
        count = class_image_counts.get(cid, 0)
        print(f"    {cid}: {name:<24s} {count:>4d} images")

    empty = sum(1 for r in records if not r["classes"])
    print(f"\n    Images with NO annotations: {empty}")
    print(f"{'=' * 70}")


def copy_files(records: list[dict], dest_root: Path) -> None:
    """Copy images into classification folder structure."""

    binary_dirs = {
        0: dest_root / "dental-panoramic" / "penyakit-non-periodontal",
        1: dest_root / "dental-panoramic" / "penyakit-periodontal",
    }
    stage_dirs = {
        i: dest_root / f"stage_{i + 1}" for i in range(4)
    }

    for d in list(binary_dirs.values()) + list(stage_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    copied_binary = Counter()
    copied_stage = Counter()

    for r in records:
        src = r["image"]
        split_prefix = r["split"]
        new_name = f"ds2_{split_prefix}_{src.name}"

        b_dest = binary_dirs[r["binary"]] / new_name
        if not b_dest.exists():
            shutil.copy2(src, b_dest)
            copied_binary[r["binary"]] += 1

        s_dest = stage_dirs[r["stage"]] / new_name
        if not s_dest.exists():
            shutil.copy2(src, s_dest)
            copied_stage[r["stage"]] += 1

    print(f"\n  Files copied to {dest_root}:")
    for label, count in sorted(copied_binary.items()):
        tag = "non-periodontal" if label == 0 else "periodontal"
        print(f"    binary/{tag}: {count}")
    for stage, count in sorted(copied_stage.items()):
        print(f"    stage_{stage + 1}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse dataset2 and optionally convert to classification format",
    )
    parser.add_argument(
        "--dataset2", type=Path,
        default=Path(__file__).resolve().parent / "dataset2" / "Dataset",
        help="Path to dataset2/Dataset root",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy images into classification folder structure",
    )
    parser.add_argument(
        "--dest", type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Destination root for classification folders (default: dataset/)",
    )
    args = parser.parse_args()

    all_records: list[dict] = []
    for split_name in ["train", "valid", "test"]:
        split_dir = args.dataset2 / split_name
        if split_dir.is_dir():
            records = scan_split(split_dir)
            all_records.extend(records)

    print_analysis(all_records)

    if args.copy:
        copy_files(all_records, args.dest)
        print("\nDone!  You can now train with the merged dataset:")
        print("  python -m src.train --task binary --epochs 30 --folds 5")
        print("  python -m src.train --task stage  --epochs 40 --folds 5")
    else:
        print("\nRun with --copy to write files into classification folders:")
        print(f"  python convert_dataset2.py --copy --dest {args.dest}")


if __name__ == "__main__":
    main()
