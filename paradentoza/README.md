# Paradentoza — periodontal panoramic analysis pipeline

Classification of periodontal disease from dental panoramic radiographs (OPGs), aligned with the **2017 periodontal disease classification** (Stages I–IV).

The pipeline supports three tasks:

1. **Binary classification** — periodontal disease present vs absent.
2. **Multi-class staging** — assign one of four severity stages (I–IV) following the 2017 classification framework.
3. **Bone-loss radiographic proxy** — a heuristic index derived from image statistics (research aid, not a clinical measurement).

The classifier is **ResNet-18** with optional ImageNet pre-training, trained with PyTorch. Training uses **stratified k-fold cross-validation**, **class-weighted loss** for imbalanced data, **early stopping**, and **learning-rate scheduling**, with full evaluation metrics (confusion matrix, per-class precision/recall/F1, AUC-ROC).

---

## Datasets

This project combines two publicly available dental panoramic radiograph datasets.

### Dataset 1 — Dental Panoramic (Kaggle)

**Source**: <https://www.kaggle.com/datasets/kasmira/dentalpanoramic>

A small collection of dental panoramic X-rays pre-classified into two folders:

- `penyakit-non-periodontal` — 50 images (no periodontal disease)
- `penyakit-periodontal` — 42 images (periodontal disease present)

Additionally, a subset of periodontal images is manually sorted into four stage folders (`stage_1` through `stage_4`, 38 images total) for the multi-class staging task.

This dataset provides **ground-truth classification labels** but is limited in size (92 images for binary, 38 for staging), which results in high variance across cross-validation folds.

### Dataset 2 — OPG Kennedy Classification (Mendeley Data)

**Source**: Waqas, M., Hasan, S., Khurshid, Z., & Kazmi, S. (2024). *OPG Dataset for Kennedy Classification of Partially Edentulous Arches*. Mendeley Data, V1. <https://doi.org/10.17632/ccw5mvg69r.1>

A larger dataset of **622 dental panoramic radiographs** (498 train / 62 valid / 62 test) annotated in YOLO object-detection format with bounding boxes for 10 classes of dental findings:

| ID | Class | Description |
|----|-------|-------------|
| 0 | Broken_Root | Retained root fragments |
| 1 | PCT | Periodontally compromised tooth |
| 2 | Free_R_Max | Free-end edentulous area, right maxilla |
| 3 | Free_L_Max | Free-end edentulous area, left maxilla |
| 4 | Not_Free_Max | Bounded edentulous area, maxilla |
| 5 | Not_Free_Center_Max | Bounded edentulous area, center maxilla |
| 6 | Free_R_Mand | Free-end edentulous area, right mandible |
| 7 | Free_L_Mand | Free-end edentulous area, left mandible |
| 8 | Not_Free_Mand | Bounded edentulous area, mandible |
| 9 | Not_Free_Center_Mand | Bounded edentulous area, center mandible |

"Free-end" (Kennedy Class I/II) denotes posterior tooth loss with no distal abutment. "Not_Free" / bounded (Kennedy Class III) denotes tooth loss between remaining teeth.

Since this dataset uses **object-detection annotations** rather than image-level classification labels, we derive classification labels through a mapping process described below.

### Detection-to-classification mapping

The script `convert_dataset2.py` reads every YOLO label file and derives an image-level classification label based on the pathological findings present in each radiograph.

**Binary mapping:**

| Condition | Label |
|-----------|-------|
| No detection annotations (no visible pathology) | Non-periodontal (0) |
| Any pathological annotation present | Periodontal (1) |

**Staging mapping** (approximation of 2017 periodontal staging based on tooth-loss patterns):

| Derived stage | Rule | Rationale |
|---------------|------|-----------|
| **Stage I** | No annotations | No radiographically visible tooth loss or pathological findings |
| **Stage II** | Only bounded edentulous areas (Not_Free), broken roots, or PCT — no free-end loss | Moderate disease; tooth loss is bounded (Kennedy Class III), retained roots or periodontally compromised teeth present |
| **Stage III** | 1–2 types of free-end edentulous regions present | Severe disease; posterior teeth lost without distal abutment (Kennedy Class I/II), indicating significant periodontal breakdown |
| **Stage IV** | 3+ types of free-end regions, or free-end combined with multiple broken roots | Advanced disease; extensive tooth loss across multiple jaw regions with retained root fragments |

The clinical reasoning: free-end edentulous areas represent loss of posterior support, which correlates with advanced periodontal destruction. The number of affected jaw regions (right/left maxilla, right/left mandible) and co-occurrence of broken roots approximate overall disease severity.

**Note**: These derived labels are approximations. The mapping uses radiographic tooth-loss patterns as a proxy for periodontal staging — it is not equivalent to clinical staging which also considers probing depths, clinical attachment loss, and other factors. This limitation should be documented in any academic work using this pipeline.

### Merged dataset summary

| Task | Dataset 1 only | After merging Dataset 2 |
|------|---------------|------------------------|
| Binary | 92 images (50 + 42) | **714 images** (190 + 524) |
| Staging | 38 images (9 + 9 + 11 + 9) | **660 images** (149 + 294 + 134 + 83) |

---

## Requirements

- Python 3.10+ recommended
- [CUDA](https://pytorch.org/get-started/locally/) optional (training and inference use GPU when available)

Install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

---

## Dataset preparation

### Step 1 — Place Dataset 1

Download from [Kaggle](https://www.kaggle.com/datasets/kasmira/dentalpanoramic) and arrange into:

```
dataset/
├── dental-panoramic/
│   ├── penyakit-non-periodontal/   # healthy OPGs
│   └── penyakit-periodontal/       # periodontal OPGs
├── stage_1/
├── stage_2/
├── stage_3/
└── stage_4/
```

### Step 2 — Place Dataset 2

Download from [Mendeley Data](https://doi.org/10.17632/ccw5mvg69r.1) and extract into `dataset2/Dataset/` so the structure is:

```
dataset2/
└── Dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    ├── data.yaml
    └── names.txt
```

### Step 3 — Convert and merge

Run the conversion script to analyse Dataset 2 and copy images into the classification folder structure:

```bash
python convert_dataset2.py          # analysis only (prints statistics)
python convert_dataset2.py --copy   # analysis + copy into dataset/
```

Converted images are prefixed with `ds2_` to avoid filename collisions with Dataset 1 images.

Paths are defined in `src/config.py`. To use different directories, edit `BINARY_HEALTHY`, `BINARY_DISEASE`, and `STAGE_DIRS`.

---

## How the algorithm works

### Architecture

The classifier is a **ResNet-18** convolutional neural network. The final fully-connected layer is replaced with a linear head matching the number of target classes (2 for binary, 4 for staging). When `--no-pretrained` is not set, the backbone is initialized with ImageNet weights, providing transfer learning from general visual features to dental radiograph features.

### Training procedure

1. **Data loading** — Images are loaded from the folder structure described above. Each subfolder name determines the class label. Images are resized to 224 x 448 pixels (preserving the ~2:1 panoramic aspect ratio).

2. **Augmentation** — Training images undergo random horizontal flips, affine transformations (rotation, translation, scaling, shear), color jitter, random grayscale conversion, and random erasing. Validation images receive only resize and normalization.

3. **Class-weighted loss** — Because the merged dataset is imbalanced (e.g., ~2.8x more periodontal than non-periodontal images in binary), the training loss uses **inverse-frequency class weights**. For each class, the weight is `total_samples / (num_classes * class_count)`, giving underrepresented classes proportionally higher influence on the gradient.

4. **Optimization** — AdamW optimizer with weight decay (1e-4) and ReduceLROnPlateau learning-rate scheduler (halves LR when validation loss plateaus).

5. **Stratified k-fold cross-validation** — The dataset is split into k folds while preserving the class distribution in each fold. Each fold trains a fresh model and evaluates on a held-out portion. This provides robust performance estimates and reduces the risk of overfitting to a particular split.

6. **Early stopping** — Training halts if validation loss does not improve for a configurable number of epochs (default: 7), preventing overfitting.

7. **Evaluation** — After training each fold, the best model (by validation loss) is evaluated, producing accuracy, AUC-ROC, per-class precision/recall/F1, and a confusion matrix.

### Inference pipeline

At inference time, a single panoramic radiograph passes through:

1. **Binary classifier** — predicts periodontal disease presence with probability.
2. **Stage classifier** — if disease is predicted, assigns a severity stage (I–IV) with per-class probabilities.
3. **Bone-loss proxy** — computes a heuristic radiographic index from image band statistics (upper vs alveolar region intensity, edge strength). This maps to a severity label and approximate stage, independent of the neural network predictions.

---

## Training

Run all commands from the **project root** (the folder that contains `src/` and `dataset/`).

### Binary task with 5-fold cross-validation (recommended)

```bash
python -m src.train --task binary --epochs 30 --folds 5
```

### Four-class stage task with cross-validation

```bash
python -m src.train --task stage --epochs 40 --folds 5 --batch-size 4
```

### Single train/val split (quick run, no CV)

```bash
python -m src.train --task binary --epochs 30 --folds 0
```

Checkpoint (best validation model): `checkpoints/binary_resnet18.pt` or `checkpoints/stage_resnet18.pt`.

### Training options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 30 | Maximum training epochs per fold |
| `--batch-size` | 8 | Batch size |
| `--lr` | `1e-4` | AdamW learning rate |
| `--img-height` | 224 | Input height after resize |
| `--img-width` | 448 | Input width after resize (preserves panoramic ~2:1 ratio) |
| `--folds` | 5 | K-fold CV folds; 0 or 1 = single split |
| `--val-ratio` | 0.2 | Validation ratio when `--folds 0` |
| `--patience` | 7 | Early stopping patience (epochs without val loss improvement) |
| `--seed` | 42 | Random seed (fully deterministic splits) |
| `--no-pretrained` | off | Train ResNet from scratch (no ImageNet weights) |
| `--checkpoint-dir` | `checkpoints` | Where to save `.pt` files |

### Training output

After training you will find:

- **Checkpoints** in `checkpoints/` — best model weights.
- **Per-epoch CSV logs** in `results/<task>/` — one per fold (or `training.csv` for single split).
- **Evaluation metrics** in `results/<task>/`:
  - `cv_summary.json` — aggregated accuracy +/- std, AUC-ROC +/- std, per-class F1.
  - `fold_details.json` — full metrics for every fold (confusion matrices, classification reports).
  - `eval_metrics.json` — when running single split.

---

## Inference (full pipeline)

Runs the **bone-loss proxy** on the image. If the default checkpoint files exist under `checkpoints/`, it also runs the **binary** and **stage** models.

```bash
python -m src.pipeline --image path/to/panoramic.png
```

Custom checkpoint paths:

```bash
python -m src.pipeline --image path/to/panoramic.png ^
  --binary-ckpt checkpoints/binary_resnet18.pt ^
  --stage-ckpt checkpoints/stage_resnet18.pt
```

(On Unix shells, use `\` line continuation instead of `^`.)

Compare the patient image to a **reference healthy** panoramic from a similar scanner/contrast:

```bash
python -m src.pipeline --image path/to/patient.png --reference-healthy path/to/healthy.png
```

Only radiographic features (no neural nets):

```bash
python -m src.pipeline --image path/to/panoramic.png --skip-ckpt
```

Machine-readable JSON only (no summary banner):

```bash
python -m src.pipeline --image path/to/panoramic.png --json-only
```

Output includes a **human-readable summary** followed by **JSON** with binary probabilities, stage probabilities (if checkpoints loaded), bone-loss proxy values, and severity interpretation.

---

## Bone-loss proxy (important)

The values in `bone_loss_proxy` (e.g. `bone_loss_index`, `radiolucency_ratio`) come from **simple band-based statistics** on the grayscale image (upper vs alveolar region, edge strength). They are **research aids**, not a substitute for:

- CEJ–alveolar crest distance, or
- Expert segmentation / calibrated radiographic bone-loss **percentage**.

The `interpretation` field maps the index to a severity label (Healthy / Mild / Moderate / Severe / Very severe) and approximate 2017 stage. Thresholds are configurable in `src/bone_loss.py`.

---

## Project structure

```
paradentoza/
├── README.md
├── requirements.txt
├── convert_dataset2.py   # detection-to-classification conversion script
├── checkpoints/          # created on training; stores .pt checkpoints
├── results/              # created on training; CSV logs + JSON metrics
│   ├── binary/
│   └── stage/
├── dataset/              # merged classification images (Dataset 1 + converted Dataset 2)
│   ├── dental-panoramic/
│   │   ├── penyakit-non-periodontal/
│   │   └── penyakit-periodontal/
│   ├── stage_1/
│   ├── stage_2/
│   ├── stage_3/
│   └── stage_4/
├── dataset2/             # raw Dataset 2 in YOLO format (input for convert_dataset2.py)
│   └── Dataset/
│       ├── train/  valid/  test/
│       ├── data.yaml
│       └── names.txt
└── src/
    ├── __init__.py
    ├── config.py         # paths to dataset folders
    ├── data.py           # datasets, stratified splits, k-fold, transforms
    ├── model.py          # ResNet-18 classifier head
    ├── metrics.py        # confusion matrix, P/R/F1, AUC-ROC, logging
    ├── train.py          # training CLI with k-fold CV
    ├── bone_loss.py      # heuristic bone-related features
    └── pipeline.py       # inference CLI
```

---

## References

1. Waqas, M., Hasan, S., Khurshid, Z., & Kazmi, S. (2024). *OPG Dataset for Kennedy Classification of Partially Edentulous Arches*. Mendeley Data, V1. DOI: [10.17632/ccw5mvg69r.1](https://doi.org/10.17632/ccw5mvg69r.1)
2. *Dental Panoramic Dataset*. Kaggle. <https://www.kaggle.com/datasets/kasmira/dentalpanoramic>
3. Tonetti, M. S., Greenwell, H., & Kornman, K. S. (2018). Staging and grading of periodontitis: Framework and proposal of a new classification and case definition. *Journal of Periodontology*, 89(S1), S159–S172.

---

## License / clinical use

This repository is a **technical prototype**. Any clinical use requires validation, regulatory compliance, and appropriate professional oversight.

The Mendeley dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
