# Paradentoza — periodontal panoramic analysis pipeline

Classification and staging of periodontal disease from dental panoramic radiographs (OPGs), aligned with the **2017 periodontal disease classification** (Tonetti et al., Stages I–IV).

## Architecture — hybrid ML + rule-based staging

The pipeline uses a **two-step hybrid approach**:

1. **Binary ML classifier** (ResNet-18) — determines whether periodontitis is present or absent using the trained model on our two datasets (healthy vs non-healthy).
2. **Rule-based algorithmic staging** — for periodontitis-positive cases, applies the 2017 Tonetti staging framework using CEJ-referenced radiographic bone-loss estimation. No separate stage-training model is needed.

Key design decisions:

- **CEJ (Cementoenamel Junction)** is the anatomical reference point for all bone-loss measurements.
- **Third molars (wisdom teeth, FDI 18/28/38/48) are excluded** from all staging calculations — their variable eruption status, frequent impaction, and pseudo-pockets introduce noise that does not reflect true periodontal status.
- Staging considers: **RBL** (radiographic bone loss %), **tooth loss**, **complexity** factors, and **extent** (localised / generalised / molar-incisor).

### Staging framework (2017 classification)

| Criterion | Stage I | Stage II | Stage III | Stage IV |
|-----------|---------|----------|-----------|----------|
| Interdental CAL | 1–2 mm | 3–4 mm | ≥ 5 mm | ≥ 5 mm |
| Radiographic bone loss | Coronal third (< 15%) | Coronal third (15–33%) | Extends to middle third or beyond | Extends to middle third or beyond |
| Tooth loss (periodontitis) | None | None | ≤ 4 teeth | ≥ 5 teeth |
| Max probing depth | ≤ 4 mm | ≤ 5 mm | ≥ 6 mm | ≥ 6 mm |
| Bone loss pattern | Mostly horizontal | Mostly horizontal | Vertical ≥ 3 mm, furcation Class II/III | + complex rehabilitation needs |
| Extent | Localised / Generalised / Molar-incisor pattern |||

The algorithm assigns the **highest stage** reached across severity, tooth loss, and complexity criteria (worst-case rule per the framework).

---

## Datasets

This project combines two publicly available dental panoramic radiograph datasets.

### Dataset 1 — Dental Panoramic (Kaggle)

**Source**: <https://www.kaggle.com/datasets/kasmira/dentalpanoramic>

- `penyakit-non-periodontal` — 50 images (no periodontal disease)
- `penyakit-periodontal` — 42 images (periodontal disease present)

Used for **binary classification** (the primary ML task).

### Dataset 2 — OPG Kennedy Classification (Mendeley Data)

**Source**: Waqas, M., Hasan, S., Khurshid, Z., & Kazmi, S. (2024). *OPG Dataset for Kennedy Classification of Partially Edentulous Arches*. Mendeley Data, V1. <https://doi.org/10.17632/ccw5mvg69r.1>

622 dental panoramic radiographs annotated in YOLO format with bounding boxes for 10 classes of dental findings (Kennedy classification).

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

### Merged dataset summary

| Task | Dataset 1 only | After merging Dataset 2 |
|------|---------------|------------------------|
| Binary | 92 images (50 + 42) | **714 images** (190 + 524) |

---

## Requirements

- Python 3.10+
- [CUDA](https://pytorch.org/get-started/locally/) optional (GPU used when available)

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
```

### Step 2 — Place Dataset 2

Download from [Mendeley Data](https://doi.org/10.17632/ccw5mvg69r.1) and extract into `dataset2/Dataset/`.

### Step 3 — Convert and merge

```bash
python convert_dataset2.py          # analysis only
python convert_dataset2.py --copy   # analysis + copy into dataset/
```

---

## Training (binary classifier only)

Only the **binary** task (healthy vs periodontitis) needs ML training. Staging is handled algorithmically.

```bash
python -m src.train --task binary --epochs 30 --folds 5
```

### Training options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 30 | Maximum training epochs per fold |
| `--batch-size` | 8 | Batch size |
| `--lr` | `1e-4` | AdamW learning rate |
| `--img-height` | 224 | Input height after resize |
| `--img-width` | 448 | Input width (preserves panoramic ~2:1 ratio) |
| `--folds` | 5 | K-fold CV folds; 0 or 1 = single split |
| `--patience` | 7 | Early stopping patience |
| `--seed` | 42 | Random seed |
| `--no-pretrained` | off | Train ResNet from scratch |

Checkpoint saved to: `checkpoints/binary_resnet18.pt`

---

## Inference (full hybrid pipeline)

The pipeline runs: binary classification → CEJ-referenced radiograph analysis → rule-based staging.

### Basic usage

```bash
python -m src.pipeline --image path/to/panoramic.png
```

### With explicit binary checkpoint

```bash
python -m src.pipeline --image path/to/panoramic.png --binary-ckpt checkpoints/binary_resnet18.pt
```

### With manual per-tooth measurements (JSON)

If you have clinical measurements, bypass the automated radiograph analysis:

```bash
python -m src.pipeline --image path/to/panoramic.png --teeth-json measurements.json
```

The JSON format:

```json
{
  "teeth": [
    {"tooth_fdi": 11, "cal": 2.0, "probing_depth": 3.0, "rbl_percent": 12, "rbl_location": "coronal_third"},
    {"tooth_fdi": 16, "cal": 5.0, "probing_depth": 7.0, "rbl_percent": 45, "rbl_location": "middle_third",
     "bone_loss_pattern": "vertical", "vertical_bone_loss_mm": 4.0, "furcation": "class_ii"}
  ],
  "masticatory_dysfunction": false,
  "bite_collapse": false,
  "tooth_drifting": false,
  "complex_rehab": false
}
```

### Skip binary classifier (staging all images)

```bash
python -m src.pipeline --image path/to/panoramic.png --skip-ckpt
```

### JSON-only output

```bash
python -m src.pipeline --image path/to/panoramic.png --json-only
```

### Compare to healthy reference

```bash
python -m src.pipeline --image patient.png --reference-healthy healthy.png
```

---

## Pipeline output

The output includes:

- **Binary classification**: periodontal / non-periodontal with confidence
- **Staging result** (for periodontitis cases):
  - Stage (I–IV)
  - Severity criteria (max CAL, RBL %, location)
  - Tooth loss count
  - Complexity factors (probing depth, vertical bone loss, furcation, ridge defect)
  - Extent (Localised / Generalised / Molar-incisor) with % teeth affected
  - Bone loss pattern (mostly horizontal / mostly vertical / mixed)
  - Staging rationale (list of reasons for each criterion)
- **Radiograph analysis** (per-region CEJ-referenced measurements)
- **Legacy bone-loss proxy** (for backward compatibility)

---

## How it works

### Step 1: Binary classification (ML)

The ResNet-18 model classifies the OPG as healthy or periodontal. Trained with stratified k-fold cross-validation, class-weighted loss, early stopping, and LR scheduling.

### Step 2: Radiograph analysis (algorithmic)

For periodontitis-positive cases, the image is divided into **12 anatomical regions** (6 per jaw: molar, premolar, anterior — each side). Third-molar zones are excluded.

For each region:
- **CEJ band** is identified as the crown-root junction zone
- **Alveolar crest band** intensity is compared to CEJ reference
- Bone loss (darker crest vs CEJ) is quantified as **RBL %**
- **Gradient analysis** detects vertical vs horizontal bone loss patterns
- **Missing teeth** are estimated from region intensity/variance

### Step 3: Rule-based staging

Per-tooth measurements from the radiograph analysis are fed into the 2017 Tonetti staging algorithm:

1. **Severity** — determined by max CAL and RBL location
2. **Tooth loss** — can upgrade the stage (≥1 → Stage III, ≥5 → Stage IV)
3. **Complexity** — probing depth, vertical defects, furcation, rehabilitation needs
4. **Final stage** = maximum across all three criteria
5. **Extent** — localised (<30% teeth), generalised (≥30%), or molar-incisor pattern

### Third molar exclusion

Teeth 18, 28, 38, 48 (FDI notation) are excluded because:
- Variable eruption status and frequent impaction
- Distal pseudo-pockets from partially erupted teeth
- Naturally different bone levels unrelated to periodontitis
- Including them would produce false-positive bone loss signals

---

## Project structure

```
paradentoza/
├── README.md
├── requirements.txt
├── convert_dataset2.py       # detection-to-classification conversion
├── checkpoints/              # trained model weights (.pt)
├── results/                  # training metrics (CSV + JSON)
│   └── binary/
├── dataset/                  # classification images
│   └── dental-panoramic/
│       ├── penyakit-non-periodontal/
│       └── penyakit-periodontal/
├── dataset2/                 # raw YOLO-format Dataset 2
└── src/
    ├── __init__.py
    ├── config.py             # paths, third-molar FDI set
    ├── data.py               # datasets, splits, transforms
    ├── model.py              # ResNet-18 classifier
    ├── metrics.py            # evaluation metrics, logging
    ├── train.py              # training CLI (binary task)
    ├── staging.py            # rule-based 2017 staging algorithm
    ├── radiograph_analysis.py # CEJ-referenced OPG analysis
    ├── bone_loss.py          # legacy heuristic proxy
    └── pipeline.py           # hybrid inference CLI
```

---

## References

1. Tonetti, M. S., Greenwell, H., & Kornman, K. S. (2018). Staging and grading of periodontitis: Framework and proposal of a new classification and case definition. *Journal of Clinical Periodontology*, 45 Suppl 20, S149–S161.
2. Waqas, M., Hasan, S., Khurshid, Z., & Kazmi, S. (2024). *OPG Dataset for Kennedy Classification of Partially Edentulous Arches*. Mendeley Data, V1. DOI: [10.17632/ccw5mvg69r.1](https://doi.org/10.17632/ccw5mvg69r.1)
3. *Dental Panoramic Dataset*. Kaggle. <https://www.kaggle.com/datasets/kasmira/dentalpanoramic>

---

## License / clinical use

This repository is a **technical prototype**. Any clinical use requires validation, regulatory compliance, and appropriate professional oversight.

The Mendeley dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
