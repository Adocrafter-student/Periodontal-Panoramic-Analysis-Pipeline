# Methodology - How we measure periodontitis staging

This document explains step by step how the pipeline analyses a dental panoramic radiograph (OPG) and assigns a periodontitis stage.

---

## Overview: two-step hybrid approach

```
  Panoramic X-ray (OPG)
         |
         v
  +---------------------+
  | Step 1: Binary ML    |   "Is periodontitis present?"
  | (ResNet-18 model)    |   Trained on our two datasets
  +---------------------+
         |
    yes  |   no --> HEALTHY (stop)
         v
  +---------------------+
  | Step 2: Algorithmic  |   "Which stage (I-IV)?"
  | staging (rule-based) |   Based on 2017 Tonetti framework
  +---------------------+
         |
         v
  Stage I / II / III / IV
  + Extent (Localised / Generalised / Molar-incisor)
```

Step 1 uses a trained neural network (our existing binary classifier).
Step 2 uses no ML -- it is a deterministic algorithm that analyses the X-ray image and applies clinical rules.

---

## Step 1: Binary classification (ML)

- Model: ResNet-18, trained on healthy vs periodontal images.
- Input: OPG resized to 224x448 pixels, normalised.
- Output: probability of periodontitis (0-100%).
- If the probability is >= 50%, the case is sent to Step 2 for staging.
- If < 50%, the case is classified as healthy and staging is skipped.

This step uses **our existing trained model** and datasets (Kaggle dental panoramic + Mendeley OPG Kennedy dataset, 714 images total).

---

## Step 2: Algorithmic staging

This step has three sub-stages: radiograph analysis, measurement extraction, and rule-based staging.

### 2a. Image segmentation into 12 anatomical regions

The OPG is divided into **12 analysis regions** (6 per jaw). Third-molar zones are excluded.

```
  Panoramic X-ray layout (patient's perspective):

  MAXILLA (top half of OPG):
  +--------+--------+---------+---------+--------+--------+
  | R molar| R pre- | R ante- | L ante- | L pre- | L molar|
  | 16, 17 | molar  | rior    | rior    | molar  | 26, 27 |
  |        | 14, 15 | 11-13   | 21-23   | 24, 25 |        |
  +--------+--------+---------+---------+--------+--------+

  MANDIBLE (bottom half of OPG):
  +--------+--------+---------+---------+--------+--------+
  | R molar| R pre- | R ante- | L ante- | L pre- | L molar|
  | 46, 47 | molar  | rior    | rior    | molar  | 36, 37 |
  |        | 44, 45 | 41-43   | 31-33   | 34, 35 |        |
  +--------+--------+---------+---------+--------+--------+

  EXCLUDED: Third molars (18, 28, 38, 48) at the far left/right edges
```

Tooth numbers follow **FDI notation** (the international standard used in dentistry).

**Why exclude third molars?** Wisdom teeth have variable eruption, frequent impaction, and naturally different bone levels. Including them would create false bone-loss signals that have nothing to do with periodontitis.

### 2b. CEJ-referenced bone loss measurement

For each of the 12 regions, we measure bone loss using the **CEJ (Cementoenamel Junction)** as the reference point. The CEJ is the anatomical line where tooth enamel meets the root surface -- this is where healthy bone level should be.

#### What we measure in each region

The region crop is divided into four horizontal bands:

```
  Maxilla (roots point up on OPG):
  +---------------------------+
  | Apical band (root tips)   |   ~top 15%
  | Mid-root band             |   ~15-40%
  | Alveolar crest band       |   ~42-55%    <-- where bone edge should be
  | CEJ band (reference)      |   ~58-68%    <-- our anchor point
  | Crown area                |   ~bottom
  +---------------------------+

  Mandible (roots point down):
  +---------------------------+
  | Crown area                |   ~top
  | CEJ band (reference)      |   ~28-38%    <-- our anchor point
  | Alveolar crest band       |   ~38-50%    <-- where bone edge should be
  | Mid-root band             |   ~50-65%
  | (mandibular canal zone)   |   ~below 70% -- AVOIDED
  +---------------------------+
```

#### How bone loss percentage (RBL%) is calculated

**Signal 1 -- CEJ-vs-crest intensity comparison:**

On an X-ray, bone appears bright (radiopaque) and bone loss appears dark (radiolucent). We compare the average brightness of the CEJ band to the alveolar crest band:

```
  raw_ratio = (CEJ_brightness - Crest_brightness) / CEJ_brightness
```

If the crest is much darker than CEJ, bone has been lost. We subtract a baseline to account for normal anatomical differences:

- Maxilla baseline: 0.12
- Mandible premolar/anterior baseline: 0.18
- Mandible molar baseline: 0.25 (higher because the ramus and mandibular canal create artifacts)

```
  RBL% = max(0, (raw_ratio - baseline) / 0.25) * 100
```

**Safety gate:** If both bands are very dark (below 0.20 brightness), the region is at the jaw edge or poorly exposed. We set RBL to 0% for that region because the ratio would be meaningless noise.

**Signal 2 -- Crest-to-mid-root drop-off (supplementary):**

If Signal 1 shows no bone loss but the mid-root zone is dramatically darker than the crest, this may indicate bone loss that Signal 1 missed (e.g. when dental restorations make the crest appear uniformly bright). This signal only fires when the drop exceeds 55% (normal anatomy drops 30-50%).

#### How RBL% maps to bone loss location

| RBL% | Location | Meaning |
|------|----------|---------|
| 0-14% | Coronal third | Bone loss confined to the upper third of root (if any) |
| 15-39% | Coronal third or middle third | Moderate loss, approaching mid-root level |
| 40-65% | Middle third | Bone loss extends to the middle third of the root |
| 66-100% | Apical third | Severe loss extending toward the root apex |

#### How RBL% converts to CAL (mm)

Since we cannot probe a patient through an X-ray, we estimate CAL from RBL:

```
  CAL (mm) = (RBL% / 100) * 13 mm
```

13 mm is the average root length across all permanent teeth. This is an approximation -- actual root lengths vary from ~11 mm (lower incisors) to ~17 mm (canines).

### 2c. Tooth presence detection (missing teeth)

For each region, we estimate whether teeth are present or missing. This uses three signals:

1. **Local contrast (45% weight):** Teeth create high contrast on X-rays (bright enamel next to dark root canals next to bright bone). Edentulous ridges have uniform, low contrast. We measure the standard deviation of pixel intensities.

2. **Intensity range (35% weight):** The range between the 10th and 90th percentile of pixel brightness. Regions with teeth show a wide range; edentulous ridges show a narrow range.

3. **Horizontal peak count (20% weight):** Each tooth creates a bright vertical column on the X-ray. We count how many distinct peaks appear in the horizontal intensity profile.

**Adaptive calibration:** Different X-ray machines produce different contrast levels. To avoid calling all teeth "missing" on a low-contrast radiograph, we compute a per-image contrast reference from the maxillary regions (which are the best-exposed part of any OPG). Each region's contrast is then judged relative to this baseline.

**Mandible safety:** Very dark mandibular regions (mean brightness < 0.20) are assumed to have teeth present, because we cannot distinguish "teeth in poor exposure" from "no teeth."

| Presence score | Interpretation | Missing teeth assigned |
|----------------|---------------|----------------------|
| >= 0.55 | Teeth likely present | 0 |
| 0.30 - 0.55 | Some teeth may be missing | 1 (or n-1 for 3-tooth regions) |
| < 0.30 | Region likely edentulous | All teeth in region |

### 2d. Bone loss pattern detection

We check for horizontal vs vertical bone loss:

- **Horizontal:** Bone margin is relatively flat across the region. Detected when column-wise intensity variance is low (< 0.06).
- **Vertical (angular defects):** Bone margin is uneven -- dips next to some teeth. Detected when column-wise variance is high (> 0.10) AND edge gradients are strong (> 0.06).
- **Mixed:** In between the two thresholds.

### 2e. Rule-based staging (2017 Tonetti framework)

All the measurements feed into a deterministic staging algorithm that implements the 2017 classification:

#### Criterion 1: Severity (determines base stage)

| Condition | Stage assigned |
|-----------|---------------|
| CAL 1-2 mm, RBL in coronal third (< 15%) | Stage I |
| CAL 3-4 mm, RBL in coronal third (15-33%) | Stage II |
| CAL >= 5 mm, or RBL in middle/apical third | Stage III |

#### Criterion 2: Tooth loss (can upgrade stage)

| Condition | Minimum stage |
|-----------|--------------|
| No teeth lost to periodontitis | No upgrade |
| 1-4 teeth lost | At least Stage III |
| >= 5 teeth lost | Stage IV |

#### Criterion 3: Complexity (can upgrade stage)

| Finding | Stage |
|---------|-------|
| Max probing depth <= 4 mm, mostly horizontal | Stage I |
| Max probing depth <= 5 mm, mostly horizontal | Stage II |
| Probing depth >= 6 mm, vertical bone loss >= 3 mm, furcation Class II/III, moderate ridge defect | Stage III |
| Masticatory dysfunction, bite collapse, tooth drifting/flaring, need for complex rehabilitation, severe ridge defect | Stage IV |

#### Final stage = maximum across all three criteria

The framework uses a **worst-case rule**: the final stage is the highest stage reached across severity, tooth loss, and complexity. For example, if severity says Stage II but the patient has 6 missing teeth, the final stage is IV.

#### Extent classification

| % of teeth affected | Classification |
|---------------------|---------------|
| < 30% | Localised |
| >= 30% | Generalised |
| Mostly molars + incisors affected | Molar-incisor pattern |

---

## What the output numbers mean

| Field | What it measures | Source |
|-------|-----------------|--------|
| `stage` (1-4) | Final periodontitis stage | Maximum of severity + tooth loss + complexity |
| `severity_stage` | Stage from CAL + RBL alone | CEJ-referenced bone loss measurement |
| `complexity_stage` | Stage from complexity factors | Probing depth, vertical defects, furcation, rehabilitation needs |
| `max_cal_mm` | Worst clinical attachment loss | Estimated from RBL% (RBL% / 100 * 13 mm) |
| `max_rbl_percent` | Worst radiographic bone loss | CEJ-vs-crest brightness comparison |
| `rbl_location` | How far down the root bone loss extends | coronal_third / middle_third / apical_third |
| `tooth_loss_count` | Teeth lost to periodontitis | Contrast-based presence detection |
| `max_probing_depth_mm` | Deepest pocket depth | Estimated same as CAL (from radiograph) |
| `extent` | How widespread the disease is | % of teeth showing bone loss or missing |
| `percent_teeth_affected` | Percentage of teeth involved | Teeth with CAL > 0 or RBL > 0 or missing |
| `bone_loss_pattern` | Horizontal vs vertical bone loss | Column-wise variance in the crest band |
| `tooth_presence_score` | Per-region: likelihood teeth are present | Contrast + intensity range + peak count |

---

## Known limitations

1. **CAL is estimated, not measured.** True clinical attachment loss requires a periodontal probe in the patient's mouth. We estimate it from radiographic bone loss using average root lengths.

2. **Probing depth = CAL on radiographs.** In reality, probing depth and CAL can differ (gingival recession vs pseudo-pockets). We have no way to measure pockets from an X-ray alone.

3. **Band-based analysis averages across regions.** Localised bone loss around a single tooth may be diluted when averaged with neighbouring healthy teeth in the same region.

4. **Low-contrast images.** Some OPGs have very low overall contrast. The algorithm adapts but may under-detect bone loss on these images.

5. **Dental restorations.** Metal crowns, bridges, and implants appear very bright on X-rays, which can mask underlying bone loss.

6. **Furcation and vertical defects** are estimated heuristically and may be missed or over-detected.

7. **OPG distortion.** Panoramic radiographs are 2D projections of 3D structures with inherent magnification and distortion, especially in the premolar region.

---

## Questions for mentor

### Clinical validation

1. **Do the RBL% thresholds make clinical sense?** We subtract a baseline (0.12 maxilla, 0.18 mandible, 0.25 mand. molar) from the CEJ-to-crest ratio. Are these reasonable, or should they be calibrated against clinically verified cases?

2. **For tooth loss detection, should we assume all missing teeth are lost to periodontitis?** Currently every missing tooth counts toward the staging tooth-loss criterion. In practice, teeth can be missing for other reasons (extraction for orthodontics, trauma, congenital absence). Should we add a way to distinguish this?

3. **The CAL estimate uses a fixed 13 mm average root length.** Would it improve accuracy to use tooth-specific root lengths (e.g. 11 mm for lower incisors, 17 mm for canines)?

### Methodology

4. **Is the approach of using band-averaged pixel intensity sufficient for a master's thesis, or do we need per-tooth segmentation?** The current approach divides the jaw into 12 regions and averages across 2-3 teeth per region. A segmentation model (e.g. detecting individual tooth + bone boundaries) would be more precise but requires annotated training data.

5. **Should we incorporate the grading dimension (A/B/C)?** The 2017 framework also has a grading system based on disease progression rate, risk factors (smoking, diabetes), and radiographic bone loss / age ratio. Currently we only implement staging. Is grading in scope for the thesis?

6. **For the extent classification, should we use the "localised" descriptor as default when the algorithm isn't confident?** Currently, if < 30% of teeth show detectable bone loss, we call it "localised" -- but this might be because the radiograph analysis under-detects, not because the disease is truly localised.

### Dataset and evaluation

7. **How should we validate the staging results?** The datasets don't have clinician-verified stage labels (the stage_1-4 folders were derived from Kennedy classification patterns, not clinical probing). Do we have access to any cases with known clinical staging for ground-truth comparison?

8. **Is the current tooth loss detection adequate, or should we consider training a separate object-detection model (e.g. YOLO) specifically for detecting present vs missing teeth?** This could be more reliable than the contrast-based heuristic.

9. **Should we collect a small test set of 10-20 OPGs with your manual staging assessment to calibrate the thresholds?** This would let us tune the baseline values and tooth-presence thresholds against expert ground truth.
