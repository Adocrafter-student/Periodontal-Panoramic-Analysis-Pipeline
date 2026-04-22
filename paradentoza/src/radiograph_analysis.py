"""
CEJ-referenced radiographic bone-loss estimation from panoramic X-rays (OPGs).

Divides the panoramic image into anatomical regions, estimates bone level
relative to the expected CEJ position, and produces per-region measurements
that feed into the rule-based staging algorithm.

Third molars (FDI 18/28/38/48) are excluded: their variable eruption status,
frequent impaction, and distal pseudo-pockets make them unreliable indicators
of true periodontal bone loss.

Limitations:
    - OPGs are 2-D projections with inherent distortion; measurements here are
      *estimates*, not clinical ground-truth.
    - True CAL requires a clinical probe; we approximate it from RBL.
    - This module is designed to be replaced or supplemented by a proper
      segmentation model (e.g. tooth + bone instance segmentation) once
      available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .staging import (
    ANALYSABLE_TEETH_FDI,
    BoneLossPattern,
    FurcationClass,
    PatientFindings,
    ToothMeasurement,
)

# ---------------------------------------------------------------------------
# Anatomical region layout on a standard panoramic radiograph
# ---------------------------------------------------------------------------
# OPGs are read left-to-right as: patient's right → patient's left.
# Vertical: upper jaw (maxilla) is the top half, lower jaw (mandible) bottom.
#
# We define 12 analysis regions (6 per jaw), skipping third-molar zones:
#   Maxilla:  R-molar(16-17), R-premolar(14-15), R-anterior(11-13),
#             L-anterior(21-23), L-premolar(24-25), L-molar(26-27)
#   Mandible: R-molar(46-47), R-premolar(44-45), R-anterior(41-43),
#             L-anterior(31-33), L-premolar(34-35), L-molar(36-37)

@dataclass
class RegionSpec:
    name: str
    fdi_teeth: List[int]
    x_start: float     # normalised [0, 1] left edge of region on OPG
    x_end: float
    y_start: float     # normalised [0, 1] top edge
    y_end: float

# X fractions approximate where each region sits on a typical OPG.
# Third-molar strips (~0-0.06 and ~0.94-1.0) are intentionally excluded.
REGIONS: List[RegionSpec] = [
    # Maxilla (upper half: y 0.08 .. 0.48)
    RegionSpec("max_R_molar",     [16, 17], 0.06, 0.22, 0.08, 0.48),
    RegionSpec("max_R_premolar",  [14, 15], 0.22, 0.36, 0.08, 0.48),
    RegionSpec("max_R_anterior",  [11, 12, 13], 0.36, 0.50, 0.08, 0.48),
    RegionSpec("max_L_anterior",  [21, 22, 23], 0.50, 0.64, 0.08, 0.48),
    RegionSpec("max_L_premolar",  [24, 25], 0.64, 0.78, 0.08, 0.48),
    RegionSpec("max_L_molar",     [26, 27], 0.78, 0.94, 0.08, 0.48),
    # Mandible (lower half: y 0.52 .. 0.92)
    RegionSpec("mand_R_molar",    [46, 47], 0.06, 0.22, 0.52, 0.92),
    RegionSpec("mand_R_premolar", [44, 45], 0.22, 0.36, 0.52, 0.92),
    RegionSpec("mand_R_anterior", [41, 42, 43], 0.36, 0.50, 0.52, 0.92),
    RegionSpec("mand_L_anterior", [31, 32, 33], 0.50, 0.64, 0.52, 0.92),
    RegionSpec("mand_L_premolar", [34, 35], 0.64, 0.78, 0.52, 0.92),
    RegionSpec("mand_L_molar",    [36, 37], 0.78, 0.94, 0.52, 0.92),
]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_gray(source: Union[Path, str, np.ndarray]) -> np.ndarray:
    """Load as float32 grayscale in [0, 1]."""
    if isinstance(source, (Path, str)):
        img = Image.open(source).convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0
    arr = source.astype(np.float32)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=-1)
    if arr.max() > 1.5:
        arr /= 255.0
    return arr


def _extract_region(gray: np.ndarray, spec: RegionSpec) -> np.ndarray:
    h, w = gray.shape
    r = gray[
        int(spec.y_start * h):int(spec.y_end * h),
        int(spec.x_start * w):int(spec.x_end * w),
    ]
    return r


# ---------------------------------------------------------------------------
# CEJ-referenced bone level estimation for a single region
# ---------------------------------------------------------------------------

@dataclass
class RegionAnalysis:
    region_name: str
    fdi_teeth: List[int]
    cej_band_mean: float          # mean intensity of the expected CEJ / crown-root junction zone
    alveolar_crest_mean: float    # mean intensity of the alveolar crest band
    mid_root_mean: float          # mean intensity of mid-root band
    apical_mean: float            # mean intensity near apex
    edge_gradient: float          # average gradient magnitude (discontinuity -> bone loss)
    rbl_ratio: float              # bone-loss proxy: deviation of crest from CEJ reference
    rbl_percent_est: float        # estimated RBL as % of root length
    rbl_location_est: str         # "coronal_third" / "middle_third" / "apical_third"
    bone_loss_pattern_est: BoneLossPattern
    vertical_defect_score: float  # higher -> more likely vertical defect
    tooth_presence_score: float = 1.0  # 0..1; low = likely edentulous


def _analyse_region(gray: np.ndarray, spec: RegionSpec) -> RegionAnalysis:
    """Estimate bone loss in one anatomical region of the OPG."""
    crop = _extract_region(gray, spec)
    h, w = crop.shape
    if h < 8 or w < 8:
        return RegionAnalysis(
            region_name=spec.name, fdi_teeth=spec.fdi_teeth,
            cej_band_mean=0, alveolar_crest_mean=0, mid_root_mean=0,
            apical_mean=0, edge_gradient=0, rbl_ratio=0,
            rbl_percent_est=0, rbl_location_est="coronal_third",
            bone_loss_pattern_est=BoneLossPattern.HORIZONTAL,
            vertical_defect_score=0,
        )

    is_maxilla = spec.name.startswith("max")

    # Band definitions are narrowly focused around the alveolar crest margin
    # relative to CEJ.  In the maxilla roots point upward on OPG (apex at top,
    # CEJ / crown junction toward bottom of crop).  Mandible is reversed.
    #
    # We use narrow bands to avoid confounding structures (mandibular canal,
    # mental foramen, chin pad artifacts).
    if is_maxilla:
        cej_band = crop[int(0.58 * h):int(0.68 * h), :]   # CEJ zone (crown-root junction)
        crest_band = crop[int(0.42 * h):int(0.55 * h), :]  # alveolar crest (just above CEJ)
        mid_root = crop[int(0.22 * h):int(0.40 * h), :]
        apical = crop[int(0.05 * h):int(0.20 * h), :]
    else:
        # Mandible: crowns at top of crop, roots point down.
        # CEJ is around 0.30-0.40, crest just below at 0.40-0.50.
        # Avoid the lower portion (>0.70) which often contains the
        # mandibular canal — a naturally dark structure.
        cej_band = crop[int(0.28 * h):int(0.38 * h), :]
        crest_band = crop[int(0.38 * h):int(0.50 * h), :]
        mid_root = crop[int(0.50 * h):int(0.65 * h), :]
        apical = crop[int(0.65 * h):int(0.75 * h), :]

    cej_m = float(np.mean(cej_band)) if cej_band.size else 0
    crest_m = float(np.mean(crest_band)) if crest_band.size else 0
    mid_m = float(np.mean(mid_root)) if mid_root.size else 0
    apex_m = float(np.mean(apical)) if apical.size else 0

    # Edge gradient in crest band — vertical defects produce strong local gradients.
    if crest_band.shape[0] >= 3 and crest_band.shape[1] >= 3:
        gx = np.abs(crest_band[:, 1:] - crest_band[:, :-1])
        gy = np.abs(crest_band[1:, :] - crest_band[:-1, :])
        edge_grad = float(0.5 * (np.mean(gx) + np.mean(gy)))
    else:
        edge_grad = 0.0

    # ---- Signal 1: CEJ-vs-crest ratio (original approach) ----
    # Bone is radiopaque (bright); loss = darker crest relative to CEJ.
    eps = 1e-6
    raw_ratio = (cej_m - crest_m) / (cej_m + eps)
    is_molar = "molar" in spec.name
    if is_maxilla:
        baseline = 0.12
    elif is_molar:
        baseline = 0.25
    else:
        baseline = 0.18
    s1_ratio = max(0.0, raw_ratio - baseline)
    s1_pct = float(np.clip(s1_ratio / 0.25 * 100, 0, 100))

    # GATE: if both bands are too dark the region is at the jaw border /
    # poorly exposed — any ratio is noise, not bone loss.
    MIN_RELIABLE_INTENSITY = 0.20
    if max(cej_m, crest_m) < MIN_RELIABLE_INTENSITY:
        s1_pct = 0.0

    # ---- Signal 2: vertical intensity-profile drop-off ----
    # In healthy bone the vertical profile (CEJ -> crest -> mid-root) stays
    # relatively bright.  In bone loss the mid-root zone drops off sharply
    # because bone that should surround the root is gone.
    #
    # IMPORTANT: Normal anatomy already shows a large crest-to-mid-root
    # drop (~30-50%) due to root canals, trabecular pattern, and
    # projection overlap.  Only *extreme* drop-offs indicate real bone
    # loss.  This signal fires only as a supplement when S1 is zero.
    if s1_pct < 5.0 and crest_m > MIN_RELIABLE_INTENSITY and mid_m > eps:
        drop_off = max(0.0, (crest_m - mid_m) / (crest_m + eps))
        # Normal anatomy drop is 0.30-0.50; only above 0.55 is suspicious.
        drop_baseline = 0.55
        s2_signal = max(0.0, drop_off - drop_baseline)
        s2_pct = float(np.clip(s2_signal / 0.25 * 100, 0, 60))
    else:
        s2_pct = 0.0

    # ---- Combine signals ----
    rbl_pct = max(s1_pct, s2_pct)

    if rbl_pct < 15:
        rbl_loc = "coronal_third"
    elif rbl_pct < 40:
        if mid_m < crest_m - 0.05:
            rbl_loc = "middle_third"
        else:
            rbl_loc = "coronal_third"
    else:
        rbl_loc = "middle_third" if rbl_pct < 66 else "apical_third"

    # Vertical defect heuristic: high column-wise gradient variance suggests
    # uneven bone margin (angular defects) rather than flat horizontal loss.
    col_means = np.mean(crest_band, axis=0) if crest_band.shape[1] > 1 else np.array([0])
    vert_score = float(np.std(col_means))

    if vert_score > 0.10 and edge_grad > 0.06:
        pattern = BoneLossPattern.VERTICAL
    elif vert_score > 0.06:
        pattern = BoneLossPattern.MIXED
    else:
        pattern = BoneLossPattern.HORIZONTAL

    return RegionAnalysis(
        region_name=spec.name,
        fdi_teeth=spec.fdi_teeth,
        cej_band_mean=round(cej_m, 4),
        alveolar_crest_mean=round(crest_m, 4),
        mid_root_mean=round(mid_m, 4),
        apical_mean=round(apex_m, 4),
        edge_gradient=round(edge_grad, 4),
        rbl_ratio=round(s1_ratio, 4),
        rbl_percent_est=round(rbl_pct, 1),
        rbl_location_est=rbl_loc,
        bone_loss_pattern_est=pattern,
        vertical_defect_score=round(vert_score, 4),
    )


# ---------------------------------------------------------------------------
# Full-image analysis → PatientFindings
# ---------------------------------------------------------------------------

@dataclass
class RadioGraphResult:
    """Complete radiographic analysis of one OPG."""
    regions: List[RegionAnalysis]
    overall_rbl_percent: float
    overall_rbl_location: str
    estimated_missing_regions: int
    findings: PatientFindings

    def as_dict(self) -> Dict:
        return {
            "overall_rbl_percent": round(self.overall_rbl_percent, 1),
            "overall_rbl_location": self.overall_rbl_location,
            "estimated_missing_regions": self.estimated_missing_regions,
            "regions": [
                {
                    "name": r.region_name,
                    "fdi_teeth": r.fdi_teeth,
                    "cej_band_mean": r.cej_band_mean,
                    "alveolar_crest_mean": r.alveolar_crest_mean,
                    "rbl_percent_est": r.rbl_percent_est,
                    "rbl_location": r.rbl_location_est,
                    "bone_loss_pattern": r.bone_loss_pattern_est.name.lower(),
                    "vertical_defect_score": r.vertical_defect_score,
                    "tooth_presence_score": r.tooth_presence_score,
                }
                for r in self.regions
            ],
        }


def _compute_image_contrast_ref(gray: np.ndarray) -> float:
    """Compute a reference contrast level for the entire image.

    This is the median std across all maxillary regions, which are
    the best-exposed part of any OPG.  Used to calibrate tooth
    presence scoring: images with globally low contrast need relaxed
    thresholds.
    """
    maxillary_specs = [s for s in REGIONS if s.name.startswith("max")]
    stds = []
    for spec in maxillary_specs:
        crop = _extract_region(gray, spec)
        h, w = crop.shape
        center = crop[int(0.15 * h):int(0.85 * h), int(0.15 * w):int(0.85 * w)]
        if center.size > 16:
            stds.append(float(np.std(center)))
    return float(np.median(stds)) if stds else 0.10


def _tooth_presence_score(
    gray: np.ndarray, spec: RegionSpec, image_contrast_ref: float = 0.10,
) -> float:
    """Score 0..1 indicating how likely teeth are present in this region.

    Teeth produce high local contrast: bright enamel/dentin, dark root
    canals/PDL spaces, bright surrounding bone.  Edentulous ridges are
    more uniform (just residual bone + soft tissue overlay).

    The score is calibrated relative to the image's own global contrast
    (``image_contrast_ref``) so that uniformly low-contrast radiographs
    don't get false "missing teeth" calls.
    """
    crop = _extract_region(gray, spec)
    if crop.size < 64:
        return 0.0

    is_maxilla = spec.name.startswith("max")

    h, w = crop.shape
    center = crop[int(0.15 * h):int(0.85 * h), int(0.15 * w):int(0.85 * w)]
    if center.size < 16:
        center = crop

    c_mean = float(np.mean(center))
    c_std = float(np.std(center))
    p10, p90 = float(np.percentile(center, 10)), float(np.percentile(center, 90))
    intensity_range = p90 - p10

    # Mandible gate: very dark regions are unreliable.
    if not is_maxilla and c_mean < 0.20:
        return 0.8

    # Horizontal profile: count distinct tooth-like peaks.
    col_profile = np.mean(crop, axis=0)
    profile_mean = float(np.mean(col_profile))
    if profile_mean > 0.08:
        threshold = profile_mean + 0.5 * float(np.std(col_profile))
        above = col_profile > threshold
        transitions = int(np.sum(np.diff(above.astype(int)) == 1))
    else:
        transitions = 0

    # Adaptive thresholds: scale by image contrast.
    # If the whole image has low contrast (ref ~0.05), a region std
    # of 0.04 is normal, not edentulous.  If the image ref is ~0.15,
    # a region std of 0.04 is genuinely low.
    #
    # We want the contrast_score to be relative:
    #   region_std / image_ref  >> 0.6 → likely has teeth
    #   region_std / image_ref  << 0.4 → likely edentulous
    ref = max(image_contrast_ref, 0.04)
    relative_std = c_std / ref

    contrast_score = float(np.clip((relative_std - 0.40) / 0.40, 0, 1))
    range_score = float(np.clip((intensity_range - 0.10) / 0.30, 0, 1))
    peak_score = float(np.clip(transitions / 3.0, 0, 1))

    score = 0.45 * contrast_score + 0.35 * range_score + 0.20 * peak_score

    # Mandibular boost for moderate-brightness regions.
    if not is_maxilla and c_mean < 0.35:
        score = 0.3 + 0.7 * score

    return score


def _estimate_missing_in_region(
    gray: np.ndarray, spec: RegionSpec, image_contrast_ref: float = 0.10,
) -> int:
    """Estimate how many teeth are missing in a region.

    Returns 0 (all present) up to len(spec.fdi_teeth) (all missing).
    """
    score = _tooth_presence_score(gray, spec, image_contrast_ref)
    n_teeth = len(spec.fdi_teeth)

    if score >= 0.55:
        return 0
    if score >= 0.30:
        return max(1, n_teeth - 1)
    return n_teeth


def analyse_radiograph(
    source: Union[Path, str, np.ndarray],
) -> RadioGraphResult:
    """Analyse a panoramic X-ray and produce PatientFindings for staging.

    Steps:
        1. Load grayscale image
        2. Analyse each of the 12 anatomical regions (3rd molars excluded)
        3. Convert region-level estimates to per-tooth measurements
        4. Detect possible edentulous regions (missing teeth)
        5. Package into PatientFindings ready for determine_stage()
    """
    gray = _load_gray(source)
    contrast_ref = _compute_image_contrast_ref(gray)
    region_results: List[RegionAnalysis] = []
    teeth: List[ToothMeasurement] = []
    total_missing = 0

    for spec in REGIONS:
        ra = _analyse_region(gray, spec)

        presence = _tooth_presence_score(gray, spec, contrast_ref)
        ra.tooth_presence_score = round(presence, 3)
        region_results.append(ra)

        n_missing = _estimate_missing_in_region(gray, spec, contrast_ref)
        total_missing += n_missing

        analysable_fdi = [f for f in spec.fdi_teeth if f in ANALYSABLE_TEETH_FDI]
        missing_fdi = set(analysable_fdi[:n_missing])

        for fdi in analysable_fdi:
            is_missing = fdi in missing_fdi
            cal_est = _rbl_pct_to_cal_estimate(ra.rbl_percent_est)
            vbl_est = ra.vertical_defect_score * 15.0 if ra.bone_loss_pattern_est == BoneLossPattern.VERTICAL else 0.0

            teeth.append(ToothMeasurement(
                tooth_fdi=fdi,
                cal=cal_est if not is_missing else 0.0,
                probing_depth=cal_est if not is_missing else 0.0,
                rbl_percent=ra.rbl_percent_est if not is_missing else 0.0,
                rbl_location=ra.rbl_location_est,
                bone_loss_pattern=ra.bone_loss_pattern_est,
                vertical_bone_loss_mm=min(vbl_est, 10.0) if not is_missing else 0.0,
                furcation=FurcationClass.NONE,
                is_missing=is_missing,
                is_present=not is_missing,
            ))

    findings = PatientFindings(teeth=teeth)

    rbl_values = [r.rbl_percent_est for r in region_results]
    overall_rbl = max(rbl_values) if rbl_values else 0.0
    loc_order = {"coronal_third": 0, "middle_third": 1, "apical_third": 2}
    overall_loc = max(
        (r.rbl_location_est for r in region_results),
        key=lambda l: loc_order.get(l, 0),
        default="coronal_third",
    )

    return RadioGraphResult(
        regions=region_results,
        overall_rbl_percent=overall_rbl,
        overall_rbl_location=overall_loc,
        estimated_missing_regions=total_missing,
        findings=findings,
    )


def _rbl_pct_to_cal_estimate(rbl_pct: float) -> float:
    """Rough mapping from RBL% to CAL in mm.

    Average root length ~13mm; RBL% of root length → mm from CEJ.
    CAL ≈ (rbl_pct / 100) * 13.  Capped at plausible clinical range.
    """
    AVERAGE_ROOT_LENGTH_MM = 13.0
    cal = (rbl_pct / 100.0) * AVERAGE_ROOT_LENGTH_MM
    return round(min(cal, 15.0), 1)
