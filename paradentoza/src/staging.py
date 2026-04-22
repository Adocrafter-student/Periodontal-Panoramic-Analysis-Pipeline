"""
Rule-based periodontitis staging per the 2017 classification framework
(Tonetti MS, Greenwell H, Kornman KS. J Clin Periodontol. 2018;45 Suppl 20:S149-S161).

Accepts structured per-tooth measurements and applies the staging criteria:
  Severity  → interdental CAL + radiographic bone loss (RBL)
  Tooth loss → periodontitis-attributable missing teeth
  Complexity → probing depth, vertical bone loss, furcation, ridge defects
  Extent     → localised (<30% teeth), generalised (≥30%), molar-incisor pattern

Third molars (FDI 18, 28, 38, 48) are excluded from all calculations because
their variable eruption, impaction, and pseudo-pocket depths introduce noise
that does not reflect true periodontal status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# FDI tooth numbering — third molars to exclude
# ---------------------------------------------------------------------------
THIRD_MOLARS_FDI = frozenset({18, 28, 38, 48})

ALL_PERMANENT_TEETH_FDI = frozenset(
    list(range(11, 19)) + list(range(21, 29))
    + list(range(31, 39)) + list(range(41, 49))
)

ANALYSABLE_TEETH_FDI = ALL_PERMANENT_TEETH_FDI - THIRD_MOLARS_FDI


class FurcationClass(IntEnum):
    NONE = 0
    CLASS_I = 1
    CLASS_II = 2
    CLASS_III = 3


class BoneLossPattern(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    MIXED = 2


class RidgeDefect(IntEnum):
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


class Extent(IntEnum):
    LOCALISED = 0
    GENERALISED = 1
    MOLAR_INCISOR = 2


# ---------------------------------------------------------------------------
# Per-tooth measurement
# ---------------------------------------------------------------------------
@dataclass
class ToothMeasurement:
    """Clinical and radiographic findings for a single tooth.

    All linear values are in millimetres. RBL is expressed as a percentage
    of the root length measured from CEJ to the apex.
    """
    tooth_fdi: int
    cal: float = 0.0                              # interdental clinical attachment loss (mm)
    probing_depth: float = 0.0                     # max probing depth (mm)
    rbl_percent: float = 0.0                       # radiographic bone loss as % of root length from CEJ
    rbl_location: str = "coronal_third"            # "coronal_third" | "middle_third" | "apical_third"
    bone_loss_pattern: BoneLossPattern = BoneLossPattern.HORIZONTAL
    vertical_bone_loss_mm: float = 0.0             # vertical / angular defect depth
    furcation: FurcationClass = FurcationClass.NONE
    is_missing: bool = False                       # lost due to periodontitis
    is_present: bool = True                        # tooth exists in the arch

    @property
    def is_third_molar(self) -> bool:
        return self.tooth_fdi in THIRD_MOLARS_FDI


# ---------------------------------------------------------------------------
# Patient-level findings aggregated across teeth
# ---------------------------------------------------------------------------
@dataclass
class PatientFindings:
    """Aggregated clinical record used as input to the staging algorithm."""
    teeth: List[ToothMeasurement] = field(default_factory=list)
    has_masticatory_dysfunction: bool = False
    has_secondary_occlusal_trauma: bool = False
    has_bite_collapse: bool = False
    has_tooth_drifting_or_flaring: bool = False
    needs_complex_rehabilitation: bool = False

    def analysable_teeth(self) -> List[ToothMeasurement]:
        """All teeth excluding third molars."""
        return [t for t in self.teeth if not t.is_third_molar]

    def present_teeth(self) -> List[ToothMeasurement]:
        return [t for t in self.analysable_teeth() if t.is_present and not t.is_missing]

    def teeth_lost_to_periodontitis(self) -> int:
        return sum(1 for t in self.analysable_teeth() if t.is_missing)

    def max_cal(self) -> float:
        present = self.present_teeth()
        return max((t.cal for t in present), default=0.0)

    def max_probing_depth(self) -> float:
        present = self.present_teeth()
        return max((t.probing_depth for t in present), default=0.0)

    def max_rbl_percent(self) -> float:
        present = self.present_teeth()
        return max((t.rbl_percent for t in present), default=0.0)

    def worst_rbl_location(self) -> str:
        """Return the worst (most apical) bone loss location across teeth."""
        order = {"coronal_third": 0, "middle_third": 1, "apical_third": 2}
        present = self.present_teeth()
        if not present:
            return "coronal_third"
        return max(present, key=lambda t: order.get(t.rbl_location, 0)).rbl_location

    def max_vertical_bone_loss(self) -> float:
        present = self.present_teeth()
        return max((t.vertical_bone_loss_mm for t in present), default=0.0)

    def worst_furcation(self) -> FurcationClass:
        present = self.present_teeth()
        return max((t.furcation for t in present), default=FurcationClass.NONE)

    def has_ridge_defect(self) -> RidgeDefect:
        tooth_loss = self.teeth_lost_to_periodontitis()
        if tooth_loss >= 5:
            return RidgeDefect.SEVERE
        if tooth_loss >= 2:
            return RidgeDefect.MODERATE
        if tooth_loss >= 1:
            return RidgeDefect.MILD
        return RidgeDefect.NONE


# ---------------------------------------------------------------------------
# Staging result
# ---------------------------------------------------------------------------
@dataclass
class StagingResult:
    stage: int                       # 1–4
    severity_stage: int              # stage from severity criteria alone
    complexity_stage: int            # stage from complexity criteria alone
    tooth_loss_count: int
    extent: Extent
    extent_label: str                # "Localised" / "Generalised" / "Molar-incisor"
    percent_teeth_affected: float
    max_cal_mm: float
    max_rbl_percent: float
    rbl_location: str
    max_probing_depth_mm: float
    max_vertical_bone_loss_mm: float
    worst_furcation: str
    ridge_defect: str
    bone_loss_pattern: str
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "stage": self.stage,
            "severity_stage": self.severity_stage,
            "complexity_stage": self.complexity_stage,
            "tooth_loss_count": self.tooth_loss_count,
            "extent": self.extent_label,
            "percent_teeth_affected": round(self.percent_teeth_affected, 1),
            "max_cal_mm": round(self.max_cal_mm, 1),
            "max_rbl_percent": round(self.max_rbl_percent, 1),
            "rbl_location": self.rbl_location,
            "max_probing_depth_mm": round(self.max_probing_depth_mm, 1),
            "max_vertical_bone_loss_mm": round(self.max_vertical_bone_loss_mm, 1),
            "worst_furcation": self.worst_furcation,
            "ridge_defect": self.ridge_defect,
            "bone_loss_pattern": self.bone_loss_pattern,
            "reasons": self.reasons,
        }


# ---------------------------------------------------------------------------
# Staging algorithm
# ---------------------------------------------------------------------------

def _severity_stage(cal: float, rbl_pct: float, rbl_loc: str) -> Tuple[int, List[str]]:
    """Determine stage from severity criteria (CAL + RBL)."""
    reasons: List[str] = []

    if cal >= 5.0 or rbl_loc in ("middle_third", "apical_third"):
        stage = 3  # Stage III or IV — disambiguated by tooth loss / complexity
        if cal >= 5.0:
            reasons.append(f"CAL {cal:.1f}mm >= 5mm")
        if rbl_loc in ("middle_third", "apical_third"):
            reasons.append(f"RBL extends to {rbl_loc.replace('_', ' ')}")
    elif cal >= 3.0 or 15.0 <= rbl_pct <= 33.0:
        stage = 2
        if cal >= 3.0:
            reasons.append(f"CAL {cal:.1f}mm in 3-4mm range")
        if 15.0 <= rbl_pct <= 33.0:
            reasons.append(f"RBL {rbl_pct:.0f}% (coronal third 15-33%)")
    elif cal >= 1.0 or 0 < rbl_pct < 15.0:
        stage = 1
        if cal >= 1.0:
            reasons.append(f"CAL {cal:.1f}mm in 1-2mm range")
        if 0 < rbl_pct < 15.0:
            reasons.append(f"RBL {rbl_pct:.0f}% (coronal third <15%)")
    else:
        stage = 1
        reasons.append("Minimal severity findings")

    return stage, reasons


def _tooth_loss_upgrade(
    base_stage: int, tooth_loss: int,
) -> Tuple[int, List[str]]:
    """Upgrade stage based on periodontitis-attributable tooth loss."""
    reasons: List[str] = []
    stage = base_stage

    if tooth_loss >= 5:
        stage = max(stage, 4)
        reasons.append(f"Tooth loss {tooth_loss} >= 5 -> Stage IV")
    elif tooth_loss >= 1:
        stage = max(stage, 3)
        reasons.append(f"Tooth loss {tooth_loss} (1-4) -> at least Stage III")

    return stage, reasons


def _complexity_stage(findings: PatientFindings) -> Tuple[int, List[str]]:
    """Determine stage from complexity factors."""
    reasons: List[str] = []
    max_pd = findings.max_probing_depth()
    max_vbl = findings.max_vertical_bone_loss()
    furc = findings.worst_furcation()
    ridge = findings.has_ridge_defect()

    stage = 1

    # Stage IV complexity
    stage_iv_markers = []
    if findings.needs_complex_rehabilitation:
        stage_iv_markers.append("needs complex rehabilitation")
    if findings.has_masticatory_dysfunction:
        stage_iv_markers.append("masticatory dysfunction")
    if findings.has_secondary_occlusal_trauma:
        stage_iv_markers.append("secondary occlusal trauma")
    if findings.has_bite_collapse:
        stage_iv_markers.append("bite collapse")
    if findings.has_tooth_drifting_or_flaring:
        stage_iv_markers.append("tooth drifting/flaring")
    if ridge == RidgeDefect.SEVERE:
        stage_iv_markers.append("severe ridge defect")

    if stage_iv_markers:
        stage = 4
        reasons.append(f"Stage IV complexity: {', '.join(stage_iv_markers)}")
        return stage, reasons

    # Stage III complexity
    stage_iii_markers = []
    if max_pd >= 6.0:
        stage_iii_markers.append(f"probing depth {max_pd:.1f}mm >= 6mm")
    if max_vbl >= 3.0:
        stage_iii_markers.append(f"vertical bone loss {max_vbl:.1f}mm >= 3mm")
    if furc >= FurcationClass.CLASS_II:
        stage_iii_markers.append(f"furcation Class {furc.name.replace('CLASS_', '')}")
    if ridge == RidgeDefect.MODERATE:
        stage_iii_markers.append("moderate ridge defect")

    if stage_iii_markers:
        stage = 3
        reasons.append(f"Stage III complexity: {', '.join(stage_iii_markers)}")
        return stage, reasons

    # Stage II complexity
    if max_pd <= 5.0 and max_pd > 4.0:
        stage = 2
        reasons.append(f"Max probing depth {max_pd:.1f}mm (<=5mm, mostly horizontal)")
    elif max_pd <= 4.0:
        stage = 1
        reasons.append(f"Max probing depth {max_pd:.1f}mm (<=4mm, mostly horizontal)")
    else:
        stage = 2
        reasons.append(f"Max probing depth {max_pd:.1f}mm")

    return stage, reasons


def _determine_extent(findings: PatientFindings) -> Tuple[Extent, str, float]:
    """Classify extent as localised, generalised, or molar-incisor pattern."""
    analysable = findings.analysable_teeth()
    present = [t for t in analysable if t.is_present]
    if not present:
        return Extent.LOCALISED, "Localised", 0.0

    affected = [
        t for t in present
        if t.cal >= 1.0 or t.rbl_percent > 0 or t.is_missing
    ]
    pct = (len(affected) / len(present)) * 100 if present else 0.0

    MOLARS_FDI = {16, 17, 26, 27, 36, 37, 46, 47}
    INCISORS_FDI = {11, 12, 21, 22, 31, 32, 41, 42}
    mi_teeth = MOLARS_FDI | INCISORS_FDI

    affected_fdi = {t.tooth_fdi for t in affected}
    mi_affected = affected_fdi & mi_teeth
    non_mi_affected = affected_fdi - mi_teeth

    if mi_affected and len(non_mi_affected) <= 2 and len(mi_affected) >= 3:
        return Extent.MOLAR_INCISOR, "Molar-incisor", pct

    if pct >= 30.0:
        return Extent.GENERALISED, "Generalised", pct

    return Extent.LOCALISED, "Localised", pct


def _dominant_bone_loss_pattern(findings: PatientFindings) -> str:
    present = findings.present_teeth()
    if not present:
        return "horizontal"
    vert = sum(1 for t in present if t.bone_loss_pattern == BoneLossPattern.VERTICAL)
    horiz = sum(1 for t in present if t.bone_loss_pattern == BoneLossPattern.HORIZONTAL)
    mixed = sum(1 for t in present if t.bone_loss_pattern == BoneLossPattern.MIXED)
    if vert > horiz and vert > mixed:
        return "mostly vertical"
    if mixed > horiz:
        return "mixed"
    return "mostly horizontal"


def determine_stage(findings: PatientFindings) -> StagingResult:
    """Apply the 2017 periodontitis staging framework to patient findings.

    The final stage is the HIGHEST stage reached across severity, tooth loss,
    and complexity criteria (the framework uses a "worst-case" rule).
    """
    max_cal = findings.max_cal()
    max_rbl = findings.max_rbl_percent()
    rbl_loc = findings.worst_rbl_location()
    tooth_loss = findings.teeth_lost_to_periodontitis()

    sev_stage, sev_reasons = _severity_stage(max_cal, max_rbl, rbl_loc)
    tl_stage, tl_reasons = _tooth_loss_upgrade(sev_stage, tooth_loss)
    comp_stage, comp_reasons = _complexity_stage(findings)

    final_stage = max(sev_stage, tl_stage, comp_stage)
    final_stage = max(1, min(4, final_stage))

    extent, extent_label, pct_affected = _determine_extent(findings)

    all_reasons = sev_reasons + tl_reasons + comp_reasons

    return StagingResult(
        stage=final_stage,
        severity_stage=sev_stage,
        complexity_stage=comp_stage,
        tooth_loss_count=tooth_loss,
        extent=extent,
        extent_label=extent_label,
        percent_teeth_affected=pct_affected,
        max_cal_mm=max_cal,
        max_rbl_percent=max_rbl,
        rbl_location=rbl_loc,
        max_probing_depth_mm=findings.max_probing_depth(),
        max_vertical_bone_loss_mm=findings.max_vertical_bone_loss(),
        worst_furcation=findings.worst_furcation().name.replace("CLASS_", "Class ").replace("NONE", "None"),
        ridge_defect=findings.has_ridge_defect().name.replace("_", " ").title(),
        bone_loss_pattern=_dominant_bone_loss_pattern(findings),
        reasons=all_reasons,
    )


# ---------------------------------------------------------------------------
# Convenience: build findings from simplified per-tooth dictionaries
# ---------------------------------------------------------------------------

def findings_from_dicts(
    tooth_records: List[Dict],
    *,
    masticatory_dysfunction: bool = False,
    secondary_occlusal_trauma: bool = False,
    bite_collapse: bool = False,
    tooth_drifting: bool = False,
    complex_rehab: bool = False,
) -> PatientFindings:
    """Build PatientFindings from a list of dicts (e.g. JSON input).

    Each dict may contain keys matching ToothMeasurement fields:
        tooth_fdi, cal, probing_depth, rbl_percent, rbl_location,
        bone_loss_pattern, vertical_bone_loss_mm, furcation, is_missing, is_present

    Third molars are accepted but will be filtered out during staging.
    """
    teeth: List[ToothMeasurement] = []
    for rec in tooth_records:
        fdi = rec.get("tooth_fdi", rec.get("fdi", 0))
        pattern = rec.get("bone_loss_pattern", 0)
        if isinstance(pattern, str):
            pattern = {"horizontal": 0, "vertical": 1, "mixed": 2}.get(pattern.lower(), 0)
        furc = rec.get("furcation", 0)
        if isinstance(furc, str):
            furc = {"none": 0, "class_i": 1, "class_ii": 2, "class_iii": 3}.get(
                furc.lower().replace(" ", "_"), 0
            )

        teeth.append(ToothMeasurement(
            tooth_fdi=int(fdi),
            cal=float(rec.get("cal", 0)),
            probing_depth=float(rec.get("probing_depth", 0)),
            rbl_percent=float(rec.get("rbl_percent", 0)),
            rbl_location=rec.get("rbl_location", "coronal_third"),
            bone_loss_pattern=BoneLossPattern(int(pattern)),
            vertical_bone_loss_mm=float(rec.get("vertical_bone_loss_mm", 0)),
            furcation=FurcationClass(int(furc)),
            is_missing=bool(rec.get("is_missing", False)),
            is_present=bool(rec.get("is_present", True)),
        ))

    return PatientFindings(
        teeth=teeth,
        has_masticatory_dysfunction=masticatory_dysfunction,
        has_secondary_occlusal_trauma=secondary_occlusal_trauma,
        has_bite_collapse=bite_collapse,
        has_tooth_drifting_or_flaring=tooth_drifting,
        needs_complex_rehabilitation=complex_rehab,
    )
