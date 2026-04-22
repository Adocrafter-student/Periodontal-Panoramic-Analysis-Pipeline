"""
End-to-end inference pipeline (v2 — hybrid approach).

1. **Binary classifier** (existing trained ResNet-18) decides healthy vs periodontitis.
2. If periodontitis is detected, the **rule-based staging algorithm** analyses the
   radiograph using CEJ-referenced bone-loss estimation and applies the 2017
   Tonetti staging framework.  Third molars are excluded automatically.
3. The legacy bone-loss proxy is still reported for continuity.

Examples:
  python -m src.pipeline --image path/to/xray.png
  python -m src.pipeline --image xray.png --binary-ckpt checkpoints/binary_resnet18.pt
  python -m src.pipeline --image xray.png --skip-ckpt          # radiograph analysis only
  python -m src.pipeline --image xray.png --teeth-json teeth.json  # manual measurements
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .bone_loss import analyze_bone_loss_proxy, compare_to_reference
from .config import CHECKPOINTS_DIR
from .model import build_classifier
from .radiograph_analysis import analyse_radiograph
from .staging import (
    PatientFindings,
    StagingResult,
    determine_stage,
    findings_from_dicts,
)


def _load_checkpoint(path: Path, device: torch.device) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _run_binary_classifier(
    image_path: Path,
    ckpt_path: Path,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Run the binary healthy/periodontal classifier."""
    ckpt = _load_checkpoint(ckpt_path, device)
    if ckpt is None:
        return None

    model = build_classifier(ckpt["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    h = int(ckpt.get("img_height", ckpt.get("img_size", 224)))
    w = int(ckpt.get("img_width", h))
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(), norm])

    pil = Image.open(image_path).convert("RGB")
    tensor = t(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = F.softmax(logits, dim=1)[0].cpu().tolist()

    is_periodontal = prob[1] >= prob[0]
    return {
        "label": int(logits.argmax(dim=1).item()),
        "label_name": "periodontal" if is_periodontal else "non_periodontal",
        "prob_non_periodontal": round(prob[0], 4),
        "prob_periodontal": round(prob[1], 4),
        "is_periodontal": is_periodontal,
    }


def predict_image(
    image_path: Path,
    binary_ckpt: Path | None = None,
    teeth_json: Path | None = None,
    healthy_reference: Path | None = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Run the full hybrid pipeline on a single image.

    Flow:
        1. Legacy bone-loss proxy (always, for backward compat)
        2. Binary classification (if checkpoint available)
        3. If periodontal (or no binary ckpt) → staging:
            a. If teeth_json provided → use manual measurements
            b. Else → CEJ-referenced radiograph analysis
        4. Apply rule-based staging algorithm
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out: Dict[str, Any] = {"image": str(image_path.resolve())}

    # --- legacy bone-loss proxy ---
    bone = analyze_bone_loss_proxy(image_path)
    out["bone_loss_proxy_legacy"] = bone.as_dict()

    if healthy_reference is not None:
        out["vs_reference"] = compare_to_reference(image_path, healthy_reference)

    # --- binary classification ---
    is_periodontal: bool = True  # default to staging if no classifier
    if binary_ckpt and binary_ckpt.is_file():
        binary_result = _run_binary_classifier(image_path, binary_ckpt, device)
        if binary_result:
            out["binary"] = binary_result
            is_periodontal = binary_result["is_periodontal"]

    # --- staging (only for periodontitis-positive cases) ---
    if is_periodontal:
        if teeth_json and teeth_json.is_file():
            raw = json.loads(teeth_json.read_text())
            tooth_records = raw if isinstance(raw, list) else raw.get("teeth", [])
            findings = findings_from_dicts(
                tooth_records,
                masticatory_dysfunction=raw.get("masticatory_dysfunction", False) if isinstance(raw, dict) else False,
                bite_collapse=raw.get("bite_collapse", False) if isinstance(raw, dict) else False,
                tooth_drifting=raw.get("tooth_drifting", False) if isinstance(raw, dict) else False,
                complex_rehab=raw.get("complex_rehab", False) if isinstance(raw, dict) else False,
            )
            out["measurement_source"] = "manual (teeth JSON)"
        else:
            radio = analyse_radiograph(image_path)
            findings = radio.findings
            out["radiograph_analysis"] = radio.as_dict()
            out["measurement_source"] = "automated (CEJ-referenced radiograph analysis)"

        staging_result: StagingResult = determine_stage(findings)
        out["staging"] = staging_result.as_dict()
    else:
        out["staging"] = {
            "stage": 0,
            "note": "No periodontitis detected by binary classifier",
        }

    return out


# ---------------------------------------------------------------------------
# CLI summary printer
# ---------------------------------------------------------------------------

def _print_summary(result: Dict[str, Any]) -> None:
    lines: list[str] = ["", "=" * 65, "  PERIODONTAL ANALYSIS SUMMARY", "=" * 65]

    # Binary
    if "binary" in result:
        b = result["binary"]
        label = b["label_name"].replace("_", " ").title()
        conf = max(b["prob_non_periodontal"], b["prob_periodontal"])
        lines.append(f"  Classification : {label} ({conf:.0%} confidence)")
    else:
        lines.append("  Classification : (no binary checkpoint — staging all images)")

    # Staging
    staging = result.get("staging", {})
    stage = staging.get("stage", "?")
    if stage == 0:
        lines.append(f"  Stage          : Healthy (no periodontitis)")
    else:
        extent = staging.get("extent", "")
        lines.append(f"  Stage          : Stage {stage} — {extent}")
        lines.append(f"  Max CAL        : {staging.get('max_cal_mm', '?')} mm")
        lines.append(f"  Max RBL        : {staging.get('max_rbl_percent', '?')}%  ({staging.get('rbl_location', '?')})")
        lines.append(f"  Tooth loss     : {staging.get('tooth_loss_count', 0)}")
        lines.append(f"  Probing depth  : {staging.get('max_probing_depth_mm', '?')} mm")
        lines.append(f"  Bone pattern   : {staging.get('bone_loss_pattern', '?')}")
        lines.append(f"  Furcation      : {staging.get('worst_furcation', 'None')}")
        lines.append(f"  Ridge defect   : {staging.get('ridge_defect', 'None')}")

        pct = staging.get("percent_teeth_affected", 0)
        lines.append(f"  Teeth affected : {pct:.0f}%")

        reasons = staging.get("reasons", [])
        if reasons:
            lines.append("")
            lines.append("  Staging rationale:")
            for r in reasons:
                lines.append(f"    • {r}")

    src = result.get("measurement_source", "")
    if src:
        lines.append(f"\n  Source: {src}")

    # Legacy proxy
    legacy = result.get("bone_loss_proxy_legacy", {})
    interp = legacy.get("interpretation", {})
    if interp:
        lines.append(f"\n  Legacy proxy   : {interp.get('severity', '?')} "
                      f"(index {interp.get('bone_loss_index', '?')})")

    lines.append("=" * 65)
    lines.append("")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Hybrid periodontal analysis: binary ML + rule-based staging",
    )
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--binary-ckpt", type=Path,
                    default=CHECKPOINTS_DIR / "binary_resnet18.pt")
    ap.add_argument("--teeth-json", type=Path, default=None,
                    help="JSON file with manual per-tooth measurements (bypasses radiograph analysis)")
    ap.add_argument("--reference-healthy", type=Path, default=None)
    ap.add_argument("--skip-ckpt", action="store_true",
                    help="Skip binary classifier, run staging on every image")
    ap.add_argument("--json-only", action="store_true",
                    help="Print only JSON, no human-readable summary")
    args = ap.parse_args()

    binary = None if args.skip_ckpt else args.binary_ckpt
    result = predict_image(
        args.image,
        binary_ckpt=binary,
        teeth_json=args.teeth_json,
        healthy_reference=args.reference_healthy,
    )

    if not args.json_only:
        _print_summary(result)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
