"""
End-to-end inference: optional binary + stage classifiers and bone-loss proxy.

Examples:
  python -m src.pipeline --image path/to/xray.png
  python -m src.pipeline --image xray.png --binary-ckpt checkpoints/binary_resnet18.pt \\
      --stage-ckpt checkpoints/stage_resnet18.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .bone_loss import analyze_bone_loss_proxy, compare_to_reference
from .config import CHECKPOINTS_DIR
from .model import build_classifier


def _load_checkpoint(path: Path, device: torch.device) -> Optional[dict]:
    if not path.is_file():
        return None
    # PyTorch 2.4+ defaults weights_only=True; full checkpoint dict needs False.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def predict_image(
    image_path: Path,
    binary_ckpt: Path | None = None,
    stage_ckpt: Path | None = None,
    healthy_reference: Path | None = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bone = analyze_bone_loss_proxy(image_path)
    out: Dict[str, Any] = {
        "image": str(image_path.resolve()),
        "bone_loss_proxy": bone.as_dict(),
    }
    if healthy_reference is not None:
        out["vs_reference"] = compare_to_reference(image_path, healthy_reference)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pil_cache: list = []

    def tensor_from_ckpt(ckpt: dict):
        if not pil_cache:
            pil_cache.append(Image.open(image_path).convert("RGB"))
        h = int(ckpt.get("img_height", ckpt.get("img_size", 224)))
        w = int(ckpt.get("img_width", h))
        t = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            norm,
        ])(pil_cache[0])
        return t.unsqueeze(0).to(device)

    if binary_ckpt and binary_ckpt.is_file():
        ckpt = _load_checkpoint(binary_ckpt, device)
        if ckpt:
            model = build_classifier(ckpt["num_classes"], pretrained=False).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            with torch.no_grad():
                logits = model(tensor_from_ckpt(ckpt))
                prob = F.softmax(logits, dim=1)[0].cpu().tolist()
            out["binary"] = {
                "label": int(logits.argmax(dim=1).item()),
                "label_name": "periodontal" if prob[1] >= prob[0] else "non_periodontal",
                "prob_non_periodontal": prob[0],
                "prob_periodontal": prob[1],
            }

    if stage_ckpt and stage_ckpt.is_file():
        ckpt = _load_checkpoint(stage_ckpt, device)
        if ckpt:
            model = build_classifier(ckpt["num_classes"], pretrained=False).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            with torch.no_grad():
                logits = model(tensor_from_ckpt(ckpt))
                prob = F.softmax(logits, dim=1)[0].cpu().tolist()
            stage = int(logits.argmax(dim=1).item()) + 1
            out["stage"] = {
                "stage_2017_class_folder": stage,
                "probabilities_by_stage": prob,
            }

    return out


def _print_summary(result: Dict[str, Any]) -> None:
    """Print a plain-language summary above the JSON dump."""
    lines: list[str] = ["", "=" * 60, "  ANALYSIS SUMMARY", "=" * 60]

    interp = result.get("bone_loss_proxy", {}).get("interpretation", {})
    if interp:
        severity = interp.get("severity", "Unknown")
        stage = interp.get("approximate_stage", "?")
        idx = interp.get("bone_loss_index", "?")
        desc = interp.get("description", "")
        stage_str = f"Stage {stage}" if stage else "N/A"
        lines.append(f"  Severity       : {severity}")
        lines.append(f"  Approx. stage  : {stage_str}")
        lines.append(f"  Bone-loss index: {idx}")
        lines.append(f"  {desc}")

    if "binary" in result:
        b = result["binary"]
        label = b["label_name"].replace("_", " ").title()
        conf = max(b["prob_non_periodontal"], b["prob_periodontal"])
        lines.append(f"  Classification : {label} ({conf:.0%} confidence)")

    if "stage" in result:
        s = result["stage"]
        lines.append(f"  Model stage    : Stage {s['stage_2017_class_folder']}")

    if "vs_reference" in result:
        delta = result["vs_reference"]["delta_bone_loss_index"]
        direction = "higher" if delta > 0 else "lower"
        lines.append(f"  vs. reference  : {abs(delta):.3f} {direction} than healthy ref.")

    lines.append("=" * 60)
    lines.append("")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--binary-ckpt", type=Path, default=CHECKPOINTS_DIR / "binary_resnet18.pt")
    ap.add_argument("--stage-ckpt", type=Path, default=CHECKPOINTS_DIR / "stage_resnet18.pt")
    ap.add_argument("--reference-healthy", type=Path, default=None)
    ap.add_argument("--skip-ckpt", action="store_true", help="Only run bone-loss proxy")
    ap.add_argument("--json-only", action="store_true", help="Print only JSON, no summary")
    args = ap.parse_args()

    binary = None if args.skip_ckpt else args.binary_ckpt
    stage = None if args.skip_ckpt else args.stage_ckpt
    result = predict_image(
        args.image,
        binary_ckpt=binary,
        stage_ckpt=stage,
        healthy_reference=args.reference_healthy,
    )

    if not args.json_only:
        _print_summary(result)

    import json

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
