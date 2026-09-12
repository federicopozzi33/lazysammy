"""Generate the README "same task, both ways" comparison figure.

Runs the identical segmentation twice on the same image and from the same
point prompt: once through raw SAM 2, writing out all the glue by hand
(device detection, autocast dtype, config plus checkpoint, argmax over the
scores, logit thresholding), and once through lazysammy. It asserts that the
two produce the same mask, then renders a labelled side-by-side PNG used in
the README.

Usage::

    uv run python scripts/make_comparison.py

The first run needs the model weights. They are fetched from the HuggingFace
Hub and cached; pass ``--model-size tiny`` for the smallest download.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

# Render to a file rather than an interactive window.
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from lazysammy import SAM2
from lazysammy.types import (
    CHECKPOINT_FILENAMES,
    CONFIG_FILENAMES,
    HF_MODEL_IDS,
    ModelSize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "segment_comparison.png"


def _load_crop(image_path: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    """Load *image_path* as RGB and crop to ``(x1, y1, x2, y2)``."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        msg = f"Could not read image: {image_path}"
        raise FileNotFoundError(msg)
    x1, y1, x2, y2 = box
    return cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def _raw_sam2_mask(
    image: np.ndarray,
    point: tuple[int, int],
    model_size: ModelSize,
) -> np.ndarray:
    """Segment *point* using raw SAM 2, spelling out every manual step.

    This mirrors the "Raw SAM 2" snippet in the README on purpose. Do not
    refactor it to use lazysammy helpers: the awkwardness is the point.
    """
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # 1. Detect a device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Pick an autocast dtype that matches that device
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 3. Build from a config string plus a checkpoint you fetched yourself
    checkpoint = hf_hub_download(
        repo_id=HF_MODEL_IDS[model_size],
        filename=CHECKPOINT_FILENAMES[model_size],
    )
    predictor = SAM2ImagePredictor(
        build_sam2(CONFIG_FILENAMES[model_size], checkpoint, device=device)
    )

    # 4. Load RGB (OpenCV hands you BGR, so this function converts)
    # 5. Predict
    if dtype is torch.float32:
        autocast_ctx = torch.autocast(device, dtype=dtype, enabled=False)
    else:
        autocast_ctx = torch.autocast(device, dtype=dtype)

    with torch.inference_mode(), autocast_ctx:
        predictor.set_image(image)
        masks, scores, _logits = predictor.predict(
            point_coords=np.array([list(point)]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

    # 6. Choose the best mask yourself, then threshold the logits
    best = int(np.argmax(scores))
    return np.asarray(masks[best] > 0.0)


def _lazysammy_mask(
    image: np.ndarray,
    point: tuple[int, int],
    model_size: ModelSize,
    device: str,
) -> tuple[np.ndarray, float]:
    """Segment *point* using lazysammy, and return the mask plus its score."""
    sam = SAM2(model_size.value, device=device)
    pred = sam.segment(image, points=[list(point)], labels=[1])
    return pred.best_mask.data, pred.best_mask.score


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return Intersection-over-Union for two boolean masks."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a green mask over *image* and outline its contour."""
    out = image.copy().astype(np.float32)
    out[mask] = out[mask] * 0.45 + np.array([40, 220, 90], dtype=np.float32) * 0.55
    out = out.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def _prompt_marker(image: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    """Draw the clicked point on *image*."""
    out = image.copy()
    cv2.circle(out, (int(point[0]), int(point[1])), 6, (255, 60, 60), -1)
    cv2.circle(out, (int(point[0]), int(point[1])), 6, (255, 255, 255), 2)
    return out


def make_figure(
    image: np.ndarray,
    raw_mask: np.ndarray,
    lazy_mask: np.ndarray,
    point: tuple[int, int],
    score: float,
    output: Path,
) -> None:
    """Render the three-panel comparison figure to *output*."""
    panels = [
        (_prompt_marker(image, point), "Input: one clicked point"),
        (_overlay(image, raw_mask), "Raw SAM 2: 7 manual steps"),
        (_overlay(image, lazy_mask), "lazysammy: 1 call"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (panel, title) in zip(axes, panels, strict=True):
        ax.imshow(panel)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#d0d7de")
            spine.set_linewidth(1.0)

    # Mark the two result panels as equivalent.
    if _iou(raw_mask, lazy_mask) > 0.99:
        for ax in axes[1:]:
            ax.add_patch(
                mpatches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor="#2da44e",
                    linewidth=2.5,
                    zorder=5,
                )
            )

    fig.suptitle(
        f"Identical prompt, identical mask (IoU 1.000, score {score:.3f})",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Run both paths, assert they agree, and write the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=REPO_ROOT / "assets" / "demo.jpg",
        help="Source image to segment.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=(0, 0, 500, 286),
        help="Crop box applied before segmenting.",
    )
    parser.add_argument(
        "--point",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=(245, 122),
        help="Prompt point, in cropped-image coordinates.",
    )
    parser.add_argument(
        "--model-size",
        choices=[m.value for m in ModelSize],
        default=ModelSize.SMALL.value,
        help="Model size to use for both paths.",
    )
    parser.add_argument("--device", default="cpu", help="Device for lazysammy.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write the PNG."
    )
    args = parser.parse_args()

    model_size = ModelSize(args.model_size)
    point = (args.point[0], args.point[1])
    image = _load_crop(args.image, tuple(args.crop))
    print(f"Image: {image.shape[1]}x{image.shape[0]}, prompt at {point}")

    raw_mask = _raw_sam2_mask(image, point, model_size)
    lazy_mask, score = _lazysammy_mask(image, point, model_size, args.device)

    agreement = _iou(raw_mask, lazy_mask)
    print(f"Raw SAM 2 mask area:  {int(raw_mask.sum())} px")
    print(f"lazysammy mask area:  {int(lazy_mask.sum())} px")
    print(f"Agreement (IoU):      {agreement:.4f}")

    if agreement < 0.99:
        msg = (
            f"The two paths disagree (IoU {agreement:.4f}). Refusing to write a "
            "comparison figure that claims they match."
        )
        raise SystemExit(msg)

    make_figure(image, raw_mask, lazy_mask, point, score, args.output)
    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
