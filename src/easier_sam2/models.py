"""Model loading utilities for SAM2.

Supports loading models from HuggingFace Hub or from local checkpoint files.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import torch

# Suppress the harmless "cannot import name '_C'" warning from SAM2.
# This fires when the optional C extension is not compiled, which is the
# common case for pip/git installs.  It does not affect results.
warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'",
    category=UserWarning,
)

from easier_sam2.types import (
    CONFIG_FILENAMES,
    CHECKPOINT_URLS,
    HF_MODEL_IDS,
    ModelSize,
)
from easier_sam2.utils import auto_detect_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Size resolution
# ---------------------------------------------------------------------------

_SIZE_ALIASES: dict[str, ModelSize] = {
    "t": ModelSize.TINY,
    "tiny": ModelSize.TINY,
    "s": ModelSize.SMALL,
    "small": ModelSize.SMALL,
    "b+": ModelSize.BASE_PLUS,
    "b": ModelSize.BASE_PLUS,
    "base": ModelSize.BASE_PLUS,
    "base_plus": ModelSize.BASE_PLUS,
    "base+": ModelSize.BASE_PLUS,
    "l": ModelSize.LARGE,
    "large": ModelSize.LARGE,
}


def resolve_model_size(size: str | ModelSize) -> ModelSize:
    """Resolve a user-provided size string to :class:`ModelSize`.

    Accepts aliases like ``"large"``, ``"l"``, ``"tiny"``, ``"t"``, ``"b+"``, etc.

    Args:
        size: A model size string or :class:`ModelSize` enum value.

    Returns:
        The resolved :class:`ModelSize`.

    Raises:
        ValueError: If the size string is not recognized.
    """
    if isinstance(size, ModelSize):
        return size
    key = size.strip().lower().replace("-", "_")
    if key in _SIZE_ALIASES:
        return _SIZE_ALIASES[key]
    # Check if it's a HuggingFace model ID
    for ms, hf_id in HF_MODEL_IDS.items():
        if size == hf_id:
            return ms
    msg = (
        f"Unknown model size: {size!r}. "
        f"Valid options: {list(_SIZE_ALIASES.keys())} or a HuggingFace model ID."
    )
    raise ValueError(msg)


def _is_hf_model_id(model_id: str) -> bool:
    """Check whether *model_id* looks like a HuggingFace repo id."""
    return "/" in model_id


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def load_image_predictor(
    model_size: str | ModelSize = ModelSize.LARGE,
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> Any:
    """Build a :class:`SAM2ImagePredictor`.

    Args:
        model_size: Model size alias or HuggingFace model ID.
        checkpoint: Optional path to a local ``.pt`` checkpoint.
        device: Device string override (auto-detected if ``None``).
        **kwargs: Extra keyword arguments forwarded to the predictor constructor.

    Returns:
        An initialised ``SAM2ImagePredictor``.
    """
    dev = auto_detect_device(device)
    model_size_str = str(model_size)

    if _is_hf_model_id(model_size_str):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        logger.info("Loading SAM2ImagePredictor from HuggingFace: %s", model_size_str)
        return SAM2ImagePredictor.from_pretrained(model_size_str, device=dev, **kwargs)

    size = resolve_model_size(model_size)

    if checkpoint is not None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        cfg = CONFIG_FILENAMES[size]
        logger.info("Loading SAM2ImagePredictor from checkpoint: %s", checkpoint)
        model = build_sam2(cfg, str(checkpoint), device=str(dev))
        return SAM2ImagePredictor(model, **kwargs)

    # Default: use HuggingFace hub
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    hf_id = HF_MODEL_IDS[size]
    logger.info("Loading SAM2ImagePredictor from HuggingFace: %s", hf_id)
    return SAM2ImagePredictor.from_pretrained(hf_id, device=dev, **kwargs)


def load_video_predictor(
    model_size: str | ModelSize = ModelSize.LARGE,
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    vos_optimized: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a :class:`SAM2VideoPredictor`.

    Args:
        model_size: Model size alias or HuggingFace model ID.
        checkpoint: Optional path to a local ``.pt`` checkpoint.
        device: Device string override (auto-detected if ``None``).
        vos_optimized: Use ``torch.compile`` for all components (faster but
            requires PyTorch >= 2.5.1 and a CUDA GPU).
        **kwargs: Extra keyword arguments forwarded to the predictor constructor.

    Returns:
        An initialised ``SAM2VideoPredictor``.
    """
    dev = auto_detect_device(device)
    model_size_str = str(model_size)

    if _is_hf_model_id(model_size_str):
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        logger.info("Loading SAM2VideoPredictor from HuggingFace: %s", model_size_str)
        return SAM2VideoPredictor.from_pretrained(model_size_str, device=dev, **kwargs)

    size = resolve_model_size(model_size)

    if checkpoint is not None:
        from sam2.build_sam import build_sam2_video_predictor

        cfg = CONFIG_FILENAMES[size]
        logger.info("Loading SAM2VideoPredictor from checkpoint: %s", checkpoint)
        return build_sam2_video_predictor(
            cfg, str(checkpoint), device=str(dev), vos_optimized=vos_optimized, **kwargs
        )

    # Default: use HuggingFace hub
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    hf_id = HF_MODEL_IDS[size]
    logger.info("Loading SAM2VideoPredictor from HuggingFace: %s", hf_id)
    return SAM2VideoPredictor.from_pretrained(hf_id, device=dev, **kwargs)


def load_auto_mask_generator(
    model_size: str | ModelSize = ModelSize.LARGE,
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.8,
    stability_score_thresh: float = 0.95,
    min_mask_region_area: int = 0,
    output_mode: str = "binary_mask",
    use_m2m: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a :class:`SAM2AutomaticMaskGenerator`.

    Args:
        model_size: Model size alias or HuggingFace model ID.
        checkpoint: Optional path to a local ``.pt`` checkpoint.
        device: Device string override.
        points_per_side: Points sampled along one side of the image.
        pred_iou_thresh: IoU threshold for filtering.
        stability_score_thresh: Stability score threshold.
        min_mask_region_area: Remove masks smaller than this (in pixels).
        output_mode: ``"binary_mask"``, ``"uncompressed_rle"``, or ``"coco_rle"``.
        use_m2m: Use mask-to-mask refinement step.
        **kwargs: Extra keyword arguments.

    Returns:
        An initialised ``SAM2AutomaticMaskGenerator``.
    """
    dev = auto_detect_device(device)
    model_size_str = str(model_size)

    if _is_hf_model_id(model_size_str):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        logger.info("Loading SAM2AutomaticMaskGenerator from HuggingFace: %s", model_size_str)
        return SAM2AutomaticMaskGenerator.from_pretrained(
            model_size_str,
            device=dev,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            output_mode=output_mode,
            use_m2m=use_m2m,
            **kwargs,
        )

    size = resolve_model_size(model_size)

    if checkpoint is not None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        cfg = CONFIG_FILENAMES[size]
        logger.info("Loading SAM2AutomaticMaskGenerator from checkpoint: %s", checkpoint)
        model = build_sam2(cfg, str(checkpoint), device=str(dev))
        return SAM2AutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            output_mode=output_mode,
            use_m2m=use_m2m,
            **kwargs,
        )

    # Default: use HuggingFace hub
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    hf_id = HF_MODEL_IDS[size]
    logger.info("Loading SAM2AutomaticMaskGenerator from HuggingFace: %s", hf_id)
    return SAM2AutomaticMaskGenerator.from_pretrained(
        hf_id,
        device=dev,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        output_mode=output_mode,
        use_m2m=use_m2m,
        **kwargs,
    )
