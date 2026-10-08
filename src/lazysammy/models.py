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

from lazysammy.types import (
    CONFIG_FILENAMES,
    HF_MODEL_IDS,
    ModelSize,
)
from lazysammy.utils import auto_detect_device

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
    dev = torch.device(auto_detect_device(device))
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
    dev = torch.device(auto_detect_device(device))
    model_size_str = str(model_size)

    if not _is_hf_model_id(model_size_str) and checkpoint is not None:
        from sam2.build_sam import build_sam2_video_predictor

        cfg = CONFIG_FILENAMES[resolve_model_size(model_size)]
        logger.info("Loading SAM2VideoPredictor from checkpoint: %s", checkpoint)
        return build_sam2_video_predictor(
            cfg, str(checkpoint), device=str(dev), vos_optimized=vos_optimized, **kwargs
        )

    # HuggingFace hub: either an explicit model id or the mapped default.
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    if _is_hf_model_id(model_size_str):
        hf_id = model_size_str
    else:
        hf_id = HF_MODEL_IDS[resolve_model_size(model_size)]
    logger.info("Loading SAM2VideoPredictor from HuggingFace: %s", hf_id)
    return SAM2VideoPredictor.from_pretrained(
        hf_id, device=dev, vos_optimized=vos_optimized, **kwargs
    )


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
    dev = torch.device(auto_detect_device(device))
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


# ---------------------------------------------------------------------------
# SAM 3 builders
# ---------------------------------------------------------------------------


def _sam3_bpe_path() -> str:
    """Return the bundled BPE vocabulary path shipped with the ``sam3`` package."""
    import sam3

    return str(Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")


def load_sam3_image_model(
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    enable_instance_interactivity: bool = True,
    compile: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a SAM 3 image model (``Sam3Image``).

    Args:
        checkpoint: Optional local checkpoint path. When ``None`` the weights
            are downloaded from the ``facebook/sam3`` HuggingFace repo (access
            must be requested first).
        device: Device string override (auto-detected if ``None``).
        enable_instance_interactivity: Also build the SAM 1/2-style interactive
            predictor so point/box/mask prompts work alongside text prompts.
        compile: Enable ``torch.compile`` on the model components.
        **kwargs: Extra arguments forwarded to ``build_sam3_image_model``.

    Returns:
        An initialised ``Sam3Image`` model.
    """
    from sam3.model_builder import build_sam3_image_model

    dev = auto_detect_device(device)
    logger.info("Loading SAM3 image model (device=%s)", dev)
    return build_sam3_image_model(
        bpe_path=_sam3_bpe_path(),
        device=dev,
        checkpoint_path=str(checkpoint) if checkpoint is not None else None,
        enable_inst_interactivity=enable_instance_interactivity,
        compile=compile,
        **kwargs,
    )


def load_sam3_video_predictor(
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    version: str = "sam3",
    compile: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a SAM 3 video predictor with the ``handle_request`` API.

    Args:
        checkpoint: Optional local checkpoint path.
        device: Device string override (auto-detected if ``None``).
        version: ``"sam3"`` for the base model or ``"sam3.1"`` for the
            Object Multiplex variant.
        compile: Enable ``torch.compile`` (SAM 3.1 only).
        **kwargs: Extra arguments forwarded to ``build_sam3_predictor``.

    Returns:
        A predictor exposing ``handle_request`` / ``handle_stream_request``.
    """
    from sam3.model_builder import build_sam3_predictor

    dev = auto_detect_device(device)
    logger.info("Loading SAM3 video predictor (version=%s, device=%s)", version, dev)
    return build_sam3_predictor(
        checkpoint_path=str(checkpoint) if checkpoint is not None else None,
        bpe_path=_sam3_bpe_path(),
        version=version,
        compile=compile,
        **kwargs,
    )
