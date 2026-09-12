"""High-level automatic mask generation wrapper around SAM2AutomaticMaskGenerator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from lazysammy.io import save_auto_mask_result
from lazysammy.types import AutoMask, AutoMaskResult, ModelSize
from lazysammy.utils import (
    auto_detect_device,
    autocast,
    get_autocast_dtype,
    load_image,
)

logger = logging.getLogger(__name__)


class AutoSegmenter:
    """Automatic "segment everything" powered by SAM2.

    Wraps :class:`SAM2AutomaticMaskGenerator` with a simplified interface.
    Given an image, generates masks for all detectable objects without any
    user prompts.

    Example::

        auto = AutoSegmenter("large")
        result = auto.generate("photo.jpg")
        for m in result.masks:
            print(f"area={m.area}, score={m.score:.2f}")
    """

    def __init__(
        self,
        model_size: str | ModelSize = "large",
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 0,
        use_m2m: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialise the automatic segmenter.

        Args:
            model_size: Model size or HuggingFace model ID.
            checkpoint: Optional local checkpoint path.
            device: Device override.
            points_per_side: Grid density for point sampling.
            pred_iou_thresh: Keep masks with predicted IoU above this.
            stability_score_thresh: Keep masks with stability above this.
            min_mask_region_area: Remove small disconnected regions (in pixels).
            use_m2m: Enable mask-to-mask refinement step.
            **kwargs: Extra arguments forwarded to the generator constructor.
        """
        from lazysammy.models import load_auto_mask_generator

        self._generator = load_auto_mask_generator(
            model_size,
            checkpoint=checkpoint,
            device=device,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            use_m2m=use_m2m,
            **kwargs,
        )
        self._device = torch.device(auto_detect_device(device))
        self._dtype = get_autocast_dtype(self._device)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        image: str | Path | npt.NDArray[np.uint8],
    ) -> AutoMaskResult:
        """Generate masks for all objects in an image.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.

        Returns:
            :class:`AutoMaskResult` containing all detected masks.
        """
        img = load_image(image)

        with torch.inference_mode(), autocast(self._device):
            raw_anns = self._generator.generate(img)

        masks = [
            AutoMask(
                data=(
                    ann["segmentation"].astype(bool)
                    if isinstance(ann["segmentation"], np.ndarray)
                    else np.zeros(img.shape[:2], dtype=bool)
                ),
                score=ann["predicted_iou"],
                area=ann["area"],
                bbox=ann["bbox"],
                stability_score=ann["stability_score"],
                point_coords=ann["point_coords"],
                crop_box=ann["crop_box"],
            )
            for ann in raw_anns
        ]
        # Sort by area (largest first)
        masks.sort(key=lambda m: m.area, reverse=True)

        logger.info("Generated %d masks", len(masks))
        return AutoMaskResult(masks=masks, image_shape=img.shape[:2])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        result: AutoMaskResult,
        output_dir: str | Path,
        *,
        fmt: str = "png",
    ) -> Path:
        """Save auto-mask results to disk.

        Args:
            result: The :class:`AutoMaskResult` to save.
            output_dir: Target directory.
            fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.

        Returns:
            Path to the output.
        """
        return save_auto_mask_result(result, output_dir, fmt=fmt)

    @property
    def device(self) -> torch.device:
        """Device in use."""
        return self._device

    @property
    def generator(self) -> Any:
        """Access the underlying ``SAM2AutomaticMaskGenerator``."""
        return self._generator
