"""High-level image segmentation wrapper around SAM2ImagePredictor."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from lazysammy.geometric import GeometricPromptMixin, build_masks
from lazysammy.io import save_image_prediction
from lazysammy.types import (
    BoundingBox,
    ImagePrediction,
    Mask,
    MaskLogits,
    ModelSize,
    PointCoords,
    PointLabels,
)
from lazysammy.utils import ImageEmbeddingCache, auto_detect_device
from lazysammy.validation import validate_segment_prompts

logger = logging.getLogger(__name__)


class ImageSegmenter(GeometricPromptMixin[ImagePrediction]):
    """Easy-to-use image segmentation powered by SAM2.

    Wraps :class:`SAM2ImagePredictor` with a simpler API that handles
    device management, autocast, and result packaging automatically.

    Example::

        seg = ImageSegmenter("large")
        result = seg.segment("photo.jpg", points=[[100, 200]], labels=[1])
        best = result.best_mask
    """

    def __init__(
        self,
        model_size: str | ModelSize = "large",
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the image segmenter.

        Args:
            model_size: Model size (``"tiny"``, ``"small"``, ``"base_plus"``, ``"large"``)
                or a HuggingFace model ID.
            checkpoint: Optional path to a local ``.pt`` checkpoint.
            device: Device override (auto-detected if ``None``).
            **kwargs: Extra arguments forwarded to the predictor constructor.
        """
        from lazysammy.models import load_image_predictor

        self._predictor = load_image_predictor(
            model_size, checkpoint=checkpoint, device=device, **kwargs
        )
        self._device = torch.device(auto_detect_device(device))
        # Cache the encoded image so repeated prompts on the same image skip
        # the (expensive) image encoder.
        self._image_cache = ImageEmbeddingCache()

    # ------------------------------------------------------------------
    # Configuration accessors
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device used by this segmenter."""
        return self._device

    @property
    def predictor(self) -> Any:
        """Access the underlying ``SAM2ImagePredictor``."""
        return self._predictor

    # ------------------------------------------------------------------
    # Image encoding (set once, prompt many)
    # ------------------------------------------------------------------

    def _set_image_embedding(self, img: npt.NDArray[np.uint8]) -> None:
        """Store the SAM 2 image embedding for *img*."""
        self._predictor.set_image(img)

    def _set_image_embedding_batch(self, imgs: Sequence[npt.NDArray[np.uint8]]) -> None:
        """Store the SAM 2 image embeddings for a batch."""
        self._predictor.set_image_batch(imgs)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def _wrap_masks(self, masks: list[Mask], image_shape: tuple[int, int]) -> ImagePrediction:
        """Wrap predicted masks in an :class:`ImagePrediction`."""
        return ImagePrediction(masks=masks, image_shape=image_shape)

    @staticmethod
    def _to_image_prediction(
        masks_np: npt.NDArray[np.bool_] | npt.NDArray[np.floating[Any]],
        scores_np: npt.NDArray[np.floating[Any]],
        logits_np: npt.NDArray[np.floating[Any]] | None,
        *,
        image_shape: tuple[int, int],
        return_logits: bool = False,
    ) -> ImagePrediction:
        """Convert raw predictor arrays into :class:`ImagePrediction`.

        When *return_logits* is ``True`` the predictor returns raw logits
        rather than a boolean mask, so the values are thresholded at ``0``
        (positive logits are foreground) before being stored as booleans.
        """
        return ImagePrediction(
            masks=build_masks(masks_np, scores_np, logits_np, return_logits=return_logits),
            image_shape=image_shape,
        )

    def _run_geometric(
        self,
        *,
        points: npt.NDArray[np.float32] | None,
        labels: npt.NDArray[np.int32] | None,
        box: npt.NDArray[np.float32] | None,
        mask_input: npt.NDArray[np.floating[Any]] | None,
        multimask_output: bool,
        return_logits: bool,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any] | None]:
        """Run the SAM 2 predictor on already-normalized prompts."""
        result = self._predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )
        return cast("tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any] | None]", result)

    def _run_geometric_batch(
        self,
        *,
        points_batch: list[npt.NDArray[np.float32] | None],
        labels_batch: list[npt.NDArray[np.int32] | None],
        box_batch: list[npt.NDArray[np.float32] | None],
        multimask_output: bool,
    ) -> tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]], list[npt.NDArray[Any] | None]]:
        """Run the SAM 2 batch predictor on already-normalized prompts."""
        result = self._predictor.predict_batch(
            point_coords_batch=points_batch,
            point_labels_batch=labels_batch,
            box_batch=box_batch,
            multimask_output=multimask_output,
        )
        return cast(
            "tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]], list[npt.NDArray[Any] | None]]",
            result,
        )

    def segment(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        *,
        points: PointCoords | None = None,
        labels: PointLabels | None = None,
        box: BoundingBox | None = None,
        mask_input: MaskLogits | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> ImagePrediction:
        """Segment an image with point/box/mask prompts.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.
            points: ``(N, 2)`` array of ``(x, y)`` point prompts.
            labels: Length-N label array (``1`` = foreground, ``0`` = background).
            box: ``[x1, y1, x2, y2]`` bounding box prompt.
            mask_input: Low-resolution mask **logits** (e.g. ``1 x 256 x 256``)
                from a previous prediction, as returned in :attr:`Mask.logits`.
            multimask_output: Return 3 masks for ambiguous prompts.
            return_logits: Keep raw logits instead of thresholding.

        Returns:
            An :class:`ImagePrediction` with the predicted masks.
        """
        validate_segment_prompts(
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
        )
        image_shape = self._ensure_image(image)
        masks = self._predict_geometric(
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )
        return ImagePrediction(masks=masks, image_shape=image_shape)

    def predict(
        self,
        *,
        points: PointCoords | None = None,
        labels: PointLabels | None = None,
        box: BoundingBox | None = None,
        mask_input: MaskLogits | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> ImagePrediction:
        """Run prompts against the image set by :meth:`set_image`.

        This is the prompt-only half of :meth:`segment`: it skips image
        encoding entirely, so it is the fast path for many prompts on one
        image.

        Args:
            points: ``(N, 2)`` array of ``(x, y)`` point prompts.
            labels: Length-N label array (``1`` = foreground, ``0`` = background).
            box: ``[x1, y1, x2, y2]`` bounding box prompt.
            mask_input: Low-resolution mask **logits** from a previous prediction.
            multimask_output: Return 3 masks for ambiguous prompts.
            return_logits: Keep raw logits instead of thresholding.

        Returns:
            An :class:`ImagePrediction` with the predicted masks.

        Raises:
            RuntimeError: If no image has been set with :meth:`set_image`.
        """
        image_shape = self._image_cache.shape
        if image_shape is None:
            msg = "No image is set. Call set_image(image) before predict()."
            raise RuntimeError(msg)

        validate_segment_prompts(
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
        )
        masks = self._predict_geometric(
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )
        return ImagePrediction(masks=masks, image_shape=image_shape)

    # ------------------------------------------------------------------
    # Save helper
    # ------------------------------------------------------------------

    def save(
        self,
        prediction: ImagePrediction,
        output_dir: str | Path,
        *,
        fmt: str = "png",
    ) -> Path:
        """Save an image prediction to disk.

        Args:
            prediction: The result to save.
            output_dir: Output directory.
            fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.

        Returns:
            Path to the output.
        """
        return save_image_prediction(prediction, output_dir, fmt=fmt)
