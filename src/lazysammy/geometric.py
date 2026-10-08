"""Shared geometric-prompt path for the SAM 2 and SAM 3 image segmenters.

Both :class:`~lazysammy.image.ImageSegmenter` and
:class:`~lazysammy.concept.ConceptSegmenter` expose the SAM 1/2-style
interactive task: encode an image once, then prompt it with points, a box, or
a mask. This mixin owns that path so the two share one implementation and
differ only in the model-specific inference call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from lazysammy.types import (
    BoundingBox,
    Mask,
    MaskLogits,
    PointCoords,
    PointLabels,
)
from lazysammy.utils import ImageEmbeddingCache, autocast, load_image
from lazysammy.validation import normalize_box


def build_masks(
    masks_np: npt.NDArray[Any],
    scores_np: npt.NDArray[Any],
    logits_np: npt.NDArray[Any] | None,
    *,
    return_logits: bool = False,
) -> list[Mask]:
    """Build :class:`Mask` objects from raw predictor arrays.

    When *return_logits* is ``True`` the predictor returns raw logits rather
    than a boolean mask, so the values are thresholded at ``0`` (positive
    logits are foreground) before being stored as booleans.
    """
    return [
        Mask(
            data=(masks_np[i] > 0.0 if return_logits else masks_np[i].astype(bool)),
            score=float(scores_np[i]),
            logits=(logits_np[i] if logits_np is not None else None),
        )
        for i in range(len(masks_np))
    ]


class GeometricPromptMixin:
    """Image encoding plus point/box/mask prompting, shared by both segmenters.

    Subclasses must set ``_device`` and ``_image_cache`` and implement
    :meth:`_set_image_embedding` and :meth:`_run_geometric`.
    """

    _device: torch.device
    _image_cache: ImageEmbeddingCache

    # ------------------------------------------------------------------
    # Model-specific hooks
    # ------------------------------------------------------------------

    def _set_image_embedding(self, img: npt.NDArray[np.uint8]) -> None:
        """Store the encoded embedding for *img*. Implemented by subclasses."""
        raise NotImplementedError

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
        """Run the model-specific inference call. Implemented by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Image encoding (set once, prompt many)
    # ------------------------------------------------------------------

    def set_image(self, image: str | Path | npt.NDArray[np.uint8]) -> tuple[int, int]:
        """Encode an image once so repeated prompts reuse the embedding.

        Calling this is optional: the prompt methods encode the image
        automatically and reuse the cached embedding when the same image is
        passed again. Call it explicitly to pre-warm the encoder.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.

        Returns:
            The ``(H, W)`` shape of the encoded image.
        """
        img = load_image(image)
        with torch.inference_mode(), autocast(self._device):
            self._set_image_embedding(img)
        self._image_cache.store(image, img.shape[:2])
        return img.shape[:2]

    def _ensure_image(self, image: str | Path | npt.NDArray[np.uint8]) -> tuple[int, int]:
        """Encode *image* unless the same image is already cached."""
        if self._image_cache.matches(image):
            shape = self._image_cache.shape
            assert shape is not None
            return shape
        return self.set_image(image)

    # ------------------------------------------------------------------
    # Geometric prediction
    # ------------------------------------------------------------------

    def _predict_geometric(
        self,
        *,
        points: PointCoords | None,
        labels: PointLabels | None,
        box: BoundingBox | None,
        mask_input: MaskLogits | None,
        multimask_output: bool,
        return_logits: bool,
    ) -> list[Mask]:
        """Normalize prompts, run inference, and build the mask list.

        Shared by :meth:`segment`, :meth:`predict`, and the multi-object
        helpers so the mask-input normalization and inference wrapper live in
        exactly one place.
        """
        # SAM2's ``_prep_prompts`` only adds a batch dimension to 3D mask
        # logits, so a bare 2D ``(H, W)`` mask must be promoted to
        # ``(1, H, W)`` here to avoid a shape error deep in the model.
        mask_input_norm = mask_input
        if mask_input_norm is not None and mask_input_norm.ndim == 2:
            mask_input_norm = mask_input_norm[None, :, :]

        with torch.inference_mode(), autocast(self._device):
            pt = np.array(points, dtype=np.float32) if points is not None else None
            lb = np.array(labels, dtype=np.int32) if labels is not None else None
            bx = normalize_box(box) if box is not None else None
            masks_np, scores_np, logits_np = self._run_geometric(
                points=pt,
                labels=lb,
                box=bx,
                mask_input=mask_input_norm,
                multimask_output=multimask_output,
                return_logits=return_logits,
            )

        return build_masks(masks_np, scores_np, logits_np, return_logits=return_logits)
