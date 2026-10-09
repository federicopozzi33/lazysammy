"""Shared geometric-prompt path for the SAM 2 and SAM 3 image segmenters.

Both :class:`~lazysammy.image.ImageSegmenter` and
:class:`~lazysammy.concept.ConceptSegmenter` expose the SAM 1/2-style
interactive task: encode an image once, then prompt it with points, a box, or
a mask. This mixin owns that path so the two share one implementation and
differ only in the model-specific inference calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar

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
from lazysammy.validation import (
    normalize_box,
    validate_points_and_labels,
    validate_sequence_length,
)

# The backend's prediction type (ImagePrediction or ConceptPrediction).
PredT = TypeVar("PredT")


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


class GeometricPromptMixin(Generic[PredT]):
    """Image encoding plus point/box/mask prompting, shared by both segmenters.

    Generic over the backend's prediction type so ``ImageSegmenter`` yields
    :class:`~lazysammy.types.ImagePrediction` and ``ConceptSegmenter`` yields
    :class:`~lazysammy.types.ConceptPrediction` from the very same code.

    Subclasses must set ``_device`` and ``_image_cache`` and implement the
    model-specific hooks (:meth:`_set_image_embedding`,
    :meth:`_run_geometric`, :meth:`_wrap_masks`, the batch hooks, and
    :meth:`segment`, which carries the per-backend result type).
    """

    _device: torch.device
    _image_cache: ImageEmbeddingCache

    # ------------------------------------------------------------------
    # Model-specific hooks
    # ------------------------------------------------------------------

    def _set_image_embedding(self, img: npt.NDArray[np.uint8]) -> None:
        """Store the encoded embedding for *img*. Implemented by subclasses."""
        raise NotImplementedError

    def _set_image_embedding_batch(self, imgs: Sequence[npt.NDArray[np.uint8]]) -> None:
        """Store the encoded embeddings for a batch. Implemented by subclasses."""
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

    def _run_geometric_batch(
        self,
        *,
        points_batch: list[npt.NDArray[np.float32] | None],
        labels_batch: list[npt.NDArray[np.int32] | None],
        box_batch: list[npt.NDArray[np.float32] | None],
        multimask_output: bool,
    ) -> tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]], list[npt.NDArray[Any] | None]]:
        """Run the model-specific batched inference call."""
        raise NotImplementedError

    def _wrap_masks(self, masks: list[Mask], image_shape: tuple[int, int]) -> PredT:
        """Wrap predicted masks in the backend's prediction type."""
        raise NotImplementedError

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
    ) -> PredT:
        """Segment an image with geometric prompts. Implemented by subclasses."""
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

        Shared by :meth:`segment`, the convenience shortcuts, and the
        multi-object helpers so the mask-input normalization and inference
        wrapper live in exactly one place.
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

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------

    def segment_point(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        x: float,
        y: float,
        *,
        foreground: bool = True,
        multimask_output: bool = True,
    ) -> PredT:
        """Segment with a single point click.

        Args:
            image: Image to segment.
            x: X coordinate of the click.
            y: Y coordinate of the click.
            foreground: ``True`` for a positive click, ``False`` for negative.
            multimask_output: Return multiple masks.

        Returns:
            The backend's prediction type.
        """
        return self.segment(
            image,
            points=[[x, y]],
            labels=[1 if foreground else 0],
            multimask_output=multimask_output,
        )

    def segment_box(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> PredT:
        """Segment with a bounding box prompt.

        Args:
            image: Image to segment.
            x1: Left x of the box.
            y1: Top y of the box.
            x2: Right x of the box.
            y2: Bottom y of the box.

        Returns:
            The backend's prediction type (single mask for a box prompt).
        """
        return self.segment(
            image,
            box=[x1, y1, x2, y2],
            multimask_output=False,
        )

    def segment_multi_box(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        boxes: Sequence[Sequence[float]],
    ) -> list[PredT]:
        """Segment multiple objects in one image, each specified by a box.

        Args:
            image: Image to segment.
            boxes: List of ``[x1, y1, x2, y2]`` bounding boxes.

        Returns:
            One prediction per box.
        """
        if not boxes:
            return []
        normalized_boxes = [
            normalize_box(box, name=f"boxes[{idx}]") for idx, box in enumerate(boxes)
        ]
        image_shape = self._ensure_image(image)
        return [
            self._wrap_masks(
                self._predict_geometric(
                    points=None,
                    labels=None,
                    box=bx,
                    mask_input=None,
                    multimask_output=False,
                    return_logits=False,
                ),
                image_shape,
            )
            for bx in normalized_boxes
        ]

    def segment_multi_point(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        points_per_object: Sequence[Sequence[Sequence[float]]],
        *,
        labels_per_object: Sequence[Sequence[int]] | None = None,
        multimask_output: bool = False,
    ) -> list[PredT]:
        """Segment multiple objects in one image, each with its own points.

        The image is encoded once and each object's points are predicted in
        turn, so this is much cheaper than calling :meth:`segment` per object.

        Args:
            image: Image to segment.
            points_per_object: One ``(N, 2)`` list of ``(x, y)`` points per object.
            labels_per_object: One length-N label list per object. Defaults to
                all-foreground (``1``) for every point.
            multimask_output: Return 3 masks per object for ambiguous prompts.

        Returns:
            One prediction per object.

        Raises:
            ValueError: If the per-object lists have inconsistent lengths.
        """
        if not points_per_object:
            return []
        if labels_per_object is not None and len(labels_per_object) != len(points_per_object):
            msg = (
                "labels_per_object must have the same length as points_per_object; "
                f"got {len(labels_per_object)} and {len(points_per_object)}."
            )
            raise ValueError(msg)

        image_shape = self._ensure_image(image)
        results: list[PredT] = []
        for idx, pts in enumerate(points_per_object):
            labels = labels_per_object[idx] if labels_per_object is not None else [1] * len(pts)
            try:
                validate_points_and_labels(pts, labels)
            except ValueError as exc:
                msg = f"Invalid points for object index {idx}: {exc}"
                raise ValueError(msg) from exc

            results.append(
                self._wrap_masks(
                    self._predict_geometric(
                        points=pts,
                        labels=labels,
                        box=None,
                        mask_input=None,
                        multimask_output=multimask_output,
                        return_logits=False,
                    ),
                    image_shape,
                )
            )
        return results

    def refine(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        previous_logits: npt.NDArray[np.floating[Any]],
        *,
        points: PointCoords | None = None,
        labels: PointLabels | None = None,
    ) -> PredT:
        """Refine a previous prediction with additional prompts.

        Uses the low-resolution logits from a prior prediction as a mask
        input for iterative refinement.

        Args:
            image: Same image used in the previous prediction.
            previous_logits: Low-res logits from :attr:`Mask.logits`.
            points: Additional point prompts.
            labels: Labels for the additional points.

        Returns:
            A refined prediction of the backend's result type.
        """
        validate_points_and_labels(points, labels)
        if previous_logits.ndim not in {2, 3}:
            msg = (
                "previous_logits must be a 2D mask-logit array or a 3D batched "
                f"mask-logit array; got shape {previous_logits.shape!r}."
            )
            raise ValueError(msg)

        logit_input = previous_logits[None, :, :] if previous_logits.ndim == 2 else previous_logits
        return self.segment(
            image,
            points=points,
            labels=labels,
            mask_input=logit_input,
            multimask_output=False,
        )

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def segment_batch(
        self,
        images: Sequence[str | Path | npt.NDArray[np.uint8]],
        *,
        points_batch: Sequence[npt.NDArray[np.floating[Any]] | None] | None = None,
        labels_batch: Sequence[npt.NDArray[np.integer[Any]] | None] | None = None,
        box_batch: Sequence[npt.NDArray[np.floating[Any]] | None] | None = None,
        multimask_output: bool = True,
    ) -> list[PredT]:
        """Segment a batch of images with per-image prompts.

        Args:
            images: List of file paths or ``(H, W, 3)`` arrays.
            points_batch: Per-image point coords (``None`` to skip an image).
            labels_batch: Per-image point labels.
            box_batch: Per-image bounding boxes.
            multimask_output: Whether to return multiple masks.

        Returns:
            One prediction per input image.
        """
        validate_sequence_length("points_batch", points_batch, len(images))
        validate_sequence_length("labels_batch", labels_batch, len(images))
        validate_sequence_length("box_batch", box_batch, len(images))

        if points_batch is not None and labels_batch is None:
            msg = "labels_batch must be provided when points_batch is provided."
            raise ValueError(msg)
        if labels_batch is not None and points_batch is None:
            msg = "points_batch must be provided when labels_batch is provided."
            raise ValueError(msg)
        if points_batch is not None and labels_batch is not None:
            for idx, (points, labels) in enumerate(zip(points_batch, labels_batch, strict=False)):
                try:
                    validate_points_and_labels(points, labels)
                except ValueError as exc:
                    msg = f"Invalid prompts for image index {idx}: {exc}"
                    raise ValueError(msg) from exc
        if box_batch is not None:
            for idx, box in enumerate(box_batch):
                if box is None:
                    continue
                try:
                    normalize_box(box, name=f"box_batch[{idx}]")
                except ValueError as exc:
                    msg = f"Invalid box for image index {idx}: {exc}"
                    raise ValueError(msg) from exc

        imgs = [load_image(im) for im in images]

        with torch.inference_mode(), autocast(self._device):
            self._set_image_embedding_batch(imgs)
            all_masks, all_scores, all_logits = self._run_geometric_batch(
                points_batch=[
                    (np.array(p, dtype=np.float32) if p is not None else None)
                    for p in (points_batch or [None] * len(imgs))
                ],
                labels_batch=[
                    (np.array(lb, dtype=np.int32) if lb is not None else None)
                    for lb in (labels_batch or [None] * len(imgs))
                ],
                box_batch=[
                    (normalize_box(b) if b is not None else None)
                    for b in (box_batch or [None] * len(imgs))
                ],
                multimask_output=multimask_output,
            )

        return [
            self._wrap_masks(
                build_masks(all_masks[i], all_scores[i], all_logits[i]),
                imgs[i].shape[:2],
            )
            for i in range(len(imgs))
        ]
