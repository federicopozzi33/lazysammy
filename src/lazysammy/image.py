"""High-level image segmentation wrapper around SAM2ImagePredictor."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

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
from lazysammy.utils import (
    ImageEmbeddingCache,
    auto_detect_device,
    autocast,
    load_image,
)
from lazysammy.validation import (
    normalize_box,
    validate_box,
    validate_points_and_labels,
    validate_segment_prompts,
    validate_sequence_length,
)

logger = logging.getLogger(__name__)


class ImageSegmenter:
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

    def set_image(self, image: str | Path | npt.NDArray[np.uint8]) -> tuple[int, int]:
        """Encode an image once so repeated prompts reuse the embedding.

        Calling this is optional: :meth:`segment` and :meth:`refine` encode the
        image automatically and reuse the cached embedding when the same image
        is passed again. Call it explicitly to pre-warm the encoder.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.

        Returns:
            The ``(H, W)`` shape of the encoded image.
        """
        img = load_image(image)
        with torch.inference_mode(), autocast(self._device):
            self._predictor.set_image(img)
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
    # Core prediction
    # ------------------------------------------------------------------

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
        mask_objs = [
            Mask(
                data=(masks_np[i] > 0.0 if return_logits else masks_np[i].astype(bool)),
                score=float(scores_np[i]),
                logits=(logits_np[i] if logits_np is not None else None),
            )
            for i in range(len(masks_np))
        ]
        return ImagePrediction(masks=mask_objs, image_shape=image_shape)

    def _predict_prompts(
        self,
        *,
        image_shape: tuple[int, int],
        points: PointCoords | None,
        labels: PointLabels | None,
        box: BoundingBox | None,
        mask_input: MaskLogits | None,
        multimask_output: bool,
        return_logits: bool,
    ) -> ImagePrediction:
        """Run already-validated prompts through the predictor.

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

            masks_np, scores_np, logits_np = self._predictor.predict(
                point_coords=pt,
                point_labels=lb,
                box=bx,
                mask_input=mask_input_norm,
                multimask_output=multimask_output,
                return_logits=return_logits,
            )

        return self._to_image_prediction(
            masks_np,
            scores_np,
            logits_np,
            image_shape=image_shape,
            return_logits=return_logits,
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
        return self._predict_prompts(
            image_shape=image_shape,
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

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
        return self._predict_prompts(
            image_shape=image_shape,
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

    def segment_batch(
        self,
        images: Sequence[str | Path | npt.NDArray[np.uint8]],
        *,
        points_batch: Sequence[npt.NDArray[np.floating[Any]] | None] | None = None,
        labels_batch: Sequence[npt.NDArray[np.integer[Any]] | None] | None = None,
        box_batch: Sequence[npt.NDArray[np.floating[Any]] | None] | None = None,
        multimask_output: bool = True,
    ) -> list[ImagePrediction]:
        """Segment a batch of images with per-image prompts.

        Args:
            images: List of file paths or ``(H, W, 3)`` arrays.
            points_batch: Per-image point coords (``None`` to skip an image).
            labels_batch: Per-image point labels.
            box_batch: Per-image bounding boxes.
            multimask_output: Whether to return multiple masks.

        Returns:
            A list of :class:`ImagePrediction`, one per input image.
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
                try:
                    validate_box(box)
                except ValueError as exc:
                    msg = f"Invalid box for image index {idx}: {exc}"
                    raise ValueError(msg) from exc

        imgs = [load_image(im) for im in images]

        with torch.inference_mode(), autocast(self._device):
            self._predictor.set_image_batch(imgs)
            all_masks, all_scores, all_logits = self._predictor.predict_batch(
                point_coords_batch=[
                    (np.array(p, dtype=np.float32) if p is not None else None)
                    for p in (points_batch or [None] * len(imgs))
                ],
                point_labels_batch=[
                    (np.array(lb, dtype=np.int32) if lb is not None else None)
                    for lb in (labels_batch or [None] * len(imgs))
                ],
                box_batch=[
                    (normalize_box(b) if b is not None else None)
                    for b in (box_batch or [None] * len(imgs))
                ],
                multimask_output=multimask_output,
            )

        results: list[ImagePrediction] = []
        for i in range(len(imgs)):
            masks_np = all_masks[i]
            scores_np = all_scores[i]
            logits_np = all_logits[i]
            results.append(
                self._to_image_prediction(
                    masks_np,
                    scores_np,
                    logits_np,
                    image_shape=imgs[i].shape[:2],
                )
            )
        return results

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
    ) -> ImagePrediction:
        """Segment with a single point click.

        Args:
            image: Image to segment.
            x: X coordinate of the click.
            y: Y coordinate of the click.
            foreground: ``True`` for a positive click, ``False`` for negative.
            multimask_output: Return multiple masks.

        Returns:
            :class:`ImagePrediction`.
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
    ) -> ImagePrediction:
        """Segment with a bounding box prompt.

        Args:
            image: Image to segment.
            x1: Left x of the box.
            y1: Top y of the box.
            x2: Right x of the box.
            y2: Bottom y of the box.

        Returns:
            :class:`ImagePrediction` (single mask output for non-ambiguous prompt).
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
    ) -> list[ImagePrediction]:
        """Segment multiple objects in one image, each specified by a box.

        Args:
            image: Image to segment.
            boxes: List of ``[x1, y1, x2, y2]`` bounding boxes.

        Returns:
            A list of :class:`ImagePrediction`, one per box.
        """
        if not boxes:
            return []
        normalized_boxes = [
            normalize_box(box, name=f"boxes[{idx}]") for idx, box in enumerate(boxes)
        ]
        image_shape = self._ensure_image(image)
        results: list[ImagePrediction] = []

        for bx in normalized_boxes:
            results.append(
                self._predict_prompts(
                    image_shape=image_shape,
                    points=None,
                    labels=None,
                    box=bx,
                    mask_input=None,
                    multimask_output=False,
                    return_logits=False,
                )
            )
        return results

    def segment_multi_point(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        points_per_object: Sequence[Sequence[Sequence[float]]],
        *,
        labels_per_object: Sequence[Sequence[int]] | None = None,
        multimask_output: bool = False,
    ) -> list[ImagePrediction]:
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
            A list of :class:`ImagePrediction`, one per object.

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
        results: list[ImagePrediction] = []

        for idx, pts in enumerate(points_per_object):
            labels = labels_per_object[idx] if labels_per_object is not None else [1] * len(pts)
            try:
                validate_points_and_labels(pts, labels)
            except ValueError as exc:
                msg = f"Invalid points for object index {idx}: {exc}"
                raise ValueError(msg) from exc

            results.append(
                self._predict_prompts(
                    image_shape=image_shape,
                    points=pts,
                    labels=labels,
                    box=None,
                    mask_input=None,
                    multimask_output=multimask_output,
                    return_logits=False,
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
    ) -> ImagePrediction:
        """Refine a previous prediction with additional prompts.

        Uses the low-resolution logits from a prior prediction as a mask
        input for iterative refinement.

        Args:
            image: Same image used in the previous prediction.
            previous_logits: Low-res logits from :attr:`Mask.logits`.
            points: Additional point prompts.
            labels: Labels for the additional points.

        Returns:
            A refined :class:`ImagePrediction`.
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
