"""SAM 3 open-vocabulary concept segmentation for images.

SAM 3 extends SAM 2's geometric prompting (points, boxes, masks) with
*concept* prompts: a short text phrase or an image exemplar. Given a concept,
SAM 3 exhaustively detects and segments every matching instance in an image.

This module wraps ``Sam3Processor`` with the same shape of API as
:class:`~lazysammy.image.ImageSegmenter`, so the two are interchangeable for
the geometric-prompt case, while :meth:`ConceptSegmenter.segment_text` and
:meth:`ConceptSegmenter.segment_exemplar` expose the new concept features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from lazysammy.types import (
    BoundingBox,
    ConceptPrediction,
    Mask,
    MaskLogits,
    PointCoords,
    PointLabels,
)
from lazysammy.utils import auto_detect_device, autocast, load_image
from lazysammy.validation import (
    normalize_box,
    validate_box,
    validate_segment_prompts,
)

logger = logging.getLogger(__name__)


class ConceptSegmenter:
    """Open-vocabulary image segmentation powered by SAM 3.

    Example::

        seg = ConceptSegmenter()
        result = seg.segment_text("photo.jpg", "a player in white")
        for mask, box in zip(result.masks, result.boxes):
            print(mask.score, box)

    The geometric-prompt path mirrors :class:`ImageSegmenter`::

        seg.set_image("photo.jpg")
        pred = seg.predict(points=[[100, 200]], labels=[1])
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        compile: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialise the concept segmenter.

        Args:
            checkpoint: Optional local checkpoint path (otherwise downloaded
                from the ``facebook/sam3`` HuggingFace repo).
            device: Device override (auto-detected if ``None``).
            confidence_threshold: Default score threshold for concept
                detections.
            compile: Enable ``torch.compile`` on the model components.
            **kwargs: Extra arguments forwarded to the model builder.
        """
        from lazysammy.models import load_sam3_image_model

        self._device = torch.device(auto_detect_device(device))
        self._model = load_sam3_image_model(
            checkpoint=checkpoint,
            device=device,
            compile=compile,
            **kwargs,
        )
        self._processor = self._make_processor(confidence_threshold)
        self._state: dict[str, Any] | None = None
        self._cached_image_key: Any = None
        self._cached_image_shape: tuple[int, int] | None = None
        self._cached_image_ref: npt.NDArray[np.uint8] | None = None

    def _make_processor(self, confidence_threshold: float) -> Any:
        from sam3.model.sam3_image_processor import Sam3Processor

        return Sam3Processor(
            self._model,
            device=str(self._device),
            confidence_threshold=confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Configuration accessors
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device used by this segmenter."""
        return self._device

    @property
    def model(self) -> Any:
        """Access the underlying ``Sam3Image`` model."""
        return self._model

    @property
    def processor(self) -> Any:
        """Access the underlying ``Sam3Processor``."""
        return self._processor

    # ------------------------------------------------------------------
    # Image encoding (set once, prompt many)
    # ------------------------------------------------------------------

    @staticmethod
    def _image_key(image: str | Path | npt.NDArray[np.uint8]) -> Any:
        """Build a cache key identifying *image* without hashing its pixels.

        Arrays are keyed by identity (``id``) plus shape/dtype; the segmenter
        holds a reference to the cached array so its id cannot be recycled.
        Mutating an array in place is not detected: pass a new array to
        invalidate the cache.
        """
        if isinstance(image, np.ndarray):
            return ("array", image.shape, image.dtype.str, id(image))
        p = Path(image)
        try:
            stat = p.stat()
        except OSError:
            return ("path", str(p), None, None)
        return ("path", str(p.resolve()), stat.st_mtime_ns, stat.st_size)

    def set_image(self, image: str | Path | npt.NDArray[np.uint8]) -> tuple[int, int]:
        """Encode an image once so repeated prompts reuse the embedding.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.

        Returns:
            The ``(H, W)`` shape of the encoded image.
        """
        img = load_image(image)
        with torch.inference_mode(), autocast(self._device):
            self._state = self._processor.set_image(img)
        self._cached_image_key = self._image_key(image)
        self._cached_image_shape = img.shape[:2]
        self._cached_image_ref = image if isinstance(image, np.ndarray) else None
        return img.shape[:2]

    def _ensure_image(self, image: str | Path | npt.NDArray[np.uint8]) -> tuple[int, int]:
        """Encode *image* unless the same image is already cached."""
        key = self._image_key(image)
        if key == self._cached_image_key and self._cached_image_shape is not None:
            return self._cached_image_shape
        return self.set_image(image)

    def reset(self) -> None:
        """Clear all prompts and results for the current image."""
        if self._state is not None:
            self._processor.reset_all_prompts(self._state)

    # ------------------------------------------------------------------
    # Concept prompts (SAM 3 new features)
    # ------------------------------------------------------------------

    def segment_text(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        text: str,
        *,
        confidence_threshold: float | None = None,
    ) -> ConceptPrediction:
        """Segment every instance of a text concept in an image.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.
            text: A short noun phrase, e.g. ``"a player in white"``.
            confidence_threshold: Override the default score threshold.

        Returns:
            A :class:`ConceptPrediction` with one mask per detected instance.
        """
        image_shape = self._ensure_image(image)
        if confidence_threshold is not None:
            self._processor.set_confidence_threshold(confidence_threshold, self._state)
        with torch.inference_mode(), autocast(self._device):
            state = self._processor.set_text_prompt(state=self._state, prompt=text)
        return self._to_prediction(state, concept=text, image_shape=image_shape)

    def segment_exemplar(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        box: BoundingBox,
        *,
        label: bool = True,
        confidence_threshold: float | None = None,
    ) -> ConceptPrediction:
        """Segment every instance matching a box exemplar.

        The box is a *visual exemplar*: SAM 3 finds all objects similar to the
        one inside it, rather than segmenting only that box.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.
            box: ``[x1, y1, x2, y2]`` exemplar box in absolute pixels.
            label: ``True`` for a positive exemplar, ``False`` to exclude.
            confidence_threshold: Override the default score threshold.

        Returns:
            A :class:`ConceptPrediction` with one mask per matched instance.
        """
        image_shape = self._ensure_image(image)
        if confidence_threshold is not None:
            self._processor.set_confidence_threshold(confidence_threshold, self._state)
        cxcywh = self._to_normalized_cxcywh(box, image_shape)
        with torch.inference_mode(), autocast(self._device):
            state = self._processor.add_geometric_prompt(box=cxcywh, label=label, state=self._state)
        return self._to_prediction(state, concept="visual", image_shape=image_shape)

    # ------------------------------------------------------------------
    # Geometric prompts (SAM 1/2 task, via the interactive predictor)
    # ------------------------------------------------------------------

    def segment(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        *,
        text: str | None = None,
        points: PointCoords | None = None,
        labels: PointLabels | None = None,
        box: BoundingBox | None = None,
        mask_input: MaskLogits | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> ConceptPrediction:
        """Segment an image with a text concept or geometric prompts.

        When *text* is given, this is concept segmentation (all instances).
        Otherwise it is the SAM 1/2-style interactive task: points, a box, or
        a mask select a single object.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.
            text: Optional text concept; takes precedence over geometry.
            points: ``(N, 2)`` array of ``(x, y)`` point prompts.
            labels: Length-N label array (``1`` = foreground, ``0`` = background).
            box: ``[x1, y1, x2, y2]`` bounding box prompt.
            mask_input: Low-resolution mask logits from a previous prediction.
            multimask_output: Return 3 masks for ambiguous prompts.
            return_logits: Keep raw logits instead of thresholding.

        Returns:
            A :class:`ConceptPrediction` with the predicted masks.
        """
        if text is not None:
            return self.segment_text(image, text)

        validate_segment_prompts(
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
        )
        image_shape = self._ensure_image(image)
        return self._predict_inst(
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
    ) -> ConceptPrediction:
        """Run geometric prompts against the image set by :meth:`set_image`.

        Args:
            points: ``(N, 2)`` array of ``(x, y)`` point prompts.
            labels: Length-N label array (``1`` = foreground, ``0`` = background).
            box: ``[x1, y1, x2, y2]`` bounding box prompt.
            mask_input: Low-resolution mask logits from a previous prediction.
            multimask_output: Return 3 masks for ambiguous prompts.
            return_logits: Keep raw logits instead of thresholding.

        Returns:
            A :class:`ConceptPrediction` with the predicted masks.

        Raises:
            RuntimeError: If no image has been set with :meth:`set_image`.
        """
        if self._cached_image_shape is None:
            msg = "No image is set. Call set_image(image) before predict()."
            raise RuntimeError(msg)
        validate_segment_prompts(
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
        )
        return self._predict_inst(
            image_shape=self._cached_image_shape,
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

    def _predict_inst(
        self,
        *,
        image_shape: tuple[int, int],
        points: PointCoords | None,
        labels: PointLabels | None,
        box: BoundingBox | None,
        mask_input: MaskLogits | None,
        multimask_output: bool,
        return_logits: bool,
    ) -> ConceptPrediction:
        """Run the SAM 1/2-style interactive predictor on the current image."""
        pt = np.array(points, dtype=np.float32) if points is not None else None
        lb = np.array(labels, dtype=np.int32) if labels is not None else None
        bx = normalize_box(box)[None, :] if box is not None else None
        mask_in = mask_input
        if mask_in is not None and mask_in.ndim == 2:
            mask_in = mask_in[None, :, :]

        with torch.inference_mode(), autocast(self._device):
            masks_np, scores_np, logits_np = self._model.predict_inst(
                self._state,
                point_coords=pt,
                point_labels=lb,
                box=bx,
                mask_input=mask_in,
                multimask_output=multimask_output,
                return_logits=return_logits,
            )

        masks = [
            Mask(
                data=(masks_np[i] > 0.0 if return_logits else masks_np[i].astype(bool)),
                score=float(scores_np[i]),
                logits=(logits_np[i] if logits_np is not None else None),
            )
            for i in range(len(masks_np))
        ]
        return ConceptPrediction(masks=masks, image_shape=image_shape)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _to_normalized_cxcywh(box: BoundingBox, image_shape: tuple[int, int]) -> list[float]:
        """Convert an absolute ``[x1, y1, x2, y2]`` box to normalized cxcywh."""
        validate_box(box)
        x1, y1, x2, y2 = (float(v) for v in normalize_box(box))
        height, width = image_shape
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        return [cx, cy, (x2 - x1) / width, (y2 - y1) / height]

    @staticmethod
    def _to_prediction(
        state: dict[str, Any],
        *,
        concept: str | None,
        image_shape: tuple[int, int],
    ) -> ConceptPrediction:
        """Convert a ``Sam3Processor`` state dict into a :class:`ConceptPrediction`."""
        masks_arr = state.get("masks")
        if masks_arr is None:
            return ConceptPrediction(masks=[], concept=concept, image_shape=image_shape)

        masks_np = np.asarray(masks_arr)
        if masks_np.ndim == 4:  # (N, 1, H, W)
            masks_np = masks_np[:, 0]
        scores = np.asarray(state.get("scores", []), dtype=np.float32).reshape(-1)
        boxes = np.asarray(state.get("boxes", []), dtype=np.float32).reshape(-1, 4)

        masks = [
            Mask(data=masks_np[i].astype(bool), score=float(scores[i]))
            for i in range(len(masks_np))
        ]
        return ConceptPrediction(
            masks=masks,
            boxes=[[float(v) for v in box] for box in boxes],
            concept=concept,
            image_shape=image_shape,
        )

    def save(
        self,
        prediction: ConceptPrediction,
        output_dir: str | Path,
        *,
        fmt: str = "png",
    ) -> Path:
        """Save a concept prediction to disk.

        Args:
            prediction: The result to save.
            output_dir: Output directory.
            fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.

        Returns:
            Path to the output directory.
        """
        from lazysammy.io import save_concept_prediction

        return save_concept_prediction(prediction, output_dir, fmt=fmt)
