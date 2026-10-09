"""SAM 3 open-vocabulary concept segmentation for images.

SAM 3 extends SAM 2's geometric prompting (points, boxes, masks) with
*concept* prompts: a short text phrase or an image exemplar. Given a concept,
SAM 3 exhaustively detects and segments every matching instance in an image.

:class:`ConceptSegmenter` exposes the same image API as
:class:`~lazysammy.image.ImageSegmenter` - geometric prompts, convenience
shortcuts, multi-object helpers, refinement, and batching, all through the
shared :class:`~lazysammy.geometric.GeometricPromptMixin` - so it is a drop-in
replacement. On top of that it adds :meth:`segment_text`,
:meth:`segment_exemplar`, and :meth:`to_image_prediction`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from lazysammy.geometric import GeometricPromptMixin, build_masks
from lazysammy.types import (
    BoundingBox,
    ConceptPrediction,
    ImagePrediction,
    Mask,
    MaskLogits,
    PointCoords,
    PointLabels,
)
from lazysammy.utils import ImageEmbeddingCache, auto_detect_device, autocast
from lazysammy.validation import (
    normalize_box,
    validate_box,
    validate_segment_prompts,
)

logger = logging.getLogger(__name__)


class ConceptSegmenter(GeometricPromptMixin[ConceptPrediction]):
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
        self._image_cache = ImageEmbeddingCache()

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

    def _set_image_embedding(self, img: npt.NDArray[np.uint8]) -> None:
        """Store the SAM 3 processor state for *img*."""
        self._state = self._processor.set_image(img)

    def _set_image_embedding_batch(self, imgs: Sequence[npt.NDArray[np.uint8]]) -> None:
        """Store the SAM 3 processor state for a batch of images."""
        self._state = self._processor.set_image_batch(list(imgs))

    def reset(self) -> None:
        """Clear all prompts and results for the current image."""
        if self._state is not None:
            self._processor.reset_all_prompts(self._state)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def _wrap_masks(self, masks: list[Mask], image_shape: tuple[int, int]) -> ConceptPrediction:
        """Wrap predicted masks in a :class:`ConceptPrediction`.

        Geometric prompts produce masks without bounding boxes, so ``boxes``
        stays empty here; the concept paths fill it in via
        :meth:`_to_prediction`.
        """
        return ConceptPrediction(masks=masks, image_shape=image_shape)

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
        """Run the SAM 3 interactive predictor on already-normalized prompts."""
        bx = box[None, :] if box is not None else None
        result = self._model.predict_inst(
            self._state,
            point_coords=points,
            point_labels=labels,
            box=bx,
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
        """Run the SAM 3 batch predictor on already-normalized prompts."""
        boxes = [b[None, :] if b is not None else None for b in box_batch]
        result = self._model.predict_inst_batch(
            self._state,
            points_batch,
            labels_batch,
            box_batch=boxes,
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
        masks = self._predict_geometric(
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )
        return ConceptPrediction(masks=masks, image_shape=image_shape)

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
        return ConceptPrediction(masks=masks, image_shape=image_shape)

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
            # Set the threshold without a state so the processor does not
            # re-run grounding on the previous results; the text prompt below
            # runs grounding once with the new threshold.
            self._processor.set_confidence_threshold(confidence_threshold)
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
            # Set the threshold without a state so the processor does not
            # re-run grounding on the previous results; the geometric prompt
            # below runs grounding once with the new threshold.
            self._processor.set_confidence_threshold(confidence_threshold)
        cxcywh = self._to_normalized_cxcywh(box, image_shape)
        with torch.inference_mode(), autocast(self._device):
            state = self._processor.add_geometric_prompt(box=cxcywh, label=label, state=self._state)
        return self._to_prediction(state, concept="visual", image_shape=image_shape)

    # ------------------------------------------------------------------
    # Result conversion
    # ------------------------------------------------------------------

    def to_image_prediction(self, prediction: ConceptPrediction) -> ImagePrediction:
        """Convert a concept prediction into an :class:`ImagePrediction`.

        The two result types are lossy in opposite directions: concept
        predictions carry boxes, image predictions carry logits. This bridges
        them for code that only understands the SAM 2 result shape, dropping
        the boxes and the ``concept`` label.
        """
        return ImagePrediction(masks=list(prediction.masks), image_shape=prediction.image_shape)

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
        """Convert a ``Sam3Processor`` state dict into a :class:`ConceptPrediction`.

        A processor state either has no detections (no ``masks`` key) or has
        ``masks``, ``scores``, and ``boxes`` together: ``_forward_grounding``
        sets all three as a unit. The three are therefore read as a unit, so a
        state that has masks but is missing scores or boxes fails loudly rather
        than silently producing a truncated prediction.
        """
        masks_arr = state.get("masks")
        if masks_arr is None:
            return ConceptPrediction(masks=[], concept=concept, image_shape=image_shape)

        masks_np = np.asarray(masks_arr)
        if masks_np.ndim == 4:  # (N, 1, H, W)
            masks_np = masks_np[:, 0]
        scores = np.asarray(state["scores"], dtype=np.float32).reshape(-1)
        boxes = np.asarray(state["boxes"], dtype=np.float32).reshape(-1, 4)

        return ConceptPrediction(
            masks=build_masks(masks_np, scores, None),
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
