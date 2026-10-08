"""Unified SAM 3 facade - open-vocabulary segmentation and tracking.

Provides a single :class:`SAM3` object that lazily initialises the image
(concept) segmenter and the video tracker as needed, sharing the same model
configuration. The interface mirrors :class:`~lazysammy.sam2.SAM2` so the two
are interchangeable for geometric prompts, while the concept methods
(:meth:`segment_text`, :meth:`segment_exemplar`, and text video prompts) expose
the capabilities SAM 3 adds over SAM 2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from lazysammy.concept import ConceptSegmenter
from lazysammy.sam3_video import SAM3VideoSession, SAM3VideoTracker
from lazysammy.types import (
    BoundingBox,
    ConceptPrediction,
    MaskLogits,
    PointCoords,
    PointLabels,
)
from lazysammy.utils import auto_detect_device

logger = logging.getLogger(__name__)


class SAM3:
    """Unified, easy-to-use interface to SAM 3's segmentation capabilities.

    A single :class:`SAM3` object lazily creates the image and video predictors
    on first use, so you only pay for what you need.

    Example::

        sam = SAM3()

        # Open-vocabulary image segmentation: every instance of a concept
        pred = sam.segment_text("photo.jpg", "a player in white")
        for mask, box in zip(pred.masks, pred.boxes):
            print(mask.score, box)

        # Geometric prompts (SAM 1/2 task)
        pred = sam.segment("photo.jpg", points=[[100, 200]], labels=[1])

        # Video concept tracking
        session = sam.video("clip.mp4")
        session.add_text(frame_idx=0, text="person")
        results = session.propagate()
        session.save_overlay("tracked.mp4")
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        version: str = "sam3",
        compile: bool = False,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Create a new SAM 3 wrapper.

        Args:
            checkpoint: Optional path to a local checkpoint.
            device: Device string override (auto-detected if ``None``).
            version: ``"sam3"`` for the base model or ``"sam3.1"`` for the
                Object Multiplex video variant.
            compile: Enable ``torch.compile`` on model components.
            confidence_threshold: Default score threshold for concept
                detections.
        """
        self._checkpoint = checkpoint
        self._device = device
        self._version = version
        self._compile = compile
        self._confidence_threshold = confidence_threshold

        self._concept_segmenter: ConceptSegmenter | None = None
        self._video_tracker: SAM3VideoTracker | None = None

    # ------------------------------------------------------------------
    # Configuration accessors
    # ------------------------------------------------------------------

    @property
    def device(self) -> str | None:
        """The configured device override, or ``None`` for auto-detection."""
        return self._device

    @property
    def checkpoint(self) -> str | Path | None:
        """The configured local checkpoint path, if any."""
        return self._checkpoint

    @property
    def version(self) -> str:
        """The configured SAM 3 version (``"sam3"`` or ``"sam3.1"``)."""
        return self._version

    @property
    def resolved_device(self) -> torch.device:
        """The concrete device that will be used, after auto-detection.

        Does not load any weights: it uses an already-created sub-component
        when one exists, and otherwise resolves the device string the same way
        the sub-components will.
        """
        for component in (self._concept_segmenter, self._video_tracker):
            if component is not None:
                return component.device
        return torch.device(auto_detect_device(self._device))

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    @property
    def concept_segmenter(self) -> ConceptSegmenter:
        """Lazily-loaded :class:`ConceptSegmenter`."""
        if self._concept_segmenter is None:
            self._concept_segmenter = ConceptSegmenter(
                checkpoint=self._checkpoint,
                device=self._device,
                confidence_threshold=self._confidence_threshold,
                compile=self._compile,
            )
        return self._concept_segmenter

    @property
    def video_tracker(self) -> SAM3VideoTracker:
        """Lazily-loaded :class:`SAM3VideoTracker`."""
        if self._video_tracker is None:
            self._video_tracker = SAM3VideoTracker(
                checkpoint=self._checkpoint,
                device=self._device,
                version=self._version,
                compile=self._compile,
            )
        return self._video_tracker

    # ------------------------------------------------------------------
    # Concept segmentation (SAM 3 new features)
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
        return self.concept_segmenter.segment_text(
            image, text, confidence_threshold=confidence_threshold
        )

    def segment_exemplar(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        box: BoundingBox,
        *,
        label: bool = True,
        confidence_threshold: float | None = None,
    ) -> ConceptPrediction:
        """Segment every instance matching a box exemplar.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.
            box: ``[x1, y1, x2, y2]`` exemplar box in absolute pixels.
            label: ``True`` for a positive exemplar, ``False`` to exclude.
            confidence_threshold: Override the default score threshold.

        Returns:
            A :class:`ConceptPrediction` with one mask per matched instance.
        """
        return self.concept_segmenter.segment_exemplar(
            image, box, label=label, confidence_threshold=confidence_threshold
        )

    # ------------------------------------------------------------------
    # Geometric segmentation (SAM 1/2 task)
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

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.
            text: Optional text concept; takes precedence over geometry.
            points: ``(N, 2)`` point prompts as ``(x, y)`` coords.
            labels: Length-N labels (``1`` = fg, ``0`` = bg).
            box: ``[x1, y1, x2, y2]`` bounding box.
            mask_input: Low-res mask logits from a previous prediction.
            multimask_output: Return 3 masks for ambiguous prompts.
            return_logits: Keep raw logits instead of thresholding.

        Returns:
            A :class:`ConceptPrediction` containing the masks.
        """
        return self.concept_segmenter.segment(
            image,
            text=text,
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

    def set_image(self, image: str | Path | npt.NDArray[np.uint8]) -> tuple[int, int]:
        """Encode an image once so repeated prompts reuse the embedding.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.

        Returns:
            The ``(H, W)`` shape of the encoded image.
        """
        return self.concept_segmenter.set_image(image)

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
            points: ``(N, 2)`` point prompts as ``(x, y)`` coords.
            labels: Length-N labels (``1`` = fg, ``0`` = bg).
            box: ``[x1, y1, x2, y2]`` bounding box.
            mask_input: Low-res mask logits from a previous prediction.
            multimask_output: Return 3 masks for ambiguous prompts.
            return_logits: Keep raw logits instead of thresholding.

        Returns:
            A :class:`ConceptPrediction` containing the masks.
        """
        return self.concept_segmenter.predict(
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        result: ConceptPrediction,
        output_dir: str | Path,
        *,
        fmt: str = "png",
    ) -> Path:
        """Save a concept prediction to disk.

        Args:
            result: The prediction to save.
            output_dir: Target directory.
            fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.

        Returns:
            Path to the output directory.
        """
        return self.concept_segmenter.save(result, output_dir, fmt=fmt)

    # ------------------------------------------------------------------
    # Video tracking
    # ------------------------------------------------------------------

    def video(
        self,
        video: str | Path,
        *,
        frames_dir: str | Path | None = None,
        every_n: int = 1,
        max_frames: int | None = None,
        clean: bool = False,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> SAM3VideoSession:
        """Start a SAM 3 video tracking session.

        *video* can be a folder of JPEG/PNG frames or a video file (mp4, avi,
        mov, ...). When a video file is given, frames are extracted
        automatically.

        Args:
            video: Directory of frames or path to a video file.
            frames_dir: Where to extract frames (video file only).
            every_n: Keep every *n*-th frame (video file only).
            max_frames: Max frames to extract (video file only).
            clean: Delete pre-existing frames before extracting.
            offload_video_to_cpu: Keep frames on CPU to save GPU memory.
            offload_state_to_cpu: Keep state on CPU to save GPU memory.

        Returns:
            A :class:`SAM3VideoSession` ready for prompts and propagation.
        """
        return self.video_tracker.new_session(
            video,
            frames_dir=frames_dir,
            every_n=every_n,
            max_frames=max_frames,
            clean=clean,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
