"""Unified SAM2 facade - the main entry point for ``lazysammy``.

Provides a single :class:`SAM2` object that lazily initialises image, video,
and auto-mask predictors as needed, sharing the same model configuration.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from lazysammy.auto_mask import AutoSegmenter
from lazysammy.image import ImageSegmenter
from lazysammy.types import (
    AutoMaskResult,
    BoundingBox,
    ImagePrediction,
    MaskInput,
    ModelSize,
    PointCoords,
    PointLabels,
)
from lazysammy.video import VideoSession, VideoTracker

logger = logging.getLogger(__name__)


class SAM2:
    """Unified, easy-to-use interface to all SAM2 capabilities.

    A single :class:`SAM2` object lazily creates the underlying image, video,
    and automatic-mask predictors on first use so you only pay for what you
    need.

    Example::

        sam = SAM2("large")

        # Image segmentation
        pred = sam.segment("photo.jpg", points=[[100, 200]], labels=[1])

        # Auto-segment everything
        auto = sam.auto_segment("photo.jpg")

        # Video tracking
        session = sam.video("frames/")
        session.add_points(frame_idx=0, obj_id=1, points=[[150, 300]], labels=[1])
        results = session.propagate()
        session.save("output/")
    """

    def __init__(
        self,
        model_size: str | ModelSize = "large",
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        vos_optimized: bool = False,
    ) -> None:
        """Create a new SAM2 wrapper.

        Args:
            model_size: Model size alias (``"tiny"``, ``"small"``, ``"base_plus"``,
                ``"large"``) or a HuggingFace model ID.
            checkpoint: Optional path to a local ``.pt`` checkpoint.
            device: Device string override (auto-detected if ``None``).
            vos_optimized: Use ``torch.compile`` on all model components for
                faster video inference (requires CUDA and PyTorch >= 2.5.1).
        """
        self._model_size = model_size
        self._checkpoint = checkpoint
        self._device = device
        self._vos_optimized = vos_optimized

        # Lazy-initialised sub-components
        self._image_segmenter: ImageSegmenter | None = None
        self._video_tracker: VideoTracker | None = None
        self._auto_segmenter: AutoSegmenter | None = None

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    @property
    def image_segmenter(self) -> ImageSegmenter:
        """Lazily-loaded :class:`ImageSegmenter`."""
        if self._image_segmenter is None:
            self._image_segmenter = ImageSegmenter(
                self._model_size,
                checkpoint=self._checkpoint,
                device=self._device,
            )
        return self._image_segmenter

    @property
    def video_tracker(self) -> VideoTracker:
        """Lazily-loaded :class:`VideoTracker`."""
        if self._video_tracker is None:
            self._video_tracker = VideoTracker(
                self._model_size,
                checkpoint=self._checkpoint,
                device=self._device,
                vos_optimized=self._vos_optimized,
            )
        return self._video_tracker

    @property
    def auto_segmenter(self) -> AutoSegmenter:
        """Lazily-loaded :class:`AutoSegmenter`."""
        if self._auto_segmenter is None:
            self._auto_segmenter = AutoSegmenter(
                self._model_size,
                checkpoint=self._checkpoint,
                device=self._device,
            )
        return self._auto_segmenter

    # ------------------------------------------------------------------
    # Image segmentation shortcuts
    # ------------------------------------------------------------------

    def segment(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        *,
        points: PointCoords | None = None,
        labels: PointLabels | None = None,
        box: BoundingBox | None = None,
        mask_input: MaskInput | None = None,
        multimask_output: bool = True,
    ) -> ImagePrediction:
        """Segment an image with point, box, or mask prompts.

        This is a shortcut to :meth:`ImageSegmenter.segment`.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.
            points: ``(N, 2)`` point prompts as ``(x, y)`` coords.
            labels: Length-N labels (``1`` = fg, ``0`` = bg).
            box: ``[x1, y1, x2, y2]`` bounding box.
            mask_input: Low-res mask from a previous prediction.
            multimask_output: Return 3 masks for ambiguous prompts.

        Returns:
            :class:`ImagePrediction` containing the masks.
        """
        return self.image_segmenter.segment(
            image,
            points=points,
            labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )

    def segment_point(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        x: float,
        y: float,
        *,
        foreground: bool = True,
    ) -> ImagePrediction:
        """Segment with a single point click.

        Args:
            image: Image to segment.
            x: X coordinate.
            y: Y coordinate.
            foreground: ``True`` for positive, ``False`` for negative.

        Returns:
            :class:`ImagePrediction`.
        """
        return self.image_segmenter.segment_point(image, x, y, foreground=foreground)

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
            x1, y1, x2, y2: Box coordinates.

        Returns:
            :class:`ImagePrediction`.
        """
        return self.image_segmenter.segment_box(image, x1, y1, x2, y2)

    def segment_multi_box(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        boxes: Sequence[Sequence[float]],
    ) -> list[ImagePrediction]:
        """Segment multiple objects, each with a bounding box.

        Args:
            image: Image to segment.
            boxes: List of ``[x1, y1, x2, y2]`` boxes.

        Returns:
            Per-box :class:`ImagePrediction` list.
        """
        return self.image_segmenter.segment_multi_box(image, boxes)

    def refine(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        previous_logits: npt.NDArray[np.floating[Any]],
        *,
        points: PointCoords | None = None,
        labels: PointLabels | None = None,
    ) -> ImagePrediction:
        """Refine a prior prediction with additional prompts.

        Args:
            image: Same image as the previous prediction.
            previous_logits: Low-res logits from :attr:`Mask.logits`.
            points: Additional point prompts for refinement.
            labels: Labels for the additional points.

        Returns:
            Refined :class:`ImagePrediction`.
        """
        return self.image_segmenter.refine(
            image, previous_logits, points=points, labels=labels
        )

    # ------------------------------------------------------------------
    # Automatic mask generation shortcut
    # ------------------------------------------------------------------

    def auto_segment(
        self,
        image: str | Path | npt.NDArray[np.uint8],
    ) -> AutoMaskResult:
        """Automatically segment every object in an image (no prompts).

        Args:
            image: File path or ``(H, W, 3)`` RGB array.

        Returns:
            :class:`AutoMaskResult` with all generated masks.
        """
        return self.auto_segmenter.generate(image)

    # ------------------------------------------------------------------
    # Video tracking shortcut
    # ------------------------------------------------------------------

    def video(
        self,
        video: str | Path,
        *,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        frames_dir: str | Path | None = None,
        every_n: int = 1,
        max_frames: int | None = None,
    ) -> VideoSession:
        """Start a video tracking session.

        *video* can be a **folder of JPEG/PNG frames** or a **video file**
        (mp4, avi, mov, …).  When a video file is given, frames are
        extracted automatically.

        Args:
            video: Directory of frames **or** path to a video file.
            offload_video_to_cpu: Reduce GPU memory by keeping frames on CPU.
            offload_state_to_cpu: Reduce GPU memory by keeping state on CPU.
            frames_dir: Where to extract frames (video file only).
            every_n: Keep every *n*-th frame (video file only).
            max_frames: Max frames to extract (video file only).

        Returns:
            A :class:`VideoSession` ready for prompt addition and propagation.
        """
        return self.video_tracker.new_session(
            video,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            frames_dir=frames_dir,
            every_n=every_n,
            max_frames=max_frames,
        )
