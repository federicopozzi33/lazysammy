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
import torch

from lazysammy.auto_mask import AutoSegmenter
from lazysammy.image import ImageSegmenter
from lazysammy.types import (
    AutoMaskResult,
    BoundingBox,
    ImagePrediction,
    MaskLogits,
    ModelSize,
    PointCoords,
    PointLabels,
)
from lazysammy.utils import auto_detect_device
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
    # Configuration accessors
    # ------------------------------------------------------------------

    @property
    def model_size(self) -> str | ModelSize:
        """The configured model size (alias or HuggingFace id)."""
        return self._model_size

    @property
    def device(self) -> str | None:
        """The configured device override, or ``None`` for auto-detection."""
        return self._device

    @property
    def checkpoint(self) -> str | Path | None:
        """The configured local checkpoint path, if any."""
        return self._checkpoint

    @property
    def vos_optimized(self) -> bool:
        """Whether ``torch.compile`` video optimization is enabled."""
        return self._vos_optimized

    @property
    def resolved_device(self) -> torch.device:
        """The device that will actually be used (after auto-detection).

        Note: resolving inspects the runtime and may load no model weights,
        but it does trigger lazy component creation only when a component
        already exists.
        """
        if self._image_segmenter is not None:
            return self._image_segmenter.device
        return torch.device(auto_detect_device(self._device))

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
        mask_input: MaskLogits | None = None,
        multimask_output: bool = True,
    ) -> ImagePrediction:
        """Segment an image with point, box, or mask prompts.

        This is a shortcut to :meth:`ImageSegmenter.segment`.

        Args:
            image: File path or ``(H, W, 3)`` RGB uint8 array.
            points: ``(N, 2)`` point prompts as ``(x, y)`` coords.
            labels: Length-N labels (``1`` = fg, ``0`` = bg).
            box: ``[x1, y1, x2, y2]`` bounding box.
            mask_input: Low-res mask **logits** from a previous prediction.
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
            points_batch: Per-image ``(N, 2)`` point arrays (``None`` to skip).
            labels_batch: Per-image point labels.
            box_batch: Per-image ``[x1, y1, x2, y2]`` boxes.
            multimask_output: Whether to return multiple masks per image.

        Returns:
            One :class:`ImagePrediction` per input image.
        """
        return self.image_segmenter.segment_batch(
            images,
            points_batch=points_batch,
            labels_batch=labels_batch,
            box_batch=box_batch,
            multimask_output=multimask_output,
        )

    def save(
        self,
        result: ImagePrediction | AutoMaskResult,
        output_dir: str | Path,
        *,
        fmt: str = "png",
    ) -> Path:
        """Save an image or auto-mask result to disk.

        Args:
            result: The prediction to save.
            output_dir: Target directory.
            fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.

        Returns:
            Path to the output directory.

        Raises:
            TypeError: If *result* is not a supported result type.
        """
        from lazysammy.io import save_auto_mask_result, save_image_prediction

        if isinstance(result, ImagePrediction):
            return save_image_prediction(result, output_dir, fmt=fmt)
        if isinstance(result, AutoMaskResult):
            return save_auto_mask_result(result, output_dir, fmt=fmt)
        msg = (
            "save() expects an ImagePrediction or AutoMaskResult; "
            f"got {type(result).__name__}."
        )
        raise TypeError(msg)

    # ------------------------------------------------------------------
    # Automatic mask generation shortcut
    # ------------------------------------------------------------------

    def auto_segment(
        self,
        image: str | Path | npt.NDArray[np.uint8],
        *,
        points_per_side: int | None = None,
        pred_iou_thresh: float | None = None,
        stability_score_thresh: float | None = None,
        min_mask_region_area: int | None = None,
        use_m2m: bool | None = None,
    ) -> AutoMaskResult:
        """Automatically segment every object in an image (no prompts).

        When tuning options are provided a dedicated, short-lived
        :class:`AutoSegmenter` is built with those settings; otherwise the
        shared lazy instance is reused.

        Args:
            image: File path or ``(H, W, 3)`` RGB array.
            points_per_side: Grid density for point sampling.
            pred_iou_thresh: Keep masks with predicted IoU above this.
            stability_score_thresh: Keep masks with stability above this.
            min_mask_region_area: Drop regions smaller than this (pixels).
            use_m2m: Enable the mask-to-mask refinement step.

        Returns:
            :class:`AutoMaskResult` with all generated masks.
        """
        if (
            points_per_side is None
            and pred_iou_thresh is None
            and stability_score_thresh is None
            and min_mask_region_area is None
            and use_m2m is None
        ):
            return self.auto_segmenter.generate(image)

        defaults = {
            "points_per_side": 32,
            "pred_iou_thresh": 0.8,
            "stability_score_thresh": 0.95,
            "min_mask_region_area": 0,
            "use_m2m": False,
        }
        overrides: dict[str, Any] = dict(defaults)
        overrides.update(
            {
                k: v
                for k, v in {
                    "points_per_side": points_per_side,
                    "pred_iou_thresh": pred_iou_thresh,
                    "stability_score_thresh": stability_score_thresh,
                    "min_mask_region_area": min_mask_region_area,
                    "use_m2m": use_m2m,
                }.items()
                if v is not None
            }
        )
        segmenter = AutoSegmenter(
            self._model_size,
            checkpoint=self._checkpoint,
            device=self._device,
            points_per_side=int(overrides["points_per_side"]),
            pred_iou_thresh=float(overrides["pred_iou_thresh"]),
            stability_score_thresh=float(overrides["stability_score_thresh"]),
            min_mask_region_area=int(overrides["min_mask_region_area"]),
            use_m2m=bool(overrides["use_m2m"]),
        )
        return segmenter.generate(image)

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
