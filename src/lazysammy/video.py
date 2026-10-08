"""High-level video tracking wrapper around SAM2VideoPredictor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from lazysammy.io import save_video_results
from lazysammy.types import (
    BoundingBox,
    FrameMasks,
    MaskInput,
    ModelSize,
    PointCoords,
    PointLabels,
    VideoResults,
)
from lazysammy.utils import (
    auto_detect_device,
    autocast,
    extract_frames,
    is_video_file,
    list_frame_files,
    load_image,
)
from lazysammy.validation import (
    normalize_box,
    validate_box,
    validate_frame_index,
    validate_mask_array,
    validate_points_and_labels,
    validate_positive_int,
)

logger = logging.getLogger(__name__)

# Threshold to binarise mask logits
_SCORE_THRESH = 0.0


class VideoTracker:
    """Easy-to-use video object tracking & segmentation powered by SAM2.

    Wraps :class:`SAM2VideoPredictor` with a friendly, stateful API.  Create a
    session from a folder of frames, add prompts on arbitrary frames for
    multiple objects, then propagate segmentations through the video.

    Example::

        tracker = VideoTracker("large")
        session = tracker.new_session("frames/")
        session.add_points(frame_idx=0, obj_id=1, points=[[150, 300]], labels=[1])
        session.add_box(frame_idx=10, obj_id=2, box=[50, 50, 200, 200])
        results = session.propagate()
        results.save("output/", fmt="png")
    """

    def __init__(
        self,
        model_size: str | ModelSize = "large",
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        vos_optimized: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialise the video tracker.

        Args:
            model_size: Model size or HuggingFace ID.
            checkpoint: Optional local checkpoint path.
            device: Device override.
            vos_optimized: Enable ``torch.compile`` for max FPS.
            **kwargs: Forwarded to the video predictor constructor.
        """
        from lazysammy.models import load_video_predictor

        self._predictor = load_video_predictor(
            model_size,
            checkpoint=checkpoint,
            device=device,
            vos_optimized=vos_optimized,
            **kwargs,
        )
        self._device = torch.device(auto_detect_device(device))

    def new_session(
        self,
        video: str | Path,
        *,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        frames_dir: str | Path | None = None,
        every_n: int = 1,
        max_frames: int | None = None,
        clean: bool = False,
    ) -> VideoSession:
        """Start a new tracking session on a video.

        *video* can be either a **directory of frames** (JPEG/PNG) or a
        **video file** (mp4, avi, mov, ...).  When a video file is given the
        frames are automatically extracted to *frames_dir* (or a sibling
        directory if ``None``).

        Args:
            video: Directory of frames **or** path to a video file.
            offload_video_to_cpu: Save GPU memory by keeping frames on CPU.
            offload_state_to_cpu: Save GPU memory by keeping state on CPU.
            frames_dir: Where to extract frames when *video* is a file.
            every_n: Keep every *n*-th frame (video file only).
            max_frames: Max frames to extract (video file only).
            clean: Delete pre-existing frames before extracting (video file
                only). Use this when re-extracting with different ``every_n``
                or ``max_frames`` so the frame set does not mix two runs.

        Returns:
            A :class:`VideoSession` that you can add prompts to and propagate.
        """
        validate_positive_int("every_n", every_n)
        validate_positive_int("max_frames", max_frames)

        video = Path(video)
        if is_video_file(video):
            video_dir = extract_frames(
                video,
                frames_dir,
                every_n=every_n,
                max_frames=max_frames,
                clean=clean,
            )
        else:
            video_dir = video
        frame_files = list_frame_files(video_dir)
        logger.info("Initialising session with %d frames from %s", len(frame_files), video_dir)

        with torch.inference_mode(), autocast(self._device):
            state = self._predictor.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            )
        return VideoSession(
            predictor=self._predictor,
            inference_state=state,
            video_dir=video_dir,
            num_frames=len(frame_files),
            frame_files=frame_files,
            device=self._device,
        )

    @property
    def device(self) -> torch.device:
        """Device used by this tracker."""
        return self._device

    @property
    def predictor(self) -> Any:
        """Access the underlying ``SAM2VideoPredictor``."""
        return self._predictor


class BaseVideoSession:
    """Shared session surface for SAM 2 and SAM 3 video tracking.

    Owns the frame bookkeeping, results access, and the save/overlay helpers so
    the two backends do not duplicate them. Subclasses supply the prompt adders
    and the propagation call for their predictor.
    """

    def __init__(
        self,
        *,
        video_dir: Path,
        num_frames: int,
        frame_files: list[Path],
        device: torch.device,
    ) -> None:
        self._video_dir = video_dir
        self._num_frames = num_frames
        self._frame_files = frame_files
        self._device = device
        self._results: VideoResults | None = None

    # ------------------------------------------------------------------
    # Results access
    # ------------------------------------------------------------------

    @property
    def results(self) -> VideoResults | None:
        """The latest propagation results, or ``None`` if not yet propagated."""
        return self._results

    @property
    def num_frames(self) -> int:
        """Total number of frames in the video."""
        return self._num_frames

    @property
    def device(self) -> torch.device:
        """Device used by this session."""
        return self._device

    @property
    def video_dir(self) -> Path:
        """Path to the frame directory."""
        return self._video_dir

    def get_frame(self, frame_idx: int) -> npt.NDArray[np.uint8]:
        """Load a frame as an RGB numpy array.

        Args:
            frame_idx: Zero-based frame index.

        Returns:
            ``(H, W, 3)`` RGB uint8 array.
        """
        validate_frame_index(frame_idx, self._num_frames)
        return load_image(self._frame_files[frame_idx])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        output_dir: str | Path,
        *,
        fmt: str = "png",
        results: VideoResults | None = None,
    ) -> Path:
        """Save tracking results to disk.

        Args:
            output_dir: Output directory.
            fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.
            results: Explicit results to save; defaults to the last propagation.

        Returns:
            Path to the output directory.

        Raises:
            RuntimeError: If no results are available.
        """
        res = results or self._results
        if res is None:
            msg = "No results to save. Call propagate() first."
            raise RuntimeError(msg)
        return save_video_results(res, output_dir, fmt=fmt)

    def save_overlay(
        self,
        output_path: str | Path,
        *,
        results: VideoResults | None = None,
        fps: float = 24.0,
        alpha: float = 0.5,
        draw_contours: bool = True,
        show_ids: bool = True,
        show_frame_number: bool = True,
        codec: str = "mp4v",
        as_frames: bool = False,
    ) -> Path:
        """Save an overlay video (MP4) or overlay frame PNGs.

        Args:
            output_path: Output file path (``.mp4``) or directory (when
                ``as_frames=True``).
            results: Explicit results; defaults to the last propagation.
            fps: Video frame rate (ignored when ``as_frames=True``).
            alpha: Mask overlay transparency.
            draw_contours: Draw mask contour lines.
            show_ids: Annotate object IDs on the overlay.
            show_frame_number: Burn frame index (video only).
            codec: FourCC codec for MP4 writing.
            as_frames: Save individual PNG frames instead of an MP4.

        Returns:
            Path to the output file or directory.

        Raises:
            RuntimeError: If no results are available.
        """
        from lazysammy.visualization import save_video_overlay, save_video_overlay_mp4

        res = results or self._results
        if res is None:
            msg = "No results to save. Call propagate() first."
            raise RuntimeError(msg)

        if as_frames:
            return save_video_overlay(
                self._video_dir,
                res,
                output_path,
                alpha=alpha,
                draw_contours=draw_contours,
            )
        return save_video_overlay_mp4(
            self._video_dir,
            res,
            output_path,
            fps=fps,
            alpha=alpha,
            draw_contours=draw_contours,
            show_ids=show_ids,
            show_frame_number=show_frame_number,
            codec=codec,
        )


class VideoSession(BaseVideoSession):
    """A tracking session on one video.

    This object is returned by :meth:`VideoTracker.new_session` and provides
    methods to add prompts, propagate, reset, and save results.
    """

    def __init__(
        self,
        predictor: Any,
        inference_state: Any,
        video_dir: Path,
        num_frames: int,
        frame_files: list[Path],
        device: torch.device,
    ) -> None:
        super().__init__(
            video_dir=video_dir,
            num_frames=num_frames,
            frame_files=frame_files,
            device=device,
        )
        self._predictor = predictor
        self._state = inference_state

    # ------------------------------------------------------------------
    # Prompt adders
    # ------------------------------------------------------------------

    def add_points(
        self,
        frame_idx: int,
        obj_id: int,
        points: PointCoords,
        labels: PointLabels,
        *,
        clear_old: bool = True,
    ) -> FrameMasks:
        """Add point prompts for an object on a frame.

        Args:
            frame_idx: Zero-based frame index.
            obj_id: Unique object identifier.
            points: ``(N, 2)`` array of ``(x, y)`` coordinates.
            labels: Length-N array (``1`` = foreground, ``0`` = background).
            clear_old: Whether to clear previous points on this frame/object.

        Returns:
            Predicted :class:`FrameMasks` for the prompted frame.
        """
        validate_frame_index(frame_idx, self._num_frames)
        validate_points_and_labels(points, labels)
        pts = np.array(points, dtype=np.float32)
        lbs = np.array(labels, dtype=np.int32)

        with torch.inference_mode(), autocast(self._device):
            fidx, obj_ids, masks = self._predictor.add_new_points_or_box(
                inference_state=self._state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=pts,
                labels=lbs,
                clear_old_points=clear_old,
            )
        return self._to_frame_masks(fidx, obj_ids, masks)

    def add_box(
        self,
        frame_idx: int,
        obj_id: int,
        box: BoundingBox,
    ) -> FrameMasks:
        """Add a bounding box prompt for an object on a frame.

        Args:
            frame_idx: Zero-based frame index.
            obj_id: Unique object identifier.
            box: ``[x1, y1, x2, y2]`` bounding box.

        Returns:
            Predicted :class:`FrameMasks` for the prompted frame.
        """
        validate_frame_index(frame_idx, self._num_frames)
        validate_box(box)
        bx = normalize_box(box)

        with torch.inference_mode(), autocast(self._device):
            fidx, obj_ids, masks = self._predictor.add_new_points_or_box(
                inference_state=self._state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=bx,
            )
        return self._to_frame_masks(fidx, obj_ids, masks)

    def add_mask(
        self,
        frame_idx: int,
        obj_id: int,
        mask: MaskInput,
    ) -> FrameMasks:
        """Add a mask prompt for an object on a frame.

        Args:
            frame_idx: Zero-based frame index.
            obj_id: Unique object identifier.
            mask: ``(H, W)`` binary mask (``1`` = foreground).

        Returns:
            Predicted :class:`FrameMasks` for the prompted frame.
        """
        validate_frame_index(frame_idx, self._num_frames)
        validate_mask_array(mask, allow_3d=False)
        with torch.inference_mode(), autocast(self._device):
            fidx, obj_ids, masks = self._predictor.add_new_mask(
                inference_state=self._state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask,
            )
        return self._to_frame_masks(fidx, obj_ids, masks)

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(
        self,
        *,
        start_frame: int | None = None,
        max_frames: int | None = None,
        reverse: bool = False,
    ) -> VideoResults:
        """Propagate prompts through the video.

        Args:
            start_frame: Frame to start propagation from (``None`` = auto).
            max_frames: Maximum number of frames to track.
            reverse: Track backwards in time.

        Returns:
            :class:`VideoResults` containing masks for every tracked frame.
        """
        if start_frame is not None:
            validate_frame_index(start_frame, self._num_frames)
        validate_positive_int("max_frames", max_frames)

        results = VideoResults(video_dir=self._video_dir, num_frames=self._num_frames)

        with torch.inference_mode(), autocast(self._device):
            for fidx, obj_ids, masks in self._predictor.propagate_in_video(
                inference_state=self._state,
                start_frame_idx=start_frame,
                max_frame_num_to_track=max_frames,
                reverse=reverse,
            ):
                fm = self._to_frame_masks(fidx, obj_ids, masks)
                results.frames.append(fm)

        self._results = results
        logger.info(
            "Propagation complete: %d frames, %d objects",
            len(results),
            len(results.object_ids),
        )
        return results

    def propagate_bidirectional(
        self,
        *,
        start_frame: int | None = None,
        max_frames: int | None = None,
    ) -> VideoResults:
        """Propagate in both directions from the start frame.

        This is useful when prompts exist on a mid-sequence frame and you
        want tracking both forward and backward.

        Args:
            start_frame: Frame to start from (``None`` = auto).
            max_frames: Maximum frames to track in each direction.

        Returns:
            :class:`VideoResults` merging forward and backward passes.
        """
        forward = self.propagate(start_frame=start_frame, max_frames=max_frames, reverse=False)
        backward = self.propagate(start_frame=start_frame, max_frames=max_frames, reverse=True)
        # Merge: forward takes priority on overlapping frames
        merged = VideoResults(video_dir=self._video_dir, num_frames=self._num_frames)
        bwd_index = {fm.frame_idx: fm for fm in backward.frames}
        fwd_index = {fm.frame_idx: fm for fm in forward.frames}
        combined = {**bwd_index, **fwd_index}  # forward takes priority
        merged.frames = [combined[idx] for idx in sorted(combined)]
        self._results = merged
        return merged

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def remove_object(self, obj_id: int) -> None:
        """Remove an object and its prompts from the session.

        Args:
            obj_id: The object to remove.
        """
        with torch.inference_mode(), autocast(self._device):
            self._predictor.remove_object(self._state, obj_id)
        logger.info("Removed object %d from session", obj_id)

    def clear_frame_prompts(self, frame_idx: int, obj_id: int) -> None:
        """Clear all prompts for an object on a specific frame.

        Args:
            frame_idx: The frame to clear.
            obj_id: The object whose prompts to clear.
        """
        validate_frame_index(frame_idx, self._num_frames)
        with torch.inference_mode(), autocast(self._device):
            self._predictor.clear_all_prompts_in_frame(self._state, frame_idx, obj_id)

    def reset(self) -> None:
        """Reset all prompts and tracking state for this session."""
        with torch.inference_mode(), autocast(self._device):
            self._predictor.reset_state(self._state)
        self._results = None
        logger.info("Session reset")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _to_frame_masks(
        self,
        frame_idx: int,
        obj_ids: Any,
        masks_tensor: torch.Tensor,
    ) -> FrameMasks:
        """Convert raw predictor output to :class:`FrameMasks`."""
        obj_id_list = [int(o) for o in obj_ids]
        masks_binary = (masks_tensor > _SCORE_THRESH).squeeze(1).cpu().numpy().astype(bool)
        masks_dict = {oid: masks_binary[i] for i, oid in enumerate(obj_id_list)}
        return FrameMasks(
            frame_idx=frame_idx,
            masks=masks_dict,
        )
