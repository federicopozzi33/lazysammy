"""SAM 3 video concept tracking.

SAM 3 tracks objects through video using the same stateful session model as
SAM 2, but adds *text* prompts: describe a concept (e.g. ``"person"``) and SAM 3
detects and tracks every matching instance, assigning each a unique object id.

The session API mirrors :class:`~lazysammy.video.VideoSession` so the two are
interchangeable for point/box tracking, while :meth:`SAM3VideoSession.add_text`
exposes the new concept-tracking feature.
"""

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
    PointCoords,
    PointLabels,
    VideoResults,
)
from lazysammy.utils import (
    auto_detect_device,
    extract_frames,
    is_video_file,
    list_frame_files,
    load_image,
)
from lazysammy.validation import (
    normalize_box,
    validate_box,
    validate_frame_index,
    validate_points_and_labels,
    validate_positive_int,
)

logger = logging.getLogger(__name__)


class SAM3VideoTracker:
    """Video object tracking & concept segmentation powered by SAM 3.

    Example::

        tracker = SAM3VideoTracker()
        session = tracker.new_session("clip.mp4")
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
        **kwargs: Any,
    ) -> None:
        """Initialise the SAM 3 video tracker.

        Args:
            checkpoint: Optional local checkpoint path.
            device: Device override (auto-detected if ``None``).
            version: ``"sam3"`` or ``"sam3.1"`` (Object Multiplex).
            compile: Enable ``torch.compile`` (SAM 3.1 only).
            **kwargs: Extra arguments forwarded to the predictor builder.
        """
        from lazysammy.models import load_sam3_video_predictor

        self._device = torch.device(auto_detect_device(device))
        self._predictor = load_sam3_video_predictor(
            checkpoint=checkpoint,
            device=device,
            version=version,
            compile=compile,
            **kwargs,
        )

    def new_session(
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
        """Start a new tracking session on a video.

        *video* can be a directory of frames or a video file (mp4, avi, ...).
        Video files are extracted to *frames_dir* first.

        Args:
            video: Directory of frames or path to a video file.
            frames_dir: Where to extract frames when *video* is a file.
            every_n: Keep every *n*-th frame (video file only).
            max_frames: Max frames to extract (video file only).
            clean: Delete pre-existing frames before extracting.
            offload_video_to_cpu: Keep frames on CPU to save GPU memory.
            offload_state_to_cpu: Keep state on CPU to save GPU memory.

        Returns:
            A :class:`SAM3VideoSession` ready for prompts and propagation.
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
        logger.info("Starting SAM3 session with %d frames from %s", len(frame_files), video_dir)

        response = self._predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(video_dir),
                "offload_video_to_cpu": offload_video_to_cpu,
                "offload_state_to_cpu": offload_state_to_cpu,
            }
        )
        return SAM3VideoSession(
            predictor=self._predictor,
            session_id=response["session_id"],
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
        """Access the underlying SAM 3 video predictor."""
        return self._predictor


class SAM3VideoSession:
    """A SAM 3 tracking session on one video.

    Returned by :meth:`SAM3VideoTracker.new_session`. Add text, point, or box
    prompts, then :meth:`propagate` to track them through the video.
    """

    def __init__(
        self,
        predictor: Any,
        session_id: str,
        video_dir: Path,
        num_frames: int,
        frame_files: list[Path],
        device: torch.device,
    ) -> None:
        self._predictor = predictor
        self._session_id = session_id
        self._video_dir = video_dir
        self._num_frames = num_frames
        self._frame_files = frame_files
        self._device = device
        self._results: VideoResults | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Prompt adders
    # ------------------------------------------------------------------

    def add_text(self, frame_idx: int, text: str) -> FrameMasks:
        """Add a text concept prompt and detect its instances on one frame.

        SAM 3 assigns a unique object id to every detected instance, so a
        single text prompt can seed many tracked objects.

        Args:
            frame_idx: Frame to run detection on.
            text: A short noun phrase, e.g. ``"person"``.

        Returns:
            :class:`FrameMasks` for the prompted frame.
        """
        validate_frame_index(frame_idx, self._num_frames)
        response = self._predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self._session_id,
                "frame_index": frame_idx,
                "text": text,
            }
        )
        return self._to_frame_masks(frame_idx, response["outputs"])

    def add_points(
        self,
        frame_idx: int,
        obj_id: int,
        points: PointCoords,
        labels: PointLabels,
        *,
        clear_old: bool = True,
    ) -> FrameMasks:
        """Add point prompts for one object on a frame.

        Args:
            frame_idx: Frame to add the points on.
            obj_id: Unique object identifier.
            points: ``(N, 2)`` array of ``(x, y)`` coordinates in pixels.
            labels: Length-N array (``1`` = foreground, ``0`` = background).
            clear_old: Whether to clear previous points on this frame/object.

        Returns:
            :class:`FrameMasks` for the prompted frame.
        """
        validate_frame_index(frame_idx, self._num_frames)
        validate_points_and_labels(points, labels)
        rel_points = self._to_relative_points(points)
        response = self._predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self._session_id,
                "frame_index": frame_idx,
                "points": rel_points,
                "point_labels": np.asarray(labels, dtype=np.int32),
                "obj_id": obj_id,
                "clear_old_points": clear_old,
            }
        )
        return self._to_frame_masks(frame_idx, response["outputs"])

    def add_box(self, frame_idx: int, obj_id: int, box: BoundingBox) -> FrameMasks:
        """Add a bounding-box prompt for one object on a frame.

        Args:
            frame_idx: Frame to add the box on.
            obj_id: Unique object identifier.
            box: ``[x1, y1, x2, y2]`` box in pixels.

        Returns:
            :class:`FrameMasks` for the prompted frame.
        """
        validate_frame_index(frame_idx, self._num_frames)
        validate_box(box)
        rel_xywh = self._to_relative_xywh(box)
        response = self._predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self._session_id,
                "frame_index": frame_idx,
                "bounding_boxes": rel_xywh,
                "bounding_box_labels": np.array([1], dtype=np.int32),
                "obj_id": obj_id,
            }
        )
        return self._to_frame_masks(frame_idx, response["outputs"])

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(
        self,
        *,
        direction: str = "both",
        start_frame: int | None = None,
        max_frames: int | None = None,
    ) -> VideoResults:
        """Propagate all prompts through the video.

        Args:
            direction: ``"forward"``, ``"backward"``, or ``"both"``.
            start_frame: Frame to start from (``None`` = earliest prompt).
            max_frames: Maximum frames to track in each direction.

        Returns:
            :class:`VideoResults` with masks for every tracked frame.

        Raises:
            ValueError: If *direction* is not one of the accepted values.
        """
        if direction not in {"forward", "backward", "both"}:
            msg = f"direction must be 'forward', 'backward', or 'both'; got {direction!r}."
            raise ValueError(msg)
        if start_frame is not None:
            validate_frame_index(start_frame, self._num_frames)
        validate_positive_int("max_frames", max_frames)

        results = VideoResults(video_dir=self._video_dir, num_frames=self._num_frames)
        request: dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": self._session_id,
            "propagation_direction": direction,
        }
        if start_frame is not None:
            request["start_frame_index"] = start_frame
        if max_frames is not None:
            request["max_frame_num_to_track"] = max_frames

        for response in self._predictor.handle_stream_request(request):
            frame_idx = int(response["frame_index"])
            results.frames.append(self._to_frame_masks(frame_idx, response["outputs"]))

        results.frames.sort(key=lambda fm: fm.frame_idx)
        self._results = results
        logger.info(
            "SAM3 propagation complete: %d frames, %d objects",
            len(results),
            len(results.object_ids),
        )
        return results

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def remove_object(self, obj_id: int) -> None:
        """Remove an object and its prompts from the session.

        Args:
            obj_id: The object to remove.
        """
        self._predictor.handle_request(
            {
                "type": "remove_object",
                "session_id": self._session_id,
                "obj_id": obj_id,
            }
        )
        logger.info("Removed object %d from SAM3 session", obj_id)

    def reset(self) -> None:
        """Reset all prompts and tracking state for this session."""
        self._predictor.handle_request({"type": "reset_session", "session_id": self._session_id})
        self._results = None
        logger.info("SAM3 session reset")

    def close(self) -> None:
        """Close the session and free its GPU resources."""
        if not self._closed:
            self._predictor.handle_request(
                {"type": "close_session", "session_id": self._session_id}
            )
            self._closed = True

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
            results: Explicit results; defaults to the last propagation.

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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _frame_shape(self) -> tuple[int, int]:
        """Return the ``(H, W)`` of the first frame."""
        return load_image(self._frame_files[0]).shape[:2]

    def _to_relative_points(self, points: PointCoords) -> npt.NDArray[np.float32]:
        """Convert absolute ``(x, y)`` points to normalized coordinates."""
        height, width = self._frame_shape()
        arr = np.asarray(points, dtype=np.float32)
        scale = np.array([width, height], dtype=np.float32)
        return arr / scale

    def _to_relative_xywh(self, box: BoundingBox) -> npt.NDArray[np.float32]:
        """Convert an absolute ``[x1, y1, x2, y2]`` box to normalized xywh."""
        height, width = self._frame_shape()
        x1, y1, x2, y2 = (float(v) for v in normalize_box(box))
        return np.array(
            [[x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]],
            dtype=np.float32,
        )

    @staticmethod
    def _to_frame_masks(frame_idx: int, outputs: dict[str, Any]) -> FrameMasks:
        """Convert a SAM 3 per-frame output dict into :class:`FrameMasks`."""
        obj_ids = outputs.get("out_obj_ids")
        masks = outputs.get("out_binary_masks")
        if obj_ids is None or masks is None:
            return FrameMasks(frame_idx=frame_idx, masks={})

        obj_id_list = [int(o) for o in np.asarray(obj_ids).reshape(-1)]
        masks_np = np.asarray(masks)
        if masks_np.ndim == 4:  # (N, 1, H, W)
            masks_np = masks_np[:, 0]
        masks_dict = {oid: masks_np[i].astype(bool) for i, oid in enumerate(obj_id_list)}
        return FrameMasks(frame_idx=frame_idx, masks=masks_dict)
