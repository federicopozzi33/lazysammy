"""Core type definitions for lazysammy."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from lazysammy.validation import validate_nonempty_masks


class ModelSize(str, Enum):
    """Available SAM2 model sizes."""

    TINY = "tiny"
    SMALL = "small"
    BASE_PLUS = "base_plus"
    LARGE = "large"


class PromptType(str, Enum):
    """Types of prompts that can be used for segmentation."""

    POINT = "point"
    BOX = "box"
    MASK = "mask"


# ---------------------------------------------------------------------------
# HuggingFace model ID mapping
# ---------------------------------------------------------------------------

HF_MODEL_IDS: dict[ModelSize, str] = {
    ModelSize.TINY: "facebook/sam2.1-hiera-tiny",
    ModelSize.SMALL: "facebook/sam2.1-hiera-small",
    ModelSize.BASE_PLUS: "facebook/sam2.1-hiera-base-plus",
    ModelSize.LARGE: "facebook/sam2.1-hiera-large",
}

# Checkpoint filenames per model size (for local loading)
CHECKPOINT_FILENAMES: dict[ModelSize, str] = {
    ModelSize.TINY: "sam2.1_hiera_tiny.pt",
    ModelSize.SMALL: "sam2.1_hiera_small.pt",
    ModelSize.BASE_PLUS: "sam2.1_hiera_base_plus.pt",
    ModelSize.LARGE: "sam2.1_hiera_large.pt",
}

# Config filenames per model size (for local loading via hydra)
CONFIG_FILENAMES: dict[ModelSize, str] = {
    ModelSize.TINY: "configs/sam2.1/sam2.1_hiera_t.yaml",
    ModelSize.SMALL: "configs/sam2.1/sam2.1_hiera_s.yaml",
    ModelSize.BASE_PLUS: "configs/sam2.1/sam2.1_hiera_b+.yaml",
    ModelSize.LARGE: "configs/sam2.1/sam2.1_hiera_l.yaml",
}


# ---- Convenience type aliases ----
PointCoords = Sequence[Sequence[float]] | npt.NDArray[np.floating[Any]]
PointLabels = Sequence[int] | npt.NDArray[np.integer[Any]]
BoundingBox = (
    Sequence[float]  # [x1, y1, x2, y2]
    | npt.NDArray[np.floating[Any]]
)
# Low-resolution mask logits (e.g. ``1 x 256 x 256``) as returned in
# :attr:`Mask.logits` and accepted by ``mask_input`` for iterative refinement.
MaskLogits = npt.NDArray[np.floating[Any]]
# Binary ``(H, W)`` mask used as a prompt (``True``/``1`` = foreground).
MaskInput = npt.NDArray[np.bool_] | npt.NDArray[np.uint8]
ImageInput = npt.NDArray[np.uint8]  # H x W x 3 RGB array


@dataclass
class Mask:
    """A single segmentation mask with metadata.

    Attributes:
        data: Binary mask array of shape ``(H, W)`` with dtype ``bool``.
        score: Model-predicted IoU quality score in ``[0, 1]``.
        logits: Low-resolution logit mask (useful for iterative refinement).
        area: Pixel area of the mask.
    """

    data: npt.NDArray[np.bool_]
    score: float
    logits: npt.NDArray[np.floating[Any]] | None = None
    area: int | None = None

    def __post_init__(self) -> None:
        if self.area is None:
            self.area = int(self.data.sum())

    def numpy(self) -> npt.NDArray[np.bool_]:
        """Return the mask as a boolean numpy array."""
        return self.data

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """Backward-compatible alias for :attr:`data`."""
        return self.data

    def as_uint8(self) -> npt.NDArray[np.uint8]:
        """Return the mask as a uint8 array (0 or 255)."""
        return self.data.astype(np.uint8) * 255

    def save(self, path: str | Path) -> None:
        """Save the mask as a PNG file.

        Args:
            path: Destination ``.png`` path. Parent directories are created.
        """
        import cv2

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), self.as_uint8())


@dataclass
class ImagePrediction:
    """Result of segmenting a single image.

    Attributes:
        masks: List of predicted :class:`Mask` objects.
        image_shape: ``(H, W)`` of the original image (optional).
    """

    masks: list[Mask]
    image_shape: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.masks)

    # -- convenience helpers ------------------------------------------------

    @property
    def best_mask(self) -> Mask:
        """Return the mask with the highest IoU score."""
        validate_nonempty_masks(len(self.masks), context="ImagePrediction")
        return max(self.masks, key=lambda m: m.score)

    def numpy(self) -> npt.NDArray[np.bool_]:
        """Stack all masks into a ``(N, H, W)`` boolean array."""
        if not self.masks:
            shape = self.image_shape or (0, 0)
            return np.zeros((0, *shape), dtype=bool)
        return np.stack([m.data for m in self.masks])

    def as_uint8(self) -> npt.NDArray[np.uint8]:
        """Stack all masks into a ``(N, H, W)`` uint8 array (0/255)."""
        return (self.numpy().astype(np.uint8)) * 255


@dataclass
class AutoMask:
    """A single mask from automatic mask generation.

    Attributes:
        data: Binary mask ``(H, W)``.
        score: Model prediction of mask quality.
        area: Pixel area.
        bbox: Bounding box in ``[x, y, w, h]`` format.
        stability_score: Stability of the mask under threshold changes.
        crop_box: The crop region used, in ``[x, y, w, h]`` format.
        point_coords: Point coordinates used to generate this mask.
    """

    data: npt.NDArray[np.bool_]
    score: float
    area: int
    bbox: list[float]
    stability_score: float
    crop_box: list[float]
    point_coords: list[list[float]] | None = None


@dataclass
class AutoMaskResult:
    """Result of automatic mask generation on one image.

    Attributes:
        masks: List of :class:`AutoMask` objects sorted by area (largest first).
        image_shape: ``(H, W)`` of the source image (optional).
    """

    masks: list[AutoMask]
    image_shape: tuple[int, int] | None = None

    def numpy(self) -> npt.NDArray[np.bool_]:
        """Stack all masks into ``(N, H, W)``."""
        if not self.masks:
            shape = self.image_shape or (0, 0)
            return np.zeros((0, *shape), dtype=bool)
        return np.stack([m.data for m in self.masks])

    def filter_by_area(self, min_area: int = 0, max_area: int | None = None) -> AutoMaskResult:
        """Return a new result keeping only masks within the area range."""
        filtered = [
            m for m in self.masks if m.area >= min_area and (max_area is None or m.area <= max_area)
        ]
        return AutoMaskResult(masks=filtered, image_shape=self.image_shape)

    def filter_by_iou(self, min_iou: float = 0.0) -> AutoMaskResult:
        """Return a new result keeping only masks above the IoU threshold."""
        filtered = [m for m in self.masks if m.score >= min_iou]
        return AutoMaskResult(masks=filtered, image_shape=self.image_shape)


@dataclass
class FrameMasks:
    """Masks for a single video frame.

    Attributes:
        frame_idx: Zero-based frame index.
        masks: Mapping from object id to binary mask ``(H, W)``.
    """

    frame_idx: int
    masks: dict[int, npt.NDArray[np.bool_]]

    @property
    def object_ids(self) -> list[int]:
        """Return sorted list of object ids for this frame."""
        return sorted(self.masks.keys())


@dataclass
class VideoResults:
    """Full propagation results for a video.

    Attributes:
        frames: List of :class:`FrameMasks`, one per tracked frame.
        video_dir: Path to the source frame directory.
        num_frames: Total number of frames in the video.
    """

    frames: list[FrameMasks] = field(default_factory=list)
    video_dir: Path | None = None
    num_frames: int = 0
    _frame_index_cache: dict[int, FrameMasks] = field(default_factory=dict, init=False, repr=False)
    _frame_index_cache_size: int = field(default=0, init=False, repr=False)

    # Internal index for fast frame lookup (populated lazily)
    def _frame_index(self) -> dict[int, FrameMasks]:
        if self._frame_index_cache_size != len(self.frames):
            self._frame_index_cache = {fm.frame_idx: fm for fm in self.frames}
            self._frame_index_cache_size = len(self.frames)
        return self._frame_index_cache

    # -- iteration helpers --------------------------------------------------

    def __iter__(self) -> Iterator[tuple[int, FrameMasks]]:
        """Iterate over ``(frame_idx, FrameMasks)`` pairs sorted by frame index.

        Example::

            for frame_idx, frame in results:
                print(frame_idx, frame.object_ids)
        """
        for fm in sorted(self.frames, key=lambda f: f.frame_idx):
            yield fm.frame_idx, fm

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, frame_idx: int) -> FrameMasks:
        """Return :class:`FrameMasks` for *frame_idx*.

        Raises:
            IndexError: If the frame was not tracked.
        """
        index = self._frame_index()
        if frame_idx not in index:
            msg = f"Frame {frame_idx} is not present in these results."
            raise IndexError(msg)
        return index[frame_idx]

    def values(self) -> list[FrameMasks]:
        """Return :class:`FrameMasks` objects sorted by frame index."""
        return sorted(self.frames, key=lambda f: f.frame_idx)

    def get_frame_masks(self, frame_idx: int) -> FrameMasks | None:
        """Return masks for a frame, or ``None`` if absent."""
        return self._frame_index().get(frame_idx)

    def get_object_masks(self, obj_id: int) -> dict[int, npt.NDArray[np.bool_]]:
        """Get all masks for a specific object across frames.

        Returns:
            Mapping from ``frame_idx`` to the boolean mask for the object.
        """
        result: dict[int, npt.NDArray[np.bool_]] = {}
        for fm in self.frames:
            if obj_id in fm.masks:
                result[fm.frame_idx] = fm.masks[obj_id]
        return result

    @property
    def object_ids(self) -> set[int]:
        """All unique object ids across all frames."""
        ids: set[int] = set()
        for fm in self.frames:
            ids.update(fm.masks.keys())
        return ids
