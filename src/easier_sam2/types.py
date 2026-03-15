"""Core type definitions for easier-sam2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt


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

# Checkpoint download URLs
CHECKPOINT_URLS: dict[ModelSize, str] = {
    ModelSize.TINY: "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    ModelSize.SMALL: "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    ModelSize.BASE_PLUS: "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    ModelSize.LARGE: "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}


# ---- Convenience type aliases ----
PointCoords = Union[Sequence[Sequence[float]], npt.NDArray[np.floating[Any]]]
PointLabels = Union[Sequence[int], npt.NDArray[np.integer[Any]]]
BoundingBox = Union[
    Sequence[float],  # [x1, y1, x2, y2]
    npt.NDArray[np.floating[Any]],
]
MaskInput = npt.NDArray[np.uint8]  # H×W binary mask
ImageInput = Union[npt.NDArray[np.uint8], "PIL.Image.Image"]  # type: ignore[name-defined]


@dataclass
class Mask:
    """A single segmentation mask with metadata.

    Attributes:
        mask: Binary mask array of shape ``(H, W)`` with dtype ``bool``.
        score: Model-predicted IoU quality score in ``[0, 1]``.
        logits: Low-resolution logit mask (useful for iterative refinement).
        area: Pixel area of the mask.
    """

    mask: npt.NDArray[np.bool_]
    score: float
    logits: Optional[npt.NDArray[np.floating[Any]]] = None
    area: Optional[int] = None

    def __post_init__(self) -> None:
        if self.area is None:
            self.area = int(self.mask.sum())


@dataclass
class ImagePrediction:
    """Result of segmenting a single image.

    Attributes:
        masks: List of predicted :class:`Mask` objects.
        image_shape: ``(H, W)`` of the original image.
    """

    masks: list[Mask]
    image_shape: tuple[int, int]

    # -- convenience helpers ------------------------------------------------

    @property
    def best_mask(self) -> Mask:
        """Return the mask with the highest IoU score."""
        return max(self.masks, key=lambda m: m.score)

    def numpy(self) -> npt.NDArray[np.bool_]:
        """Stack all masks into a ``(N, H, W)`` boolean array."""
        return np.stack([m.mask for m in self.masks])

    def as_uint8(self) -> npt.NDArray[np.uint8]:
        """Stack all masks into a ``(N, H, W)`` uint8 array (0/255)."""
        return (self.numpy().astype(np.uint8)) * 255


@dataclass
class AutoMask:
    """A single mask from automatic mask generation.

    Attributes:
        mask: Binary mask ``(H, W)``.
        area: Pixel area.
        bbox: Bounding box in ``[x, y, w, h]`` format.
        predicted_iou: Model prediction of mask quality.
        stability_score: Stability of the mask under threshold changes.
        point_coords: Point coordinates used to generate this mask.
        crop_box: The crop region used, in ``[x, y, w, h]`` format.
    """

    mask: npt.NDArray[np.bool_]
    area: int
    bbox: list[float]
    predicted_iou: float
    stability_score: float
    point_coords: list[list[float]]
    crop_box: list[float]


@dataclass
class AutoMaskResult:
    """Result of automatic mask generation on one image.

    Attributes:
        masks: List of :class:`AutoMask` objects sorted by area (largest first).
        image_shape: ``(H, W)`` of the source image.
    """

    masks: list[AutoMask]
    image_shape: tuple[int, int]

    def numpy(self) -> npt.NDArray[np.bool_]:
        """Stack all masks into ``(N, H, W)``."""
        if not self.masks:
            return np.zeros((0, *self.image_shape), dtype=bool)
        return np.stack([m.mask for m in self.masks])

    def filter_by_area(self, min_area: int = 0, max_area: int | None = None) -> AutoMaskResult:
        """Return a new result keeping only masks within the area range."""
        filtered = [
            m
            for m in self.masks
            if m.area >= min_area and (max_area is None or m.area <= max_area)
        ]
        return AutoMaskResult(masks=filtered, image_shape=self.image_shape)

    def filter_by_iou(self, min_iou: float = 0.0) -> AutoMaskResult:
        """Return a new result keeping only masks above the IoU threshold."""
        filtered = [m for m in self.masks if m.predicted_iou >= min_iou]
        return AutoMaskResult(masks=filtered, image_shape=self.image_shape)


@dataclass
class FrameMasks:
    """Masks for a single video frame.

    Attributes:
        frame_idx: Zero-based frame index.
        object_ids: Object ids corresponding to each mask (length ``N``).
        masks: Binary masks of shape ``(N, H, W)``.
        scores: Raw mask logits/scores per object of shape ``(N, 1, H, W)``.
    """

    frame_idx: int
    object_ids: list[int]
    masks: npt.NDArray[np.bool_]
    scores: Optional[npt.NDArray[np.floating[Any]]] = None


@dataclass
class VideoResults:
    """Full propagation results for a video.

    Attributes:
        frame_masks: Mapping from frame index to :class:`FrameMasks`.
        video_dir: Path to the source frame directory.
        num_frames: Total number of frames in the video.
    """

    frame_masks: dict[int, FrameMasks] = field(default_factory=dict)
    video_dir: Optional[Path] = None
    num_frames: int = 0

    # -- iteration helpers --------------------------------------------------

    def __iter__(self):  # noqa: ANN204
        """Iterate over frame masks sorted by frame index."""
        for idx in sorted(self.frame_masks):
            yield self.frame_masks[idx]

    def __len__(self) -> int:
        return len(self.frame_masks)

    def __getitem__(self, frame_idx: int) -> FrameMasks:
        return self.frame_masks[frame_idx]

    def get_object_masks(self, obj_id: int) -> dict[int, npt.NDArray[np.bool_]]:
        """Get all masks for a specific object across frames.

        Returns:
            Mapping from ``frame_idx`` to the boolean mask for the object.
        """
        result: dict[int, npt.NDArray[np.bool_]] = {}
        for frame_idx, fm in self.frame_masks.items():
            if obj_id in fm.object_ids:
                idx = fm.object_ids.index(obj_id)
                result[frame_idx] = fm.masks[idx]
        return result

    @property
    def object_ids(self) -> list[int]:
        """All unique object ids across all frames."""
        ids: set[int] = set()
        for fm in self.frame_masks.values():
            ids.update(fm.object_ids)
        return sorted(ids)
