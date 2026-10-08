"""Core type definitions for lazysammy."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from lazysammy.utils import mask_to_rle, rle_to_mask
from lazysammy.validation import validate_nonempty_masks, validate_save_format


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

    # -- numpy interop ------------------------------------------------------

    def __array__(self, dtype: Any = None) -> npt.NDArray[Any]:
        """Expose the mask to numpy (``np.asarray(mask)``)."""
        return np.asarray(self.data, dtype=dtype)

    def __repr__(self) -> str:
        """Return a compact repr that does not dump the whole array."""
        h, w = self.data.shape if self.data.ndim == 2 else (0, 0)
        return f"Mask(shape=({h}, {w}), score={self.score:.3f}, area={self.area})"

    def __eq__(self, other: object) -> bool:
        """Compare masks by value, tolerating numpy array fields."""
        if not isinstance(other, Mask):
            return NotImplemented
        if self.score != other.score or self.area != other.area:
            return False
        if not np.array_equal(self.data, other.data):
            return False
        if (self.logits is None) != (other.logits is None):
            return False
        if self.logits is not None and other.logits is not None:
            return bool(np.array_equal(self.logits, other.logits))
        return True

    __hash__ = None  # type: ignore[assignment]  # mutable, array-backed

    # -- set operations -----------------------------------------------------

    def _combine(self, other: Mask, op: Any) -> Mask:
        if not isinstance(other, Mask):
            return NotImplemented
        if self.data.shape != other.data.shape:
            msg = (
                f"Cannot combine masks with different shapes: "
                f"{self.data.shape} vs {other.data.shape}."
            )
            raise ValueError(msg)
        return Mask(data=op(self.data, other.data), score=min(self.score, other.score))

    def __and__(self, other: Mask) -> Mask:
        """Intersection of two masks (``mask_a & mask_b``)."""
        return self._combine(other, np.logical_and)

    def __or__(self, other: Mask) -> Mask:
        """Union of two masks (``mask_a | mask_b``)."""
        return self._combine(other, np.logical_or)

    def __invert__(self) -> Mask:
        """Complement of the mask (``~mask``)."""
        return Mask(data=np.logical_not(self.data), score=self.score)

    # -- geometry -----------------------------------------------------------

    @property
    def bbox(self) -> list[int]:
        """Bounding box as ``[x_min, y_min, x_max, y_max]`` (inclusive).

        Raises:
            ValueError: If the mask is empty.
        """
        if not np.any(self.data):
            msg = "mask is empty; cannot compute a bounding box."
            raise ValueError(msg)
        rows = np.any(self.data, axis=1)
        cols = np.any(self.data, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    @property
    def centroid(self) -> tuple[float, float]:
        """Centroid ``(x, y)`` of the mask.

        Raises:
            ValueError: If the mask is empty.
        """
        if not np.any(self.data):
            msg = "mask is empty; cannot compute a centroid."
            raise ValueError(msg)
        ys, xs = np.nonzero(self.data)
        return float(xs.mean()), float(ys.mean())

    def iou(self, other: Mask) -> float:
        """Intersection-over-Union with *other* in ``[0, 1]``."""
        if self.data.shape != other.data.shape:
            msg = (
                f"Cannot compute IoU for masks with different shapes: "
                f"{self.data.shape} vs {other.data.shape}."
            )
            raise ValueError(msg)
        intersection = np.logical_and(self.data, other.data).sum()
        union = np.logical_or(self.data, other.data).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    # -- conversion ---------------------------------------------------------

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

    def to_rle(self) -> dict[str, Any]:
        """Encode the mask as a COCO-style RLE dict (``size`` + ``counts``)."""
        return mask_to_rle(self.data)

    @classmethod
    def from_rle(
        cls,
        rle: dict[str, Any],
        *,
        score: float = 0.0,
    ) -> Mask:
        """Decode a COCO-style RLE dict into a :class:`Mask`."""
        return cls(data=rle_to_mask(rle), score=score)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (mask as COCO RLE)."""
        return {"score": self.score, "area": self.area, "rle": self.to_rle()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mask:
        """Reconstruct a :class:`Mask` from :meth:`to_dict` output."""
        return cls.from_rle(data["rle"], score=float(data["score"]))

    def save(self, path: str | Path, *, fmt: str | None = None) -> Path:
        """Save the mask to disk.

        The format is inferred from the file extension unless *fmt* is given.

        Args:
            path: Destination path. Parent directories are created.
            fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``. When ``None`` the
                extension decides (``.png``/``.npy``/``.json``); a path with no
                extension defaults to PNG.

        Returns:
            The written path.

        Raises:
            ValueError: If the format is unsupported.
        """
        import json

        import cv2

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if fmt is None:
            suffix = p.suffix.lower()
            if suffix in {"", ".png"}:
                fmt = "png"
            elif suffix == ".npy":
                fmt = "npy"
            elif suffix == ".json":
                fmt = "coco_rle"
            else:
                msg = (
                    f"Cannot infer a mask format from {p.suffix!r}; pass fmt= "
                    "('png', 'npy', or 'coco_rle')."
                )
                raise ValueError(msg)

        normalized = validate_save_format(fmt)
        if normalized == "png":
            cv2.imwrite(str(p), self.as_uint8())
        elif normalized == "npy":
            # Write through a file handle so numpy does not append ".npy" and
            # silently ignore the caller's chosen path.
            with open(p, "wb") as handle:
                np.save(handle, self.data)
        else:  # coco_rle
            p.write_text(json.dumps(self.to_rle()))
        return p


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

    def __iter__(self) -> Iterator[Mask]:
        """Iterate over the predicted masks."""
        return iter(self.masks)

    def __getitem__(self, index: int) -> Mask:
        """Return the mask at *index*."""
        return self.masks[index]

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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (masks as COCO RLE)."""
        return {
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "masks": [m.to_dict() for m in self.masks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImagePrediction:
        """Reconstruct an :class:`ImagePrediction` from :meth:`to_dict` output."""
        shape = data.get("image_shape")
        masks = [Mask.from_dict(entry) for entry in data.get("masks", [])]
        return cls(masks=masks, image_shape=tuple(shape) if shape else None)


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

    def __array__(self, dtype: Any = None) -> npt.NDArray[Any]:
        """Expose the mask to numpy (``np.asarray(auto_mask)``)."""
        return np.asarray(self.data, dtype=dtype)

    def __repr__(self) -> str:
        """Return a compact repr that does not dump the whole array."""
        h, w = self.data.shape if self.data.ndim == 2 else (0, 0)
        return (
            f"AutoMask(shape=({h}, {w}), score={self.score:.3f}, "
            f"area={self.area}, stability={self.stability_score:.3f})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare auto-masks by value, tolerating numpy array fields."""
        if not isinstance(other, AutoMask):
            return NotImplemented
        return (
            self.score == other.score
            and self.area == other.area
            and self.stability_score == other.stability_score
            and self.bbox == other.bbox
            and self.crop_box == other.crop_box
            and self.point_coords == other.point_coords
            and bool(np.array_equal(self.data, other.data))
        )

    __hash__ = None  # type: ignore[assignment]  # mutable, array-backed

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (mask as COCO RLE)."""
        return {
            "score": self.score,
            "area": self.area,
            "bbox": list(self.bbox),
            "stability_score": self.stability_score,
            "crop_box": list(self.crop_box),
            "point_coords": self.point_coords,
            "rle": mask_to_rle(self.data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutoMask:
        """Reconstruct an :class:`AutoMask` from :meth:`to_dict` output."""
        return cls(
            data=rle_to_mask(data["rle"]),
            score=float(data["score"]),
            area=int(data["area"]),
            bbox=list(data["bbox"]),
            stability_score=float(data["stability_score"]),
            crop_box=list(data["crop_box"]),
            point_coords=data.get("point_coords"),
        )


@dataclass
class AutoMaskResult:
    """Result of automatic mask generation on one image.

    Attributes:
        masks: List of :class:`AutoMask` objects sorted by area (largest first).
        image_shape: ``(H, W)`` of the source image (optional).
    """

    masks: list[AutoMask]
    image_shape: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.masks)

    def __iter__(self) -> Iterator[AutoMask]:
        """Iterate over the generated masks (largest area first)."""
        return iter(self.masks)

    def __getitem__(self, index: int | slice) -> AutoMask | AutoMaskResult:
        """Return the mask at *index*, or a new result for a slice."""
        if isinstance(index, slice):
            return AutoMaskResult(masks=self.masks[index], image_shape=self.image_shape)
        return self.masks[index]

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

    def sort_by(self, key: str = "area", *, descending: bool = True) -> AutoMaskResult:
        """Return a new result sorted by *key*.

        Args:
            key: One of ``"area"``, ``"score"``, or ``"stability_score"``.
            descending: Sort largest/highest first (default).

        Returns:
            A new :class:`AutoMaskResult` with reordered masks.

        Raises:
            ValueError: If *key* is not a sortable attribute.
        """
        if key not in {"area", "score", "stability_score"}:
            msg = f"sort_by key must be 'area', 'score', or 'stability_score'; got {key!r}."
            raise ValueError(msg)
        ordered = sorted(self.masks, key=lambda m: getattr(m, key), reverse=descending)
        return AutoMaskResult(masks=ordered, image_shape=self.image_shape)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "masks": [m.to_dict() for m in self.masks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutoMaskResult:
        """Reconstruct an :class:`AutoMaskResult` from :meth:`to_dict` output."""
        shape = data.get("image_shape")
        return cls(
            masks=[AutoMask.from_dict(entry) for entry in data.get("masks", [])],
            image_shape=tuple(shape) if shape else None,
        )


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

    def __eq__(self, other: object) -> bool:
        """Compare frames by value, tolerating numpy array fields."""
        if not isinstance(other, FrameMasks):
            return NotImplemented
        if self.frame_idx != other.frame_idx:
            return False
        if self.masks.keys() != other.masks.keys():
            return False
        return all(bool(np.array_equal(self.masks[oid], other.masks[oid])) for oid in self.masks)

    __hash__ = None  # type: ignore[assignment]  # mutable, array-backed

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (masks as COCO RLE)."""
        return {
            "frame_idx": self.frame_idx,
            "masks": {str(oid): mask_to_rle(m) for oid, m in self.masks.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameMasks:
        """Reconstruct a :class:`FrameMasks` from :meth:`to_dict` output."""
        return cls(
            frame_idx=int(data["frame_idx"]),
            masks={int(oid): rle_to_mask(rle) for oid, rle in data["masks"].items()},
        )


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

    def __getitem__(self, frame_idx: int | slice) -> FrameMasks | VideoResults:
        """Return :class:`FrameMasks` for *frame_idx*, or a sliced result.

        Args:
            frame_idx: A frame index, or a slice over the sorted frame list.

        Returns:
            A :class:`FrameMasks` for an integer index, or a new
            :class:`VideoResults` for a slice.

        Raises:
            IndexError: If the frame was not tracked.
        """
        if isinstance(frame_idx, slice):
            selected = self.values()[frame_idx]
            return VideoResults(
                frames=selected,
                video_dir=self.video_dir,
                num_frames=self.num_frames,
            )
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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (masks as COCO RLE)."""
        return {
            "video_dir": str(self.video_dir) if self.video_dir is not None else None,
            "num_frames": self.num_frames,
            "frames": [fm.to_dict() for fm in self.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoResults:
        """Reconstruct a :class:`VideoResults` from :meth:`to_dict` output."""
        video_dir = data.get("video_dir")
        return cls(
            frames=[FrameMasks.from_dict(entry) for entry in data.get("frames", [])],
            video_dir=Path(video_dir) if video_dir else None,
            num_frames=int(data.get("num_frames", 0)),
        )


# ---------------------------------------------------------------------------
# SAM 3 concept segmentation
# ---------------------------------------------------------------------------


@dataclass
class ConceptPrediction:
    """Result of a SAM 3 open-vocabulary concept segmentation.

    SAM 3 detects and segments *every* instance of a concept described by a
    short text phrase (or image exemplars), so a single prediction can contain
    many objects. Each object carries its own mask, score, and bounding box.

    Attributes:
        masks: One :class:`Mask` per detected instance.
        boxes: Bounding boxes as ``[x1, y1, x2, y2]``, aligned with *masks*.
        concept: The text phrase (or ``"visual"`` for exemplar prompts) that
            produced these instances.
        image_shape: ``(H, W)`` of the source image (optional).
    """

    masks: list[Mask]
    boxes: list[list[float]] = field(default_factory=list)
    concept: str | None = None
    image_shape: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.masks)

    def __iter__(self) -> Iterator[Mask]:
        """Iterate over the detected instance masks."""
        return iter(self.masks)

    def __getitem__(self, index: int) -> Mask:
        """Return the instance mask at *index*."""
        return self.masks[index]

    @property
    def best_mask(self) -> Mask:
        """Return the highest-scoring instance mask.

        Raises:
            ValueError: If the prediction contains no masks.
        """
        validate_nonempty_masks(len(self.masks), context="ConceptPrediction")
        return max(self.masks, key=lambda m: m.score)

    def numpy(self) -> npt.NDArray[np.bool_]:
        """Stack all instance masks into a ``(N, H, W)`` boolean array."""
        if not self.masks:
            shape = self.image_shape or (0, 0)
            return np.zeros((0, *shape), dtype=bool)
        return np.stack([m.data for m in self.masks])

    def filter_by_score(self, min_score: float) -> ConceptPrediction:
        """Return a new prediction keeping only instances above *min_score*.

        Boxes are optional (the interactive path produces masks without boxes),
        so they are carried through by index rather than zipped: a prediction
        with masks but no boxes still filters correctly.
        """
        keep = [i for i, mask in enumerate(self.masks) if mask.score >= min_score]
        return ConceptPrediction(
            masks=[self.masks[i] for i in keep],
            boxes=[self.boxes[i] for i in keep if i < len(self.boxes)],
            concept=self.concept,
            image_shape=self.image_shape,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (masks as COCO RLE)."""
        return {
            "concept": self.concept,
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "boxes": [list(box) for box in self.boxes],
            "masks": [m.to_dict() for m in self.masks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptPrediction:
        """Reconstruct a :class:`ConceptPrediction` from :meth:`to_dict` output."""
        shape = data.get("image_shape")
        return cls(
            masks=[Mask.from_dict(entry) for entry in data.get("masks", [])],
            boxes=[list(box) for box in data.get("boxes", [])],
            concept=data.get("concept"),
            image_shape=tuple(shape) if shape else None,
        )
