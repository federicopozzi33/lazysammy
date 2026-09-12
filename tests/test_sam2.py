"""Tests for the SAM2 facade - no SAM2 model required.

Sub-components are stubbed so we can assert the facade delegates correctly
without loading weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from lazysammy.sam2 import SAM2
from lazysammy.types import (
    AutoMask,
    AutoMaskResult,
    ImagePrediction,
    Mask,
)


class _StubImageSegmenter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.device = torch.device("cpu")

    def segment(self, image: Any, **kwargs: Any) -> ImagePrediction:
        self.calls.append(("segment", kwargs))
        return ImagePrediction(masks=[Mask(data=np.ones((4, 4), dtype=bool), score=0.9)])

    def segment_point(self, image: Any, x: float, y: float, **kwargs: Any) -> ImagePrediction:
        self.calls.append(("segment_point", {"x": x, "y": y, **kwargs}))
        return ImagePrediction(masks=[Mask(data=np.ones((4, 4), dtype=bool), score=0.9)])

    def segment_box(
        self, image: Any, x1: float, y1: float, x2: float, y2: float
    ) -> ImagePrediction:
        self.calls.append(("segment_box", {"x1": x1, "y1": y1, "x2": x2, "y2": y2}))
        return ImagePrediction(masks=[Mask(data=np.ones((4, 4), dtype=bool), score=0.9)])

    def segment_multi_box(self, image: Any, boxes: Any) -> list[ImagePrediction]:
        self.calls.append(("segment_multi_box", {"boxes": boxes}))
        return [ImagePrediction(masks=[Mask(data=np.ones((4, 4), dtype=bool), score=0.9)])]

    def segment_batch(self, images: Any, **kwargs: Any) -> list[ImagePrediction]:
        self.calls.append(("segment_batch", kwargs))
        return [ImagePrediction(masks=[Mask(data=np.ones((4, 4), dtype=bool), score=0.9)])]

    def refine(self, image: Any, previous_logits: Any, **kwargs: Any) -> ImagePrediction:
        self.calls.append(("refine", kwargs))
        return ImagePrediction(masks=[Mask(data=np.ones((4, 4), dtype=bool), score=0.9)])


class _StubAutoSegmenter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, image: Any) -> AutoMaskResult:
        self.calls.append("generate")
        return AutoMaskResult(
            masks=[
                AutoMask(
                    data=np.ones((4, 4), dtype=bool),
                    score=0.9,
                    area=16,
                    bbox=[0, 0, 4, 4],
                    stability_score=0.9,
                    crop_box=[0, 0, 4, 4],
                )
            ],
            image_shape=(4, 4),
        )


class _StubVideoTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def new_session(self, video: Any, **kwargs: Any) -> str:
        self.calls.append(("new_session", kwargs))
        return "session"


def _stubbed_sam(**kwargs: Any) -> SAM2:
    """Return a SAM2 whose sub-components are replaced by stubs."""
    sam = SAM2("large", **kwargs)
    sam._image_segmenter = _StubImageSegmenter()  # type: ignore[assignment]
    sam._auto_segmenter = _StubAutoSegmenter()  # type: ignore[assignment]
    sam._video_tracker = _StubVideoTracker()  # type: ignore[assignment]
    return sam


class TestSAM2Init:
    """SAM2 facade initialisation tests."""

    def test_lazy_init(self) -> None:
        """Sub-components should not be created until accessed."""
        sam = SAM2("large")
        assert sam._image_segmenter is None
        assert sam._video_tracker is None
        assert sam._auto_segmenter is None

    def test_stores_config(self) -> None:
        sam = SAM2("small", device="cpu")
        assert sam._model_size == "small"
        assert sam._device == "cpu"
        assert sam._checkpoint is None
        assert sam._vos_optimized is False


class TestSAM2Properties:
    def test_exposes_configuration(self) -> None:
        sam = SAM2("tiny", device="cuda:1", checkpoint="/tmp/ckpt.pt", vos_optimized=True)
        assert sam.model_size == "tiny"
        assert sam.device == "cuda:1"
        assert sam.checkpoint == "/tmp/ckpt.pt"
        assert sam.vos_optimized is True

    def test_resolved_device_uses_autodetect(self) -> None:
        sam = SAM2("tiny", device="cpu")
        assert sam.resolved_device == torch.device("cpu")


class TestSAM2ImageDelegation:
    def test_segment_delegates(self, small_image: np.ndarray) -> None:
        sam = _stubbed_sam()
        pred = sam.segment(small_image, points=[[1, 2]], labels=[1])
        assert isinstance(pred, ImagePrediction)
        assert sam._image_segmenter.calls[0][0] == "segment"  # type: ignore[union-attr]

    def test_segment_point_delegates(self, small_image: np.ndarray) -> None:
        sam = _stubbed_sam()
        sam.segment_point(small_image, x=5, y=6)
        _, kwargs = sam._image_segmenter.calls[0]  # type: ignore[union-attr]
        assert kwargs["x"] == 5
        assert kwargs["y"] == 6

    def test_segment_box_delegates(self, small_image: np.ndarray) -> None:
        sam = _stubbed_sam()
        sam.segment_box(small_image, 1, 2, 3, 4)
        _, kwargs = sam._image_segmenter.calls[0]  # type: ignore[union-attr]
        assert kwargs == {"x1": 1, "y1": 2, "x2": 3, "y2": 4}

    def test_segment_multi_box_delegates(self, small_image: np.ndarray) -> None:
        sam = _stubbed_sam()
        out = sam.segment_multi_box(small_image, [[1, 2, 3, 4]])
        assert len(out) == 1

    def test_segment_batch_delegates(self, small_image: np.ndarray) -> None:
        sam = _stubbed_sam()
        out = sam.segment_batch([small_image])
        assert len(out) == 1
        assert sam._image_segmenter.calls[0][0] == "segment_batch"  # type: ignore[union-attr]

    def test_refine_delegates(self, small_image: np.ndarray) -> None:
        sam = _stubbed_sam()
        logits = np.zeros((1, 256, 256), dtype=np.float32)
        sam.refine(small_image, logits, points=[[1, 1]], labels=[1])
        assert sam._image_segmenter.calls[0][0] == "refine"  # type: ignore[union-attr]


class TestSAM2AutoDelegation:
    def test_auto_segment_uses_shared_segmenter(self, small_image: np.ndarray) -> None:
        sam = _stubbed_sam()
        result = sam.auto_segment(small_image)

        assert isinstance(result, AutoMaskResult)
        assert sam._auto_segmenter.calls == ["generate"]  # type: ignore[union-attr]

    def test_auto_segment_with_tuning_builds_new_segmenter(
        self, small_image: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: dict[str, Any] = {}

        class _Tuned:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                created.update(kwargs)

            def generate(self, image: Any) -> AutoMaskResult:
                return AutoMaskResult(masks=[], image_shape=(4, 4))

        monkeypatch.setattr("lazysammy.sam2.AutoSegmenter", _Tuned)

        sam = _stubbed_sam()
        result = sam.auto_segment(small_image, points_per_side=8, use_m2m=True)

        assert result.masks == []
        assert created["points_per_side"] == 8
        assert created["use_m2m"] is True
        # Defaults are filled in for the options that were not supplied.
        assert created["pred_iou_thresh"] == pytest.approx(0.8)


class TestSAM2VideoDelegation:
    def test_video_delegates_to_tracker(self) -> None:
        sam = _stubbed_sam()
        session = sam.video("frames/", every_n=2)
        assert session == "session"
        _, kwargs = sam._video_tracker.calls[0]  # type: ignore[union-attr]
        assert kwargs["every_n"] == 2


class TestSAM2Save:
    def test_save_image_prediction(self, tmp_path: Path) -> None:
        sam = _stubbed_sam()
        pred = ImagePrediction(masks=[Mask(data=np.ones((4, 4), dtype=bool), score=0.9)])
        out = sam.save(pred, tmp_path / "img")
        assert (out / "mask_0000.png").exists()

    def test_save_auto_mask_result(self, tmp_path: Path) -> None:
        sam = _stubbed_sam()
        result = sam.auto_segment(np.zeros((4, 4, 3), dtype=np.uint8))
        out = sam.save(result, tmp_path / "auto")
        assert (out / "auto_mask_0000.png").exists()

    def test_save_rejects_unsupported_type(self, tmp_path: Path) -> None:
        sam = _stubbed_sam()
        with pytest.raises(TypeError, match="ImagePrediction or AutoMaskResult"):
            sam.save("not-a-result", tmp_path)  # type: ignore[arg-type]
