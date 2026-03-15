"""Tests for types module – no SAM2 model required."""

from __future__ import annotations

import numpy as np

from easier_sam2.types import (
    AutoMask,
    AutoMaskResult,
    FrameMasks,
    ImagePrediction,
    Mask,
    ModelSize,
    VideoResults,
)


class TestModelSize:
    """ModelSize enum tests."""

    def test_members(self) -> None:
        assert set(ModelSize) == {
            ModelSize.TINY,
            ModelSize.SMALL,
            ModelSize.BASE_PLUS,
            ModelSize.LARGE,
        }

    def test_values(self) -> None:
        assert ModelSize.TINY.value == "tiny"
        assert ModelSize.BASE_PLUS.value == "base_plus"


class TestMask:
    """Mask dataclass tests."""

    def test_numpy(self) -> None:
        data = np.ones((10, 10), dtype=bool)
        logits = np.zeros((1, 256, 256), dtype=np.float32)
        m = Mask(data=data, score=0.95, logits=logits)
        assert m.numpy().dtype == bool
        assert m.numpy().shape == (10, 10)

    def test_as_uint8(self) -> None:
        data = np.ones((10, 10), dtype=bool)
        logits = np.zeros((1, 256, 256), dtype=np.float32)
        m = Mask(data=data, score=0.95, logits=logits)
        u = m.as_uint8()
        assert u.dtype == np.uint8
        assert u.max() == 255

    def test_save(self, tmp_path: object) -> None:
        from pathlib import Path

        data = np.ones((10, 10), dtype=bool)
        logits = np.zeros((1, 256, 256), dtype=np.float32)
        m = Mask(data=data, score=0.95, logits=logits)

        out = Path(str(tmp_path)) / "mask.png"
        m.save(out)
        assert out.exists()


class TestImagePrediction:
    """ImagePrediction tests."""

    def test_best_mask(self) -> None:
        logits = np.zeros((1, 256, 256), dtype=np.float32)
        masks = [
            Mask(data=np.zeros((10, 10), dtype=bool), score=0.5, logits=logits),
            Mask(data=np.ones((10, 10), dtype=bool), score=0.9, logits=logits),
            Mask(data=np.zeros((10, 10), dtype=bool), score=0.7, logits=logits),
        ]
        pred = ImagePrediction(masks=masks)
        assert pred.best_mask.score == 0.9

    def test_len(self) -> None:
        logits = np.zeros((1, 256, 256), dtype=np.float32)
        masks = [
            Mask(data=np.zeros((10, 10), dtype=bool), score=0.5, logits=logits),
        ]
        pred = ImagePrediction(masks=masks)
        assert len(pred) == 1


class TestAutoMaskResult:
    """AutoMaskResult filtering tests."""

    def _make_result(self) -> AutoMaskResult:
        return AutoMaskResult(
            masks=[
                AutoMask(
                    data=np.ones((10, 10), dtype=bool),
                    score=0.95,
                    area=100,
                    bbox=[0, 0, 10, 10],
                    stability_score=0.9,
                    crop_box=[0, 0, 10, 10],
                ),
                AutoMask(
                    data=np.ones((5, 5), dtype=bool),
                    score=0.7,
                    area=25,
                    bbox=[0, 0, 5, 5],
                    stability_score=0.8,
                    crop_box=[0, 0, 5, 5],
                ),
            ]
        )

    def test_filter_by_area(self) -> None:
        r = self._make_result()
        filtered = r.filter_by_area(min_area=50)
        assert len(filtered.masks) == 1
        assert filtered.masks[0].area == 100

    def test_filter_by_iou(self) -> None:
        r = self._make_result()
        filtered = r.filter_by_iou(min_iou=0.8)
        assert len(filtered.masks) == 1
        assert filtered.masks[0].score == 0.95


class TestVideoResults:
    """VideoResults tests."""

    def test_iteration(self) -> None:
        fm = FrameMasks(
            frame_idx=0, masks={1: np.ones((10, 10), dtype=bool)}
        )
        vr = VideoResults(frames=[fm])
        assert len(vr) == 1
        idx, frame_masks = next(iter(vr))
        assert idx == 0

    def test_getitem(self) -> None:
        fm0 = FrameMasks(frame_idx=0, masks={1: np.ones((10, 10), dtype=bool)})
        fm1 = FrameMasks(frame_idx=1, masks={1: np.zeros((10, 10), dtype=bool)})
        vr = VideoResults(frames=[fm0, fm1])
        assert vr[0].frame_idx == 0
        assert vr[1].frame_idx == 1

    def test_get_object_masks(self) -> None:
        fm0 = FrameMasks(frame_idx=0, masks={1: np.ones((10, 10), dtype=bool)})
        fm1 = FrameMasks(
            frame_idx=1,
            masks={1: np.zeros((10, 10), dtype=bool), 2: np.ones((10, 10), dtype=bool)},
        )
        vr = VideoResults(frames=[fm0, fm1])
        obj1 = vr.get_object_masks(1)
        assert len(obj1) == 2
        obj2 = vr.get_object_masks(2)
        assert len(obj2) == 1

    def test_object_ids(self) -> None:
        fm0 = FrameMasks(frame_idx=0, masks={1: np.ones((10, 10), dtype=bool)})
        fm1 = FrameMasks(frame_idx=1, masks={1: np.ones((10, 10), dtype=bool), 3: np.ones((10, 10), dtype=bool)})
        vr = VideoResults(frames=[fm0, fm1])
        assert vr.object_ids == {1, 3}
