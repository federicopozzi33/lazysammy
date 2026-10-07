"""Tests for types module - no SAM2 model required."""

from __future__ import annotations

import numpy as np
import pytest

from lazysammy.types import (
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

    def test_best_mask_raises_for_empty_prediction(self) -> None:
        pred = ImagePrediction(masks=[])
        with pytest.raises(ValueError, match="does not contain any masks"):
            _ = pred.best_mask


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
        fm = FrameMasks(frame_idx=0, masks={1: np.ones((10, 10), dtype=bool)})
        vr = VideoResults(frames=[fm])
        assert len(vr) == 1
        idx, _frame_masks = next(iter(vr))
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

    def test_get_frame_masks(self) -> None:
        fm0 = FrameMasks(frame_idx=0, masks={1: np.ones((10, 10), dtype=bool)})
        vr = VideoResults(frames=[fm0])
        assert vr.get_frame_masks(0) is fm0
        assert vr.get_frame_masks(1) is None

    def test_object_ids(self) -> None:
        fm0 = FrameMasks(frame_idx=0, masks={1: np.ones((10, 10), dtype=bool)})
        fm1 = FrameMasks(
            frame_idx=1, masks={1: np.ones((10, 10), dtype=bool), 3: np.ones((10, 10), dtype=bool)}
        )
        vr = VideoResults(frames=[fm0, fm1])
        assert vr.object_ids == {1, 3}


class TestMaskEqualityAndRepr:
    """Regression: dataclass __eq__ raised on numpy array fields."""

    def test_equal_masks_compare_true(self) -> None:
        a = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        b = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        assert a == b

    def test_different_data_compare_false(self) -> None:
        a = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        b = Mask(data=np.zeros((4, 4), dtype=bool), score=0.9)
        assert a != b

    def test_different_score_compare_false(self) -> None:
        a = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        b = Mask(data=np.ones((4, 4), dtype=bool), score=0.5)
        assert a != b

    def test_repr_is_compact(self) -> None:
        m = Mask(data=np.ones((480, 640), dtype=bool), score=0.9)
        text = repr(m)
        assert "480" in text
        assert "640" in text
        assert "True" not in text  # must not dump the array

    def test_numpy_asarray(self) -> None:
        m = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        assert np.asarray(m).shape == (4, 4)


class TestMaskGeometryAndOperators:
    def test_bbox(self) -> None:
        data = np.zeros((10, 10), dtype=bool)
        data[2:5, 3:7] = True
        assert Mask(data=data, score=0.9).bbox == [3, 2, 6, 4]

    def test_bbox_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _ = Mask(data=np.zeros((4, 4), dtype=bool), score=0.9).bbox

    def test_centroid(self) -> None:
        data = np.zeros((10, 10), dtype=bool)
        data[2:4, 4:6] = True
        assert Mask(data=data, score=0.9).centroid == (4.5, 2.5)

    def test_iou(self) -> None:
        a = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        assert a.iou(a) == 1.0

    def test_iou_shape_mismatch_raises(self) -> None:
        a = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        b = Mask(data=np.ones((5, 5), dtype=bool), score=0.9)
        with pytest.raises(ValueError, match="different shapes"):
            a.iou(b)

    def test_and_or_invert(self) -> None:
        a = Mask(data=np.array([[True, False], [False, False]]), score=0.9)
        b = Mask(data=np.array([[False, True], [False, False]]), score=0.8)
        assert (a & b).area == 0
        assert (a | b).area == 2
        assert (~a).area == 3

    def test_combine_shape_mismatch_raises(self) -> None:
        a = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        b = Mask(data=np.ones((5, 5), dtype=bool), score=0.9)
        with pytest.raises(ValueError, match="different shapes"):
            _ = a & b


class TestMaskRleAndSave:
    def test_rle_roundtrip(self) -> None:
        data = np.zeros((8, 8), dtype=bool)
        data[2:6, 2:6] = True
        m = Mask(data=data, score=0.9)
        restored = Mask.from_rle(m.to_rle(), score=0.9)
        assert np.array_equal(restored.data, data)

    def test_dict_roundtrip(self) -> None:
        data = np.zeros((8, 8), dtype=bool)
        data[2:6, 2:6] = True
        m = Mask(data=data, score=0.9)
        restored = Mask.from_dict(m.to_dict())
        assert restored.score == 0.9
        assert np.array_equal(restored.data, data)

    def test_save_infers_npy_from_extension(self, tmp_path) -> None:
        m = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        out = m.save(tmp_path / "mask.npy")
        assert out.suffix == ".npy"
        assert np.load(out).shape == (4, 4)

    def test_save_infers_png_from_extension(self, tmp_path) -> None:
        m = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        out = m.save(tmp_path / "mask.png")
        assert out.exists()

    def test_save_infers_coco_rle_from_json(self, tmp_path) -> None:
        import json

        m = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        out = m.save(tmp_path / "mask.json")
        assert "counts" in json.loads(out.read_text())

    def test_save_rejects_unknown_extension(self, tmp_path) -> None:
        m = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        with pytest.raises(ValueError, match="Cannot infer"):
            m.save(tmp_path / "mask.tiff")

    def test_save_explicit_fmt_overrides_extension(self, tmp_path) -> None:
        m = Mask(data=np.ones((4, 4), dtype=bool), score=0.9)
        out = m.save(tmp_path / "mask.dat", fmt="npy")
        assert np.load(out).shape == (4, 4)


class TestImagePredictionIteration:
    def test_iter_and_index(self) -> None:
        masks = [Mask(data=np.ones((4, 4), dtype=bool), score=0.9)]
        pred = ImagePrediction(masks=masks)
        assert list(pred) == masks
        assert pred[0] is masks[0]

    def test_to_dict_roundtrip(self) -> None:
        data = np.zeros((6, 6), dtype=bool)
        data[1:4, 1:4] = True
        pred = ImagePrediction(masks=[Mask(data=data, score=0.9)], image_shape=(6, 6))
        restored = ImagePrediction.from_dict(pred.to_dict())
        assert restored.image_shape == (6, 6)
        assert np.array_equal(restored.masks[0].data, data)


class TestAutoMaskResultExtras:
    def _result(self) -> AutoMaskResult:
        def mk(area: int, score: float) -> AutoMask:
            data = np.zeros((10, 10), dtype=bool)
            data.flat[:area] = True
            return AutoMask(
                data=data,
                score=score,
                area=area,
                bbox=[0, 0, 1, 1],
                stability_score=score,
                crop_box=[0, 0, 10, 10],
            )

        return AutoMaskResult(masks=[mk(4, 0.5), mk(9, 0.9), mk(1, 0.7)])

    def test_slice_returns_result(self) -> None:
        sliced = self._result()[:2]
        assert isinstance(sliced, AutoMaskResult)
        assert len(sliced) == 2

    def test_sort_by_score(self) -> None:
        ordered = self._result().sort_by("score")
        assert [m.score for m in ordered] == [0.9, 0.7, 0.5]

    def test_sort_by_area_ascending(self) -> None:
        ordered = self._result().sort_by("area", descending=False)
        assert [m.area for m in ordered] == [1, 4, 9]

    def test_sort_by_invalid_key_raises(self) -> None:
        with pytest.raises(ValueError, match="sort_by key"):
            self._result().sort_by("nope")

    def test_to_dict_roundtrip(self) -> None:
        restored = AutoMaskResult.from_dict(self._result().to_dict())
        assert len(restored) == 3
        assert restored.masks[0].area == 4


class TestVideoResultsExtras:
    def _results(self) -> VideoResults:
        frames = [FrameMasks(frame_idx=i, masks={1: np.ones((4, 4), dtype=bool)}) for i in range(5)]
        return VideoResults(frames=frames, num_frames=5)

    def test_slice_returns_results(self) -> None:
        sliced = self._results()[1:3]
        assert isinstance(sliced, VideoResults)
        assert [fm.frame_idx for fm in sliced.values()] == [1, 2]

    def test_to_dict_roundtrip(self) -> None:
        restored = VideoResults.from_dict(self._results().to_dict())
        assert len(restored) == 5
        assert restored.num_frames == 5
        assert np.array_equal(restored[0].masks[1], np.ones((4, 4), dtype=bool))

    def test_frame_masks_equality(self) -> None:
        a = FrameMasks(frame_idx=0, masks={1: np.ones((4, 4), dtype=bool)})
        b = FrameMasks(frame_idx=0, masks={1: np.ones((4, 4), dtype=bool)})
        assert a == b
