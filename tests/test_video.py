"""Tests for the video tracking layer, using stub predictors (no SAM 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from lazysammy.types import FrameMasks
from lazysammy.video import VideoSession, VideoTracker


class _StubVideoPredictor:
    """Minimal stand-in for ``SAM2VideoPredictor``.

    Records the calls it receives and returns deterministic tensors so the
    session plumbing (validation, conversion, result assembly) can be tested
    without model weights.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.removed: list[int] = []
        self.cleared: list[tuple[int, int]] = []
        self.reset_count = 0
        self.obj_ids: list[int] = [1]

    # -- setup -----------------------------------------------------------
    def init_state(self, *, video_path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("init_state", {"video_path": video_path, **kwargs}))
        return {"video_path": video_path}

    # -- prompting -------------------------------------------------------
    def add_new_points_or_box(self, **kwargs: Any) -> tuple[int, list[int], torch.Tensor]:
        self.calls.append(("add_new_points_or_box", kwargs))
        if "obj_id" in kwargs:
            self.obj_ids = sorted({*self.obj_ids, int(kwargs["obj_id"])})
        masks = torch.ones((len(self.obj_ids), 1, 8, 8))
        return int(kwargs["frame_idx"]), list(self.obj_ids), masks

    def add_new_mask(self, **kwargs: Any) -> tuple[int, list[int], torch.Tensor]:
        self.calls.append(("add_new_mask", kwargs))
        if "obj_id" in kwargs:
            self.obj_ids = sorted({*self.obj_ids, int(kwargs["obj_id"])})
        masks = torch.ones((len(self.obj_ids), 1, 8, 8))
        return int(kwargs["frame_idx"]), list(self.obj_ids), masks

    # -- propagation -----------------------------------------------------
    def propagate_in_video(
        self,
        *,
        inference_state: Any,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Any:
        self.calls.append(
            (
                "propagate_in_video",
                {
                    "start": start_frame_idx,
                    "max": max_frame_num_to_track,
                    "reverse": reverse,
                },
            )
        )
        order = [1, 0] if reverse else [0, 1, 2]
        if start_frame_idx is not None:
            order = [
                f for f in order if (f >= start_frame_idx if not reverse else f <= start_frame_idx)
            ]
        if max_frame_num_to_track is not None:
            order = order[:max_frame_num_to_track]
        for idx in order:
            masks = torch.ones((len(self.obj_ids), 1, 8, 8))
            # Negate some pixels so binarisation is observable
            masks[:, :, :2, :2] = -1.0
            yield idx, list(self.obj_ids), masks

    # -- management ------------------------------------------------------
    def remove_object(self, state: Any, obj_id: int) -> None:
        self.removed.append(obj_id)
        self.obj_ids = [o for o in self.obj_ids if o != obj_id]

    def clear_all_prompts_in_frame(self, state: Any, frame_idx: int, obj_id: int) -> None:
        self.cleared.append((frame_idx, obj_id))

    def reset_state(self, state: Any) -> None:
        self.reset_count += 1
        self.obj_ids = []


def _make_session(num_frames: int = 3) -> tuple[VideoSession, _StubVideoPredictor]:
    predictor = _StubVideoPredictor()
    session = VideoSession(
        predictor=predictor,
        inference_state={},
        video_dir=Path("/tmp/frames"),
        num_frames=num_frames,
        frame_files=[Path(f"/tmp/frames/{i:05d}.jpg") for i in range(num_frames)],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return session, predictor


class TestAddPromptsValidation:
    def test_rejects_out_of_range_frame(self) -> None:
        session, _ = _make_session()
        with pytest.raises(IndexError, match="out of range"):
            session.add_points(frame_idx=99, obj_id=1, points=[[1, 2]], labels=[1])

    def test_rejects_mismatched_points_and_labels(self) -> None:
        session, _ = _make_session()
        with pytest.raises(ValueError, match="same length"):
            session.add_points(frame_idx=0, obj_id=1, points=[[1, 2]], labels=[1, 0])

    def test_rejects_missing_labels(self) -> None:
        session, _ = _make_session()
        with pytest.raises(ValueError, match="provided together"):
            session.add_points(frame_idx=0, obj_id=1, points=[[1, 2]], labels=None)  # type: ignore[arg-type]

    def test_rejects_invalid_box(self) -> None:
        session, _ = _make_session()
        with pytest.raises(ValueError, match="non-zero area"):
            session.add_box(frame_idx=0, obj_id=1, box=[5, 5, 5, 9])

    def test_add_mask_rejects_3d(self) -> None:
        session, _ = _make_session()
        with pytest.raises(ValueError, match="2D"):
            session.add_mask(frame_idx=0, obj_id=1, mask=np.ones((1, 4, 4), dtype=bool))


class TestPromptConversion:
    def test_add_points_returns_frame_masks(self) -> None:
        session, predictor = _make_session()
        fm = session.add_points(frame_idx=1, obj_id=2, points=[[3, 4]], labels=[1])

        assert isinstance(fm, FrameMasks)
        assert fm.frame_idx == 1
        assert fm.object_ids == [1, 2]
        call = dict(predictor.calls)["add_new_points_or_box"]
        assert call["frame_idx"] == 1
        assert call["obj_id"] == 2
        np.testing.assert_array_equal(call["points"], np.array([[3, 4]], dtype=np.float32))

    def test_add_box_normalizes_inverted_corners(self) -> None:
        session, predictor = _make_session()
        session.add_box(frame_idx=0, obj_id=1, box=[10, 20, 1, 2])

        call = dict(predictor.calls)["add_new_points_or_box"]
        np.testing.assert_array_equal(call["box"], np.array([1, 2, 10, 20], dtype=np.float32))

    def test_masks_are_binarised_by_threshold(self) -> None:
        session, _ = _make_session()
        session.add_points(frame_idx=0, obj_id=1, points=[[1, 1]], labels=[1])
        results = session.propagate()

        # The propagation stub sets the top-left 2x2 patch to -1.0, which is
        # below the 0.0 logit threshold and must therefore become False.
        frame = results[0]
        assert frame.masks[1][:2, :2].sum() == 0
        assert frame.masks[1][2:, 2:].all()
        assert frame.masks[1].dtype == bool


class TestPropagation:
    def test_forward_propagation_builds_results(self) -> None:
        session, _ = _make_session()
        session.add_points(frame_idx=0, obj_id=1, points=[[1, 1]], labels=[1])
        results = session.propagate()

        assert isinstance(results, type(session.results))
        assert [idx for idx, _ in results] == [0, 1, 2]
        assert results.num_frames == 3
        assert results.video_dir == Path("/tmp/frames")
        assert session.results is results

    def test_reverse_flag_is_forwarded(self) -> None:
        session, predictor = _make_session()
        session.add_points(frame_idx=0, obj_id=1, points=[[1, 1]], labels=[1])
        session.propagate(reverse=True)

        call = dict(predictor.calls)["propagate_in_video"]
        assert call["reverse"] is True

    def test_max_frames_limits_results(self) -> None:
        session, _ = _make_session()
        session.add_points(frame_idx=0, obj_id=1, points=[[1, 1]], labels=[1])
        results = session.propagate(max_frames=1)
        assert len(results) == 1

    def test_rejects_invalid_start_frame(self) -> None:
        session, _ = _make_session()
        with pytest.raises(IndexError, match="out of range"):
            session.propagate(start_frame=50)

    def test_rejects_non_positive_max_frames(self) -> None:
        session, _ = _make_session()
        with pytest.raises(ValueError, match="max_frames"):
            session.propagate(max_frames=0)

    def test_bidirectional_merges_forward_and_backward(self) -> None:
        session, predictor = _make_session()
        session.add_points(frame_idx=1, obj_id=1, points=[[1, 1]], labels=[1])
        merged = session.propagate_bidirectional(start_frame=1)

        calls = [name for name, _ in predictor.calls].count("propagate_in_video")
        assert calls == 2
        assert [idx for idx, _ in merged] == sorted(idx for idx, _ in merged)
        assert 0 in [idx for idx, _ in merged]
        assert session.results is merged


class TestObjectManagement:
    def test_remove_object_delegates(self) -> None:
        session, predictor = _make_session()
        session.remove_object(2)
        assert predictor.removed == [2]

    def test_clear_frame_prompts_validates_frame(self) -> None:
        session, predictor = _make_session()
        session.clear_frame_prompts(frame_idx=2, obj_id=1)
        assert predictor.cleared == [(2, 1)]

        with pytest.raises(IndexError, match="out of range"):
            session.clear_frame_prompts(frame_idx=9, obj_id=1)

    def test_reset_clears_results(self) -> None:
        session, predictor = _make_session()
        session.add_points(frame_idx=0, obj_id=1, points=[[1, 1]], labels=[1])
        session.propagate()
        assert session.results is not None

        session.reset()
        assert predictor.reset_count == 1
        assert session.results is None


class TestSaveGuards:
    def test_save_without_results_raises(self, tmp_path: Path) -> None:
        session, _ = _make_session()
        with pytest.raises(RuntimeError, match="Call propagate"):
            session.save(tmp_path / "out")

    def test_save_overlay_without_results_raises(self, tmp_path: Path) -> None:
        session, _ = _make_session()
        with pytest.raises(RuntimeError, match="Call propagate"):
            session.save_overlay(tmp_path / "out.mp4")


class TestProperties:
    def test_num_frames_and_video_dir(self) -> None:
        session, _ = _make_session(num_frames=7)
        assert session.num_frames == 7
        assert session.video_dir == Path("/tmp/frames")

    def test_results_is_none_before_propagation(self) -> None:
        session, _ = _make_session()
        assert session.results is None


class TestVideoTrackerDevice:
    def test_autocast_dtype_for_cpu(self) -> None:
        tracker = VideoTracker.__new__(VideoTracker)
        tracker._device = torch.device("cpu")
        tracker._dtype = torch.float32
        assert tracker.device == torch.device("cpu")
