"""Tests for the SAM 3 integration - no SAM 3 model required.

The ``sam3`` package is an optional dependency, so these tests stub the model
builders and the ``Sam3Processor`` to assert the wrapper's behavior (prompt
plumbing, output conversion, delegation) without loading weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from lazysammy.concept import ConceptSegmenter
from lazysammy.sam3 import SAM3
from lazysammy.sam3_video import SAM3VideoSession, SAM3VideoTracker
from lazysammy.types import ConceptPrediction, Mask
from lazysammy.utils import load_image

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubProcessor:
    """Minimal stand-in for ``Sam3Processor``."""

    def __init__(self, model: Any, **kwargs: Any) -> None:
        self.model = model
        self.confidence_threshold = kwargs.get("confidence_threshold", 0.5)
        self.calls: list[tuple[str, Any]] = []

    def set_image(self, image: np.ndarray) -> dict[str, Any]:
        self.calls.append(("set_image", image.shape))
        return {"original_height": image.shape[0], "original_width": image.shape[1]}

    def set_image_batch(self, images: list[np.ndarray]) -> dict[str, Any]:
        self.calls.append(("set_image_batch", len(images)))
        return {
            "original_heights": [im.shape[0] for im in images],
            "original_widths": [im.shape[1] for im in images],
        }

    def set_text_prompt(self, *, state: dict[str, Any], prompt: str) -> dict[str, Any]:
        self.calls.append(("set_text_prompt", prompt))
        return self._with_detections(state, concept=prompt)

    def add_geometric_prompt(
        self, *, box: list[float], label: bool, state: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(("add_geometric_prompt", {"box": box, "label": label}))
        return self._with_detections(state, concept="visual")

    def set_confidence_threshold(
        self, threshold: float, state: dict[str, Any] | None = None
    ) -> None:
        self.calls.append(("set_confidence_threshold", threshold))
        self.confidence_threshold = threshold

    def reset_all_prompts(self, state: dict[str, Any]) -> None:
        self.calls.append(("reset_all_prompts", None))

    @staticmethod
    def _with_detections(state: dict[str, Any], *, concept: str) -> dict[str, Any]:
        masks = np.zeros((2, 1, 8, 8), dtype=bool)
        masks[0, 0, 1:3, 1:3] = True
        masks[1, 0, 4:6, 4:6] = True
        state = dict(state)
        state.update(
            {
                "masks": masks,
                "scores": np.array([0.9, 0.6], dtype=np.float32),
                "boxes": np.array([[1, 1, 3, 3], [4, 4, 6, 6]], dtype=np.float32),
                "concept": concept,
            }
        )
        return state


class _StubImageModel:
    """Minimal stand-in for ``Sam3Image`` (interactive path)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def predict_inst(self, state: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, None]:
        self.calls.append(("predict_inst", kwargs))
        masks = np.ones((1, 8, 8), dtype=bool)
        scores = np.array([0.8], dtype=np.float32)
        return masks, scores, None

    def predict_inst_batch(
        self, state: Any, points_batch: Any, labels_batch: Any, **kwargs: Any
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[None]]:
        self.calls.append(("predict_inst_batch", kwargs))
        batch_size = len(state["original_heights"])
        masks = [np.ones((1, 8, 8), dtype=bool) for _ in range(batch_size)]
        scores = [np.array([0.8], dtype=np.float32) for _ in range(batch_size)]
        return masks, scores, [None] * batch_size


@pytest.fixture
def stub_concept_segmenter(monkeypatch: pytest.MonkeyPatch) -> ConceptSegmenter:
    """A ConceptSegmenter whose model and processor are stubbed."""
    monkeypatch.setattr(
        "lazysammy.models.load_sam3_image_model", lambda **kwargs: _StubImageModel()
    )
    monkeypatch.setattr(
        ConceptSegmenter,
        "_make_processor",
        lambda self, confidence_threshold: _StubProcessor(
            self._model, confidence_threshold=confidence_threshold
        ),
    )
    return ConceptSegmenter(device="cpu")


# ---------------------------------------------------------------------------
# ConceptPrediction type
# ---------------------------------------------------------------------------


class TestConceptPrediction:
    def test_iteration_and_indexing(self) -> None:
        pred = ConceptPrediction(
            masks=[
                Mask(data=np.ones((4, 4), dtype=bool), score=0.5),
                Mask(data=np.zeros((4, 4), dtype=bool), score=0.9),
            ],
            boxes=[[0, 0, 4, 4], [1, 1, 2, 2]],
            concept="cat",
        )
        assert len(pred) == 2
        assert pred[0].score == 0.5
        assert [m.score for m in pred] == [0.5, 0.9]
        assert pred.best_mask.score == 0.9

    def test_best_mask_raises_when_empty(self) -> None:
        with pytest.raises(ValueError, match="does not contain any masks"):
            _ = ConceptPrediction(masks=[]).best_mask

    def test_numpy_stacks_masks(self) -> None:
        pred = ConceptPrediction(
            masks=[
                Mask(data=np.ones((4, 4), dtype=bool), score=0.5),
                Mask(data=np.zeros((4, 4), dtype=bool), score=0.9),
            ]
        )
        assert pred.numpy().shape == (2, 4, 4)

    def test_filter_by_score_keeps_boxes_aligned(self) -> None:
        pred = ConceptPrediction(
            masks=[
                Mask(data=np.ones((4, 4), dtype=bool), score=0.5),
                Mask(data=np.zeros((4, 4), dtype=bool), score=0.9),
            ],
            boxes=[[0, 0, 4, 4], [1, 1, 2, 2]],
            concept="cat",
        )
        filtered = pred.filter_by_score(0.7)
        assert len(filtered) == 1
        assert filtered.boxes == [[1, 1, 2, 2]]
        assert filtered.concept == "cat"

    def test_filter_by_score_without_boxes_keeps_masks(self) -> None:
        """Regression: zipping masks against empty boxes dropped every mask."""
        pred = ConceptPrediction(
            masks=[
                Mask(data=np.ones((4, 4), dtype=bool), score=0.5),
                Mask(data=np.zeros((4, 4), dtype=bool), score=0.9),
            ],
            concept="cat",
        )
        filtered = pred.filter_by_score(0.7)
        assert len(filtered) == 1
        assert filtered.masks[0].score == 0.9
        assert filtered.boxes == []

    def test_dict_round_trip(self) -> None:
        pred = ConceptPrediction(
            masks=[Mask(data=np.ones((4, 4), dtype=bool), score=0.5)],
            boxes=[[0, 0, 4, 4]],
            concept="cat",
            image_shape=(4, 4),
        )
        restored = ConceptPrediction.from_dict(pred.to_dict())
        assert restored.concept == "cat"
        assert restored.boxes == [[0, 0, 4, 4]]
        assert restored.image_shape == (4, 4)
        assert restored.masks[0].score == 0.5


# ---------------------------------------------------------------------------
# ConceptSegmenter
# ---------------------------------------------------------------------------


class TestConceptSegmenter:
    def test_segment_text_returns_all_instances(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        pred = stub_concept_segmenter.segment_text(image, "a player in white")
        assert isinstance(pred, ConceptPrediction)
        assert len(pred) == 2
        assert pred.concept == "a player in white"
        assert pred.boxes == [[1.0, 1.0, 3.0, 3.0], [4.0, 4.0, 6.0, 6.0]]
        assert pred.image_shape == (8, 8)

    def test_segment_text_reuses_cached_image(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        stub_concept_segmenter.segment_text(image, "cat")
        stub_concept_segmenter.segment_text(image, "dog")
        set_image_calls = [c for c in stub_concept_segmenter.processor.calls if c[0] == "set_image"]
        assert len(set_image_calls) == 1

    def test_segment_exemplar_normalizes_box(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        pred = stub_concept_segmenter.segment_exemplar(image, [0, 0, 4, 4])
        assert pred.concept == "visual"
        call = next(
            c for c in stub_concept_segmenter.processor.calls if c[0] == "add_geometric_prompt"
        )
        # [x1,y1,x2,y2] = [0,0,4,4] on an 8x8 image -> cxcywh [0.25, 0.25, 0.5, 0.5]
        assert call[1]["box"] == pytest.approx([0.25, 0.25, 0.5, 0.5])

    def test_segment_with_text_delegates_to_concept(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        pred = stub_concept_segmenter.segment(image, text="cat")
        assert pred.concept == "cat"

    def test_segment_with_points_uses_interactive_predictor(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        pred = stub_concept_segmenter.segment(image, points=[[1, 1]], labels=[1])
        assert len(pred) == 1
        assert pred.masks[0].score == pytest.approx(0.8)
        assert stub_concept_segmenter.model.calls[0][0] == "predict_inst"

    def test_predict_without_set_image_raises(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        with pytest.raises(RuntimeError, match="set_image"):
            stub_concept_segmenter.predict(points=[[1, 1]], labels=[1])

    def test_set_image_then_predict(self, stub_concept_segmenter: ConceptSegmenter) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        assert stub_concept_segmenter.set_image(image) == (8, 8)
        pred = stub_concept_segmenter.predict(points=[[1, 1]], labels=[1])
        assert len(pred) == 1

    def test_confidence_threshold_override(self, stub_concept_segmenter: ConceptSegmenter) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        stub_concept_segmenter.segment_text(image, "cat", confidence_threshold=0.8)
        assert ("set_confidence_threshold", 0.8) in stub_concept_segmenter.processor.calls


# ---------------------------------------------------------------------------
# SAM 2 image-feature parity
# ---------------------------------------------------------------------------


class TestSAM2Parity:
    """SAM 3 exposes every SAM 2 image feature, through the shared mixin."""

    def test_segment_point_runs_interactive_predictor(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        pred = stub_concept_segmenter.segment_point(image, 4, 4)
        assert len(pred) == 1
        assert pred.masks[0].score == pytest.approx(0.8)

    def test_segment_box_passes_single_box(self, stub_concept_segmenter: ConceptSegmenter) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        stub_concept_segmenter.segment_box(image, 0, 0, 4, 4)
        call = next(c for c in stub_concept_segmenter.model.calls if c[0] == "predict_inst")
        assert call[1]["box"].shape == (1, 4)

    def test_segment_multi_box_returns_one_per_box(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        preds = stub_concept_segmenter.segment_multi_box(image, [[0, 0, 4, 4], [1, 1, 5, 5]])
        assert len(preds) == 2
        assert all(isinstance(p, ConceptPrediction) for p in preds)

    def test_segment_multi_box_empty(self, stub_concept_segmenter: ConceptSegmenter) -> None:
        assert stub_concept_segmenter.segment_multi_box("photo.jpg", []) == []

    def test_segment_multi_point_returns_one_per_object(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        preds = stub_concept_segmenter.segment_multi_point(image, [[[1, 1]], [[2, 2]]])
        assert len(preds) == 2

    def test_segment_multi_point_rejects_mismatched_labels(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="same length"):
            stub_concept_segmenter.segment_multi_point(
                image, [[[1, 1]], [[2, 2]]], labels_per_object=[[1]]
            )

    def test_refine_uses_previous_logits(self, stub_concept_segmenter: ConceptSegmenter) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        pred = stub_concept_segmenter.refine(image, np.zeros((16, 16), dtype=np.float32))
        assert len(pred) == 1
        call = next(c for c in stub_concept_segmenter.model.calls if c[0] == "predict_inst")
        assert call[1]["mask_input"].shape == (1, 16, 16)

    def test_refine_rejects_bad_ndim(self, stub_concept_segmenter: ConceptSegmenter) -> None:
        with pytest.raises(ValueError, match="mask-logit"):
            stub_concept_segmenter.refine("photo.jpg", np.zeros((1, 2, 3, 4), dtype=np.float32))

    def test_segment_batch_encodes_all_images(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
        preds = stub_concept_segmenter.segment_batch(images, box_batch=[[0, 0, 4, 4], None])
        assert len(preds) == 2
        assert ("set_image_batch", 2) in stub_concept_segmenter.processor.calls
        assert stub_concept_segmenter.model.calls[0][0] == "predict_inst_batch"

    def test_segment_batch_rejects_length_mismatch(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
        with pytest.raises(ValueError, match="length 2"):
            stub_concept_segmenter.segment_batch(images, box_batch=[[0, 0, 4, 4]])

    def test_to_image_prediction_bridges_result_types(
        self, stub_concept_segmenter: ConceptSegmenter
    ) -> None:
        concept = ConceptPrediction(
            masks=[Mask(data=np.ones((4, 4), dtype=bool), score=0.5)],
            boxes=[[0, 0, 4, 4]],
            concept="cat",
            image_shape=(4, 4),
        )
        image_pred = stub_concept_segmenter.to_image_prediction(concept)
        assert image_pred.image_shape == (4, 4)
        assert len(image_pred) == 1
        assert image_pred.best_mask.score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# SAM3 facade
# ---------------------------------------------------------------------------


class _StubConceptSegmenter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.device = torch.device("cpu")

    def segment_text(self, image: Any, text: str, **kwargs: Any) -> ConceptPrediction:
        self.calls.append(("segment_text", {"text": text, **kwargs}))
        return ConceptPrediction(masks=[], concept=text)

    def segment_exemplar(self, image: Any, box: Any, **kwargs: Any) -> ConceptPrediction:
        self.calls.append(("segment_exemplar", {"box": box, **kwargs}))
        return ConceptPrediction(masks=[], concept="visual")

    def segment(self, image: Any, **kwargs: Any) -> ConceptPrediction:
        self.calls.append(("segment", kwargs))
        return ConceptPrediction(masks=[])

    def set_image(self, image: Any) -> tuple[int, int]:
        self.calls.append(("set_image", {}))
        return (4, 4)

    def predict(self, **kwargs: Any) -> ConceptPrediction:
        self.calls.append(("predict", kwargs))
        return ConceptPrediction(masks=[])

    def segment_point(self, image: Any, x: float, y: float, **kwargs: Any) -> ConceptPrediction:
        self.calls.append(("segment_point", {"x": x, "y": y, **kwargs}))
        return ConceptPrediction(masks=[])

    def segment_box(self, image: Any, x1: float, y1: float, x2: float, y2: float) -> Any:
        self.calls.append(("segment_box", {"box": [x1, y1, x2, y2]}))
        return ConceptPrediction(masks=[])

    def segment_multi_box(self, image: Any, boxes: Any) -> list[ConceptPrediction]:
        self.calls.append(("segment_multi_box", {"n": len(boxes)}))
        return []

    def segment_multi_point(self, image: Any, points_per_object: Any, **kwargs: Any) -> Any:
        self.calls.append(("segment_multi_point", {"n": len(points_per_object)}))
        return []

    def refine(self, image: Any, previous_logits: Any, **kwargs: Any) -> ConceptPrediction:
        self.calls.append(("refine", kwargs))
        return ConceptPrediction(masks=[])

    def segment_batch(self, images: Any, **kwargs: Any) -> list[ConceptPrediction]:
        self.calls.append(("segment_batch", {"n": len(images)}))
        return []

    def to_image_prediction(self, prediction: Any) -> Any:
        self.calls.append(("to_image_prediction", {}))
        return prediction

    def save(self, result: Any, output_dir: Any, **kwargs: Any) -> Path:
        self.calls.append(("save", kwargs))
        return Path(output_dir)


class _StubVideoTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.device = torch.device("cpu")

    def new_session(self, video: Any, **kwargs: Any) -> str:
        self.calls.append(("new_session", {"video": video, **kwargs}))
        return "session"


def _stubbed_sam(monkeypatch: pytest.MonkeyPatch) -> SAM3:
    sam = SAM3(device="cpu")
    sam._concept_segmenter = _StubConceptSegmenter()  # type: ignore[assignment]
    sam._video_tracker = _StubVideoTracker()  # type: ignore[assignment]
    return sam


class TestSAM3Facade:
    def test_segment_text_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sam = _stubbed_sam(monkeypatch)
        pred = sam.segment_text("photo.jpg", "cat", confidence_threshold=0.7)
        assert pred.concept == "cat"
        assert sam._concept_segmenter.calls[0][0] == "segment_text"  # type: ignore[union-attr]

    def test_segment_exemplar_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sam = _stubbed_sam(monkeypatch)
        sam.segment_exemplar("photo.jpg", [0, 0, 4, 4], label=False)
        _, kwargs = sam._concept_segmenter.calls[0]  # type: ignore[union-attr]
        assert kwargs["label"] is False

    def test_segment_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sam = _stubbed_sam(monkeypatch)
        sam.segment("photo.jpg", points=[[1, 1]], labels=[1])
        assert sam._concept_segmenter.calls[0][0] == "segment"  # type: ignore[union-attr]

    def test_set_image_and_predict_delegate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sam = _stubbed_sam(monkeypatch)
        assert sam.set_image("photo.jpg") == (4, 4)
        sam.predict(points=[[1, 1]], labels=[1])
        names = [c[0] for c in sam._concept_segmenter.calls]  # type: ignore[union-attr]
        assert names == ["set_image", "predict"]

    def test_segment_point_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sam = _stubbed_sam(monkeypatch)
        sam.segment_point("photo.jpg", 4, 5, foreground=False)
        name, kwargs = sam._concept_segmenter.calls[0]  # type: ignore[union-attr]
        assert name == "segment_point"
        assert kwargs["x"] == 4
        assert kwargs["y"] == 5
        assert kwargs["foreground"] is False

    def test_segment_box_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sam = _stubbed_sam(monkeypatch)
        sam.segment_box("photo.jpg", 0, 0, 4, 4)
        name, kwargs = sam._concept_segmenter.calls[0]  # type: ignore[union-attr]
        assert name == "segment_box"
        assert kwargs["box"] == [0, 0, 4, 4]

    def test_multi_helpers_delegate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sam = _stubbed_sam(monkeypatch)
        sam.segment_multi_box("photo.jpg", [[0, 0, 4, 4], [1, 1, 5, 5]])
        sam.segment_multi_point("photo.jpg", [[[1, 1]], [[2, 2]]])
        names = [c[0] for c in sam._concept_segmenter.calls]  # type: ignore[union-attr]
        assert names == ["segment_multi_box", "segment_multi_point"]

    def test_refine_and_batch_delegate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sam = _stubbed_sam(monkeypatch)
        sam.refine("photo.jpg", np.zeros((16, 16), dtype=np.float32))
        sam.segment_batch(["a.jpg", "b.jpg"])
        names = [c[0] for c in sam._concept_segmenter.calls]  # type: ignore[union-attr]
        assert names == ["refine", "segment_batch"]

    def test_save_accepts_image_prediction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lazysammy.types import ImagePrediction

        sam = _stubbed_sam(monkeypatch)
        out = sam.save(ImagePrediction(masks=[]), "out/", fmt="npy")
        assert str(out).endswith("out")
        # Image predictions are saved without touching the concept segmenter.
        assert sam._concept_segmenter.calls == []  # type: ignore[union-attr]

    def test_video_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sam = _stubbed_sam(monkeypatch)
        session = sam.video("clip.mp4", every_n=2)
        assert session == "session"
        _, kwargs = sam._video_tracker.calls[0]  # type: ignore[union-attr]
        assert kwargs["every_n"] == 2

    def test_resolved_device_uses_subcomponent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sam = _stubbed_sam(monkeypatch)
        assert sam.resolved_device == torch.device("cpu")

    def test_lazy_components_start_none(self) -> None:
        sam = SAM3(device="cpu")
        assert sam._concept_segmenter is None
        assert sam._video_tracker is None


# ---------------------------------------------------------------------------
# SAM3VideoSession
# ---------------------------------------------------------------------------


class _StubVideoPredictor:
    """Records requests and returns deterministic per-frame outputs."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "sess-1"}
        return {"frame_index": request.get("frame_index", 0), "outputs": self._outputs()}

    def handle_stream_request(self, request: dict[str, Any]) -> Any:
        self.requests.append(request)
        for frame_idx in range(3):
            yield {"frame_index": frame_idx, "outputs": self._outputs()}

    @staticmethod
    def _outputs() -> dict[str, Any]:
        masks = np.zeros((2, 1, 8, 8), dtype=bool)
        masks[0, 0, 1:3, 1:3] = True
        masks[1, 0, 4:6, 4:6] = True
        return {
            "out_obj_ids": np.array([1, 2], dtype=np.int64),
            "out_binary_masks": masks,
        }


def _write_frames(tmp_path: Path, count: int, shape: tuple[int, int] = (8, 8)) -> list[Path]:
    """Write *count* real JPEG frames so coordinate normalization can read them."""
    import cv2

    frames = []
    for i in range(count):
        frame = tmp_path / f"{i:05d}.jpg"
        cv2.imwrite(str(frame), np.zeros((*shape, 3), dtype=np.uint8))
        frames.append(frame)
    return frames


def _session(tmp_path: Path) -> tuple[SAM3VideoSession, _StubVideoPredictor]:
    predictor = _StubVideoPredictor()
    frames = _write_frames(tmp_path, 3)
    session = SAM3VideoSession(
        predictor=predictor,
        session_id="sess-1",
        video_dir=tmp_path,
        num_frames=3,
        frame_files=frames,
        device=torch.device("cpu"),
    )
    return session, predictor


class TestSAM3VideoSession:
    def test_add_text_sends_text_prompt(self, tmp_path: Path) -> None:
        session, predictor = _session(tmp_path)
        frame = session.add_text(frame_idx=0, text="person")
        assert frame.object_ids == [1, 2]
        assert frame.masks[1].shape == (8, 8)
        request = predictor.requests[0]
        assert request["type"] == "add_prompt"
        assert request["text"] == "person"

    def test_add_points_converts_to_relative(self, tmp_path: Path) -> None:
        session, predictor = _session(tmp_path)
        session.add_points(frame_idx=0, obj_id=1, points=[[4, 4]], labels=[1])
        request = predictor.requests[0]
        # 8x8 frame -> (4, 4) becomes (0.5, 0.5)
        np.testing.assert_allclose(request["points"], [[0.5, 0.5]])
        assert request["obj_id"] == 1

    def test_add_box_converts_to_relative_xywh(self, tmp_path: Path) -> None:
        session, predictor = _session(tmp_path)
        session.add_box(frame_idx=0, obj_id=1, box=[0, 0, 4, 4])
        request = predictor.requests[0]
        np.testing.assert_allclose(request["bounding_boxes"], [[0.0, 0.0, 0.5, 0.5]])

    def test_frame_shape_is_read_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: every prompt re-read the first frame from disk."""
        session, _ = _session(tmp_path)
        calls = {"n": 0}
        real_load = load_image

        def _counting_load(source: Any) -> Any:
            calls["n"] += 1
            return real_load(source)

        monkeypatch.setattr("lazysammy.sam3_video.load_image", _counting_load)
        session.add_points(frame_idx=0, obj_id=1, points=[[4, 4]], labels=[1])
        session.add_box(frame_idx=0, obj_id=1, box=[0, 0, 4, 4])
        assert calls["n"] == 1

    def test_propagate_collects_frames(self, tmp_path: Path) -> None:
        session, predictor = _session(tmp_path)
        results = session.propagate(direction="forward")
        assert len(results) == 3
        assert results.object_ids == {1, 2}
        assert predictor.requests[0]["propagation_direction"] == "forward"

    def test_propagate_rejects_bad_direction(self, tmp_path: Path) -> None:
        session, _ = _session(tmp_path)
        with pytest.raises(ValueError, match="direction"):
            session.propagate(direction="sideways")

    def test_remove_object_sends_request(self, tmp_path: Path) -> None:
        session, predictor = _session(tmp_path)
        session.remove_object(2)
        assert predictor.requests[0]["type"] == "remove_object"
        assert predictor.requests[0]["obj_id"] == 2

    def test_reset_and_close(self, tmp_path: Path) -> None:
        session, predictor = _session(tmp_path)
        session.reset()
        session.close()
        session.close()  # idempotent
        types = [r["type"] for r in predictor.requests]
        assert types == ["reset_session", "close_session"]

    def test_save_without_results_raises(self, tmp_path: Path) -> None:
        session, _ = _session(tmp_path)
        with pytest.raises(RuntimeError, match="propagate"):
            session.save(tmp_path / "out")


class TestSAM3VideoTracker:
    def test_new_session_starts_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        predictor = _StubVideoPredictor()
        monkeypatch.setattr(
            "lazysammy.models.load_sam3_video_predictor", lambda **kwargs: predictor
        )
        _write_frames(tmp_path, 2)

        tracker = SAM3VideoTracker(device="cpu")
        session = tracker.new_session(tmp_path)
        assert isinstance(session, SAM3VideoSession)
        assert session.num_frames == 2
        assert predictor.requests[0]["type"] == "start_session"
