"""Tests for the SAM2 facade - no SAM2 model required."""

from __future__ import annotations

from lazysammy.sam2 import SAM2


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
