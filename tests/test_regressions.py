"""Regression tests for previously-fixed bugs.

Each test here maps to a specific bug that shipped (or nearly shipped) in the
project. Keep them even if the surrounding implementation is refactored: they
encode the *contract*, not the implementation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

from lazysammy.utils import (
    autocast,
    extract_frames,
    get_autocast_dtype,
    list_frame_files,
    natural_sort_key,
)


def _touch(path: Path) -> None:
    path.write_bytes(b"x")


@contextmanager
def _fail_on_warning() -> Iterator[None]:
    """Fail the test if any warning is emitted inside the block."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


class TestNaturalFrameOrdering:
    """Regression: frames were ordered lexicographically (1, 10, 100, 11, 2)."""

    def test_unpadded_frames_sort_numerically(self, tmp_path: Path) -> None:
        for name in ["0.jpg", "1.jpg", "2.jpg", "10.jpg", "11.jpg", "100.jpg"]:
            _touch(tmp_path / name)

        names = [f.name for f in list_frame_files(tmp_path)]

        assert names == ["0.jpg", "1.jpg", "2.jpg", "10.jpg", "11.jpg", "100.jpg"]

    def test_mixed_prefix_frames_sort_naturally(self, tmp_path: Path) -> None:
        for name in ["frame_1.png", "frame_2.png", "frame_10.png", "frame_20.png"]:
            _touch(tmp_path / name)

        names = [f.name for f in list_frame_files(tmp_path)]

        assert names == ["frame_1.png", "frame_2.png", "frame_10.png", "frame_20.png"]

    def test_zero_padded_frames_still_sort_numerically(self, tmp_path: Path) -> None:
        for name in ["00000.jpg", "00001.jpg", "00009.jpg", "00010.jpg"]:
            _touch(tmp_path / name)

        names = [f.name for f in list_frame_files(tmp_path)]

        assert names == ["00000.jpg", "00001.jpg", "00009.jpg", "00010.jpg"]

    def test_natural_sort_key_is_deterministic(self) -> None:
        assert natural_sort_key(Path("2.jpg")) < natural_sort_key(Path("10.jpg"))
        assert natural_sort_key(Path("10.jpg")) < natural_sort_key(Path("100.jpg"))


class TestAutocastCPU:
    """Regression: CPU runs emitted a warning on every autocast call."""

    def test_cpu_dtype_is_float32(self) -> None:
        assert get_autocast_dtype(torch.device("cpu")) is torch.float32

    def test_cpu_autocast_does_not_warn(self) -> None:
        with _fail_on_warning(), autocast(torch.device("cpu")):
            _ = torch.ones(2)

    def test_cpu_autocast_is_usable(self) -> None:
        with autocast(torch.device("cpu")):
            result = torch.ones(3) * 2
        assert torch.equal(result, torch.full((3,), 2.0))

    def test_cuda_dtype_selection(self) -> None:
        assert get_autocast_dtype(torch.device("cuda")) is torch.bfloat16

    def test_mps_dtype_selection(self) -> None:
        assert get_autocast_dtype(torch.device("mps")) is torch.float16


class TestExtractFramesValidation:
    """Regression: max_frames=0 silently produced one frame."""

    def test_rejects_zero_every_n(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="every_n"):
            extract_frames(tmp_path / "missing.mp4", every_n=0)

    def test_rejects_zero_max_frames(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_frames"):
            extract_frames(tmp_path / "missing.mp4", max_frames=0)


class TestMaskInputTyping:
    """Regression: MaskInput was documented as uint8 binary but used for logits."""

    def test_mask_logits_alias_is_floating(self) -> None:
        from lazysammy.types import MaskLogits

        arr: MaskLogits = np.zeros((1, 256, 256), dtype=np.float32)
        assert arr.dtype == np.float32

    def test_mask_input_alias_accepts_bool_and_uint8(self) -> None:
        from lazysammy.types import MaskInput

        bool_mask: MaskInput = np.zeros((4, 4), dtype=bool)
        uint8_mask: MaskInput = np.zeros((4, 4), dtype=np.uint8)
        assert bool_mask.dtype == bool
        assert uint8_mask.dtype == np.uint8


class TestVideoResultsAccess:
    """Regression: __getitem__ raised KeyError instead of IndexError."""

    def _results(self):
        from lazysammy.types import FrameMasks, VideoResults

        return VideoResults(
            frames=[
                FrameMasks(frame_idx=0, masks={1: np.ones((4, 4), dtype=bool)}),
                FrameMasks(frame_idx=5, masks={1: np.ones((4, 4), dtype=bool)}),
            ]
        )

    def test_missing_frame_raises_index_error(self) -> None:
        results = self._results()
        with pytest.raises(IndexError, match="not present"):
            _ = results[99]

    def test_values_are_sorted(self) -> None:
        results = self._results()
        assert [fm.frame_idx for fm in results.values()] == [0, 5]

    def test_iteration_yields_pairs(self) -> None:
        results = self._results()
        pairs = list(results)
        assert [idx for idx, _ in pairs] == [0, 5]
