"""Tests for validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from lazysammy.validation import (
    normalize_box,
    validate_box,
    validate_points_and_labels,
    validate_save_format,
    validate_segment_prompts,
)


class TestValidatePointsAndLabels:
    def test_accepts_matching_points_and_labels(self) -> None:
        validate_points_and_labels([[1, 2], [3, 4]], [1, 0])

    def test_rejects_missing_labels(self) -> None:
        with pytest.raises(ValueError, match="provided together"):
            validate_points_and_labels([[1, 2]], None)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            validate_points_and_labels([[1, 2], [3, 4]], [1])


class TestValidateBox:
    def test_accepts_valid_box(self) -> None:
        validate_box([1, 2, 10, 20])

    def test_rejects_zero_area_box(self) -> None:
        with pytest.raises(ValueError, match="non-zero area"):
            validate_box([1, 2, 1, 20])


class TestNormalizeBox:
    def test_keeps_canonical_box(self) -> None:
        normalized = normalize_box([1, 2, 10, 20])
        np.testing.assert_array_equal(normalized, np.array([1, 2, 10, 20], dtype=np.float32))

    def test_normalizes_inverted_box(self) -> None:
        normalized = normalize_box([10, 20, 1, 2])
        np.testing.assert_array_equal(normalized, np.array([1, 2, 10, 20], dtype=np.float32))


class TestValidateSegmentPrompts:
    def test_rejects_missing_all_prompts(self) -> None:
        with pytest.raises(ValueError, match="At least one prompt"):
            validate_segment_prompts(points=None, labels=None, box=None, mask_input=None)

    def test_accepts_mask_input_only(self) -> None:
        mask = np.ones((4, 4), dtype=np.float32)
        validate_segment_prompts(points=None, labels=None, box=None, mask_input=mask)


class TestValidateSaveFormat:
    def test_normalizes_valid_format(self) -> None:
        assert validate_save_format("PNG") == "png"

    def test_rejects_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported save format"):
            validate_save_format("jpeg")
