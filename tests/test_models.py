"""Tests for the model loading module – no SAM2 model required."""

from __future__ import annotations

import pytest

from lazysammy.models import resolve_model_size
from lazysammy.types import ModelSize


class TestResolveModelSize:
    """Model size resolution tests."""

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("tiny", ModelSize.TINY),
            ("t", ModelSize.TINY),
            ("small", ModelSize.SMALL),
            ("s", ModelSize.SMALL),
            ("base_plus", ModelSize.BASE_PLUS),
            ("b+", ModelSize.BASE_PLUS),
            ("large", ModelSize.LARGE),
            ("l", ModelSize.LARGE),
            ("LARGE", ModelSize.LARGE),
            ("Tiny", ModelSize.TINY),
        ],
    )
    def test_valid_aliases(self, alias: str, expected: ModelSize) -> None:
        assert resolve_model_size(alias) == expected

    def test_model_size_passthrough(self) -> None:
        assert resolve_model_size(ModelSize.LARGE) == ModelSize.LARGE

    def test_invalid_alias_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            resolve_model_size("nonexistent_model")
