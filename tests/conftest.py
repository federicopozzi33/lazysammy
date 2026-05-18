"""Shared test fixtures for lazysammy2."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def sample_image() -> np.ndarray:
    """Return a dummy (H, W, 3) RGB uint8 image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def sample_mask() -> np.ndarray:
    """Return a dummy (H, W) boolean mask."""
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 150:300] = True
    return mask


@pytest.fixture()
def tmp_frames_dir(tmp_path: object, sample_image: np.ndarray) -> object:
    """Create a temporary directory with numbered frame images."""
    from pathlib import Path

    frames = Path(str(tmp_path)) / "frames"
    frames.mkdir()
    # We just create tiny placeholder files – SAM2 won't be loaded in unit tests
    for i in range(5):
        from PIL import Image

        img = Image.fromarray(sample_image)
        img.save(frames / f"{i:05d}.jpg")
    return frames
