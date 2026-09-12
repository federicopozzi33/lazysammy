"""Shared test fixtures for lazysammy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_image() -> np.ndarray:
    """Return a dummy (H, W, 3) RGB uint8 image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def small_image() -> np.ndarray:
    """Return a small deterministic (H, W, 3) RGB uint8 image."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = 255
    return img


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Return a dummy (H, W) boolean mask."""
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 150:300] = True
    return mask


@pytest.fixture
def tmp_frames_dir(tmp_path: Path, sample_image: np.ndarray) -> Path:
    """Create a temporary directory with five zero-padded frame images."""
    from PIL import Image

    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(5):
        Image.fromarray(sample_image).save(frames / f"{i:05d}.jpg")
    return frames


@pytest.fixture
def tmp_video_file(tmp_path: Path) -> Path:
    """Create a short synthetic MP4 video for frame-extraction tests."""
    import cv2

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (64, 64)
    )
    if not writer.isOpened():  # pragma: no cover - codec availability
        pytest.skip("No MP4 codec available in this OpenCV build.")
    try:
        for i in range(8):
            frame = np.full((64, 64, 3), i * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path
