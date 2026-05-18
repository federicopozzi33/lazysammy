"""lazysammy2 - A simplified wrapper around Meta's SAM 2.

Quick start::

    from lazysammy2 import SAM2

    sam = SAM2("large")

    # Single-image segmentation
    pred = sam.segment("photo.jpg", points=[[100, 200]], labels=[1])
    pred.best_mask.save("mask.png")

    # Auto-segment everything
    auto = sam.auto_segment("photo.jpg")

    # Video tracking
    session = sam.video("frames/")
    session.add_points(frame_idx=0, obj_id=1, points=[[150, 300]], labels=[1])
    results = session.propagate()
    session.save("output/")
"""

from __future__ import annotations

from lazysammy2.auto_mask import AutoSegmenter
from lazysammy2.image import ImageSegmenter
from lazysammy2.prompts import PromptPicker, pick_box_on_image, pick_points_on_image, preview_frame
from lazysammy2.sam2 import SAM2
from lazysammy2.types import (
    AutoMask,
    AutoMaskResult,
    FrameMasks,
    ImagePrediction,
    Mask,
    ModelSize,
    VideoResults,
)
from lazysammy2.utils import extract_frames
from lazysammy2.video import VideoSession, VideoTracker
from lazysammy2.visualization import (
    draw_box_on_image,
    draw_masks_on_image,
    draw_points_on_image,
    save_video_overlay,
    save_video_overlay_mp4,
    show_auto_masks,
    show_image_prediction,
    show_video_frame,
)

__all__ = [
    "SAM2",
    "AutoMask",
    "AutoMaskResult",
    "AutoSegmenter",
    "FrameMasks",
    "ImagePrediction",
    "ImageSegmenter",
    "Mask",
    "ModelSize",
    "PromptPicker",
    "VideoResults",
    "VideoSession",
    "VideoTracker",
    "draw_box_on_image",
    "draw_masks_on_image",
    "draw_points_on_image",
    "extract_frames",
    "pick_box_on_image",
    "pick_points_on_image",
    "preview_frame",
    "save_video_overlay",
    "save_video_overlay_mp4",
    "show_auto_masks",
    "show_image_prediction",
    "show_video_frame",
]
