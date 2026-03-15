"""easier-sam2 – A simplified wrapper around Meta's SAM 2.

Quick start::

    from easier_sam2 import SAM2

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

from easier_sam2.auto_mask import AutoSegmenter
from easier_sam2.image import ImageSegmenter
from easier_sam2.prompts import PromptPicker, pick_box_on_image, pick_points_on_image, preview_frame
from easier_sam2.sam2 import SAM2
from easier_sam2.types import (
    AutoMask,
    AutoMaskResult,
    FrameMasks,
    ImagePrediction,
    Mask,
    ModelSize,
    VideoResults,
)
from easier_sam2.utils import extract_frames
from easier_sam2.video import VideoSession, VideoTracker
from easier_sam2.visualization import (
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
    # Main facade
    "SAM2",
    # Sub-components
    "ImageSegmenter",
    "VideoTracker",
    "VideoSession",
    "AutoSegmenter",
    # Interactive prompts
    "PromptPicker",
    "pick_points_on_image",
    "pick_box_on_image",
    "preview_frame",
    # Utilities
    "extract_frames",
    # Result types
    "Mask",
    "ImagePrediction",
    "AutoMask",
    "AutoMaskResult",
    "FrameMasks",
    "VideoResults",
    "ModelSize",
    # Visualization
    "draw_masks_on_image",
    "draw_points_on_image",
    "draw_box_on_image",
    "show_image_prediction",
    "show_auto_masks",
    "show_video_frame",
    "save_video_overlay",
    "save_video_overlay_mp4",
]
