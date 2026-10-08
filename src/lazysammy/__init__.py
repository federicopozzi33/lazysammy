"""lazysammy - A simplified wrapper around Meta's SAM 2.

Quick start::

    from lazysammy import SAM2

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

from lazysammy.auto_mask import AutoSegmenter
from lazysammy.concept import ConceptSegmenter
from lazysammy.image import ImageSegmenter
from lazysammy.io import (
    SaveFormat,
    save_auto_mask_result,
    save_concept_prediction,
    save_image_prediction,
    save_video_results,
)
from lazysammy.prompts import (
    PromptPicker,
    is_interactive_backend,
    pick_box_on_image,
    pick_points_on_image,
    preview_frame,
)
from lazysammy.sam2 import SAM2
from lazysammy.sam3 import SAM3
from lazysammy.sam3_video import SAM3VideoSession, SAM3VideoTracker
from lazysammy.types import (
    AutoMask,
    AutoMaskResult,
    BoundingBox,
    ConceptPrediction,
    FrameMasks,
    ImagePrediction,
    Mask,
    ModelSize,
    PointCoords,
    PointLabels,
    VideoResults,
)
from lazysammy.utils import (
    combine_masks,
    extract_frames,
    load_image,
    mask_iou,
    mask_to_bbox,
    mask_to_rle,
    masks_to_colored_overlay,
    masks_to_rle,
    natural_sort_key,
    rle_to_mask,
    save_masks_as_coco_rle,
    save_masks_as_npy,
    save_masks_as_png,
)
from lazysammy.video import VideoSession, VideoTracker
from lazysammy.visualization import (
    draw_box_on_image,
    draw_masks_on_image,
    draw_points_on_image,
    save_video_overlay,
    save_video_overlay_mp4,
    show_auto_masks,
    show_image_prediction,
    show_video_frame,
)

__version__ = "0.1.0"

__all__ = [
    "SAM2",
    "SAM3",
    "AutoMask",
    "AutoMaskResult",
    "AutoSegmenter",
    "BoundingBox",
    "ConceptPrediction",
    "ConceptSegmenter",
    "FrameMasks",
    "ImagePrediction",
    "ImageSegmenter",
    "Mask",
    "ModelSize",
    "PointCoords",
    "PointLabels",
    "PromptPicker",
    "SAM3VideoSession",
    "SAM3VideoTracker",
    "SaveFormat",
    "VideoResults",
    "VideoSession",
    "VideoTracker",
    "__version__",
    "combine_masks",
    "draw_box_on_image",
    "draw_masks_on_image",
    "draw_points_on_image",
    "extract_frames",
    "is_interactive_backend",
    "load_image",
    "mask_iou",
    "mask_to_bbox",
    "mask_to_rle",
    "masks_to_colored_overlay",
    "masks_to_rle",
    "natural_sort_key",
    "pick_box_on_image",
    "pick_points_on_image",
    "preview_frame",
    "rle_to_mask",
    "save_auto_mask_result",
    "save_concept_prediction",
    "save_image_prediction",
    "save_masks_as_coco_rle",
    "save_masks_as_npy",
    "save_masks_as_png",
    "save_video_overlay",
    "save_video_overlay_mp4",
    "save_video_results",
    "show_auto_masks",
    "show_image_prediction",
    "show_video_frame",
]
