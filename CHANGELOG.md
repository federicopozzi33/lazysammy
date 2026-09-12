# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

Initial public release.

### Added

- `SAM2` facade as a single entry point for image segmentation, video object
  tracking, and automatic mask generation, with lazy sub-component loading.
- Typed result objects: `Mask`, `ImagePrediction`, `AutoMask`, `AutoMaskResult`,
  `FrameMasks`, and `VideoResults`.
- Image segmentation via `segment()`, `segment_point()`, `segment_box()`,
  `segment_multi_box()`, `segment_batch()`, and iterative `refine()`.
- Video tracking via `VideoSession` with point, box, and mask prompts on any
  frame, forward/reverse/bidirectional propagation, and object management.
- Automatic mask generation with area and IoU filtering.
- Serialization to PNG, NumPy `.npy`, and COCO RLE.
- Visualization and overlay helpers, including `save_video_overlay_mp4()`.
- Interactive click-to-prompt helpers (`PromptPicker`, `pick_points_on_image()`,
  `pick_box_on_image()`, `preview_frame()`) for notebooks.
- Automatic device detection across CUDA, Apple MPS, and CPU.
- A Gradio demo application with image, auto-segment, and video tabs.
- A guided example notebook (`examples/video_tracking.ipynb`).
- `py.typed` marker and `mypy --strict` clean type annotations.
- Fast unit tests plus an opt-in real-weights integration suite.
