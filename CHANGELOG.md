# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **SAM 3 integration** (optional, `pip install lazysammy[sam3]`). A new
  `SAM3` facade mirrors the `SAM2` interface for geometric prompts while
  exposing SAM 3's open-vocabulary concept features:
  - `SAM3.segment_text(image, "a player in white")` detects and segments
    *every* instance of a text concept in an image.
  - `SAM3.segment_exemplar(image, box)` finds all instances matching a box
    exemplar.
  - `SAM3.video(...)` returns a `SAM3VideoSession` whose `add_text()` prompt
    tracks every instance of a concept through a video, assigning each a
    unique object id.
  - `ConceptSegmenter` and `SAM3VideoTracker` are the underlying components;
    `ConceptPrediction` is the new result type (masks + boxes + concept).
  - `save_concept_prediction()` writes concept masks to disk.
- The example notebook gains a SAM 3 section (open-vocabulary concept
  segmentation and video tracking), guarded by an availability check so it is
  skipped when the optional extra is not installed.
- The Gradio demo gains a **Concept Segment (SAM 3)** tab (text-prompt
  segmentation of every matching instance) and a **Concept** text prompt in the
  Video Tracking tab. Adding a concept routes tracking to SAM 3; point/box-only
  tracking still uses SAM 2. Both degrade gracefully with a status message when
  the `sam3` extra is not installed.
- `scripts/build_notebook.py`, which generates `examples/video_tracking.ipynb`
  from reviewable Python source so the notebook stays reproducible and diffable.
- `scripts/make_comparison.py`, which derives the README's side-by-side figure
  from two real inference runs and refuses to write it unless the raw SAM 2 and
  `lazysammy` masks agree.
- `combine_masks()`, `mask_to_bbox()`, and `mask_iou()` are now exported from the
  top-level `lazysammy` package.
- `mask_to_rle()` and `rle_to_mask()` encode/decode single masks to and from
  COCO RLE in memory; `masks_to_rle()` is also exported.
- `SAM2.segment()` accepts `return_logits` to keep raw logits instead of
  thresholding.
- `SAM2.set_image()` / `SAM2.predict()` split image encoding from prompting, so
  many prompts on one image skip the image encoder. `segment()` and `refine()`
  reuse the cached embedding automatically.
- `SAM2.segment_multi_point()` segments several objects in one pass, each with
  its own point prompts.
- `Mask` gains `.bbox`, `.centroid`, `.iou()`, `.to_rle()`, `.from_rle()`, set
  operators (`&`, `|`, `~`), numpy interop (`np.asarray`), and a compact repr.
- `ImagePrediction` is now iterable and indexable, matching the other result
  types.
- `AutoMaskResult.sort_by()` and slice indexing; `VideoResults` slice indexing.
- `to_dict()` / `from_dict()` on `Mask`, `AutoMask`, `AutoMaskResult`,
  `ImagePrediction`, `FrameMasks`, and `VideoResults` for JSON-friendly
  round-tripping (masks as COCO RLE).

### Changed

- `extract_frames()` accepts `clean=True` and now warns when a re-extraction
  leaves a mix of frames from two different settings in the output directory.
- `extract_frames()` validates `frame_format` and raises if a frame cannot be
  written, instead of failing silently.
- `SAM2.auto_segment()` caches tuned auto-segmenters by their settings, so
  repeated calls with the same options no longer reload the model weights.
- `SAM2.resolved_device` now reflects an already-created video or auto-mask
  sub-component, not just the image segmenter.
- `save_masks_as_coco_rle()` reuses the shared RLE encoder.
- `scripts/build_notebook.py` now assigns deterministic cell ids, so
  regenerating `examples/video_tracking.ipynb` produces a byte-identical file
  instead of a spurious diff on every run.
- The example notebook's bidirectional section now prompts on a mid-video frame
  (the case bidirectional tracking exists for) and its mask-prompt section no
  longer builds an unused frame.
- The Gradio demo's prompt bookkeeping and event handlers are now module-level
  functions instead of nested closures, so they can be unit-tested directly.
- `ImageSegmenter.segment()`, `predict()`, and the multi-object helpers now
  share one internal prediction path, so mask-input normalization and the
  inference wrapper live in a single place.
- `SAM2.auto_segment()` keys its tuned-segmenter cache with a frozen
  `AutoSegmentSettings` dataclass instead of a positional tuple.
- The demo's video prompts are typed `PointPrompt` / `BoxPrompt` objects rather
  than `dict`s with a `"type"` string, so drawing, session application, and
  summary rendering are polymorphic instead of re-branched per call site.
- The demo's image tab now uses the same typed prompt model (an `ImagePrompts`
  holding a `PointPrompt` and pending box corners) instead of `list[Any]` with
  magic indices, and its handlers return an `ImageTabState` NamedTuple.
- `build_app()` is split into `_build_image_tab()`, `_build_auto_tab()`, and
  `_build_video_tab()`; the top-level builder is now a short composition.
- The demo's image segmentation point/box paths share one
  `_run_image_segmentation()` helper, and the video handlers share
  `_render_frame()`.
- The demo separates the model-load lock from the inference lock, so loading a
  second model size no longer blocks inference on an already-loaded one.
- The demo writes rendered tracking videos into one shared output directory
  instead of leaking a fresh temp directory per track.

### Fixed

- The Gradio demo's image upload/reset listener used `gr.Image.change`, which
  also fires on programmatic updates. Because the click handler writes the
  annotated image back into the same component, every click re-entered the
  upload handler and wiped the prompt state, so Segment always answered "Add
  some prompts first". It now uses `.input()` (user-only). The frame slider had
  the same problem and is fixed the same way.
- The demo's video frame extraction now passes `clean=True`, so re-extracting
  with different settings no longer mixes frames from two runs.
- Removed a dead `_format_prompts` helper and an unreachable branch in the
  video click handler.
- `Mask` equality no longer raises `ValueError` on its numpy array field; the
  dataclass `__eq__` is now numpy-safe (and `__hash__` is disabled, as the type
  is mutable).
- `Mask.save()` now infers the format from the file extension (`.png`, `.npy`,
  `.json`) or an explicit `fmt=`, instead of always writing PNG and silently
  ignoring the extension.
- `vos_optimized=True` was silently ignored when loading the video predictor
  from the HuggingFace Hub (the default path); it is now forwarded.
- `segment(..., return_logits=True)` stored raw logits as booleans, so negative
  logits were treated as foreground. Logits are now thresholded at `0`.
- A 2D `mask_input` passed to `segment()` is now promoted to `(1, H, W)` before
  being handed to SAM 2, which only batches 3D mask logits.
- The README and CONTRIBUTING documented `uv sync --extra dev`, but `dev` is a
  dependency group, not an extra; the commands now use `uv sync`.

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
