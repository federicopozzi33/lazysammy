"""Generate the validated example notebook.

Run from the repo root with the notebook extra installed::

    uv run python scripts/build_notebook.py

Kept as a script (rather than editing JSON by hand) so the notebook stays
reproducible and its cell content is reviewable in a normal diff.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "examples" / "video_tracking.ipynb"


def md(text: str) -> nbf.NotebookNode:
    """Build a markdown cell from *text*."""
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    """Build a code cell from *text*."""
    return nbf.v4.new_code_cell(text.strip("\n"))


cells = [
    md(
        """
# Video Object Tracking with lazysammy

This notebook walks through the **video capabilities** of `lazysammy`, a
high-level wrapper around Meta's [SAM 2](https://github.com/facebookresearch/sam2).

**What you will learn**

1. Load a video — either an `.mp4` file or a folder of frames
2. Place prompts interactively (click) or manually (coordinates)
3. Track multiple objects with prompts on arbitrary frames
4. Propagate forward, backward, and bidirectionally
5. Inspect, visualise, and save results
6. Manage objects mid-sequence (add, correct, remove, reset)

> **Setup:** install the notebook extras first:
> `uv sync --extra notebook --extra viz`

> **Interactive clicking:** the setup cell selects `%matplotlib widget`, which
> requires `ipympl` (included in the `notebook` extra). If your environment
> falls back to a non-interactive backend, this notebook detects it and tells
> you — the manual coordinate API below always works either way.
"""
    ),
    md("## 0 — Setup"),
    code(
        """
from pathlib import Path

import cv2
import matplotlib

# Choose a matplotlib backend. 'widget' supports click-to-prompt (ipympl, from
# the notebook extra); 'inline' is the safe fallback. A non-widget backend is
# not an error: the notebook detects it and skips the interactive cell.
try:
    get_ipython().run_line_magic("matplotlib", "widget")
except Exception:
    get_ipython().run_line_magic("matplotlib", "inline")

import matplotlib.pyplot as plt
import numpy as np

import lazysammy
from lazysammy import (
    SAM2,
    PromptPicker,
    VideoTracker,
    draw_masks_on_image,
    extract_frames,
    is_interactive_backend,
    pick_box_on_image,
    pick_points_on_image,
    preview_frame,
    save_video_overlay,
    show_video_frame,
)

plt.rcParams["figure.dpi"] = 110

print(f"lazysammy {lazysammy.__version__}")
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"interactive backend available: {is_interactive_backend()}")
"""
    ),
    md(
        """
## 1 — Load a video

`lazysammy` accepts **either** a video file (`.mp4`, `.avi`, `.mov`, …) **or**
a directory of frames. When you pass a video file, frames are extracted for you.
"""
    ),
    code(
        """
# Point this at your own clip, or use the bundled example.
VIDEO_PATH = "assets/fair.mp4"   # relative to the examples/ directory
# VIDEO_PATH = "path/to/frames/" # a frame directory works too

# If you need finer control over extraction, do it explicitly instead:
# frames_dir = extract_frames(
#     "assets/fair.mp4",
#     output_dir="my_frames/",   # default: <stem>_frames/
#     every_n=2,                 # keep every 2nd frame
#     max_frames=200,            # stop after 200 frames
# )
# VIDEO_PATH = frames_dir

video_path = Path(VIDEO_PATH)
if not video_path.exists():
    raise FileNotFoundError(
        f"{video_path} not found. Run this notebook from the examples/ directory "
        "or set VIDEO_PATH to your own video or frame directory."
    )
print(f"Input: {video_path} ({'video file' if video_path.is_file() else 'frame directory'})")
"""
    ),
    code(
        """
# A smaller model keeps this notebook quick; use "large" for best quality.
MODEL_SIZE = "small"   # "tiny" | "small" | "base_plus" | "large"

sam = SAM2(MODEL_SIZE)

# The bundled clip is ~10s long. Tracking every frame on CPU takes a while and
# this notebook propagates several times, so we cap the length by default.
# Raise MAX_FRAMES (or set it to None) on a GPU, or point VIDEO_PATH at your
# own clip.
MAX_FRAMES = 60

session = sam.video(
    VIDEO_PATH,
    # --- video-file options (ignored for a frame directory) ---
    every_n=1,          # keep every frame
    max_frames=MAX_FRAMES,
    clean=True,         # clear stale frames so max_frames is actually honoured
    # frames_dir=None,  # default: <stem>_frames/
)
print(f"Session ready: {session.num_frames} frames from {session.video_dir}")
"""
    ),
    md(
        """
## 2 — Preview a frame and find coordinates

Before adding prompts you need to know *where* to click. `preview_frame()` shows a
frame with a coordinate grid; on an interactive backend the toolbar also reports
live `(x, y)` under the cursor.
"""
    ),
    code(
        """
# Show frame 0 with a coordinate grid.
preview_frame(session.video_dir, frame_idx=0, grid_step=100)
"""
    ),
    code(
        """
# You can also pull a frame as a plain RGB numpy array.
first_frame = session.get_frame(0)
print(f"Frame shape: {first_frame.shape}  (H, W, C)")
"""
    ),
    md(
        """
## 3 — Placing prompts

### 3.1 — Manually, with coordinates

The most portable approach: pass explicit `(x, y)` points, read off the grid
above. This works regardless of matplotlib backend.
"""
    ),
    code(
        """
# Add a foreground point for object 1 on frame 0.
frame_result = session.add_points(
    frame_idx=0,
    obj_id=1,
    points=[[350, 250]],
    labels=[1],   # 1 = foreground (include), 0 = background (exclude)
)
print(f"Objects with masks on frame 0: {frame_result.object_ids}")
for oid, mask in frame_result.masks.items():
    print(f"  object {oid}: {int(mask.sum())} px")
"""
    ),
    code(
        """
# Preview what that single click produced.
fig = show_video_frame(first_frame, frame_result, alpha=0.5)
plt.show()
"""
    ),
    md(
        """
### 3.2 — Interactively, by clicking

`PromptPicker` binds to the session: click directly on the frame and the prompt
is sent to SAM 2 for you.

| Action | How |
|--------|-----|
| Foreground point | **Left-click** |
| Background point | **Right-click** |
| Finish picking | **Enter**, **q**, or **Esc** |
| Draw a box | **Left-click** both corners |

> Requires an interactive backend. If `is_interactive_backend()` was `False`
> above, skip this section — it will warn instead of silently doing nothing.
"""
    ),
    code(
        """
picker = PromptPicker(session)
picker.preview(frame_idx=0)
"""
    ),
    code(
        """
# Click to place points for object 2. Left = foreground, right = background.
# Finish with Enter/q/Esc.
#
# This is the one cell that needs a human at the keyboard. We fall back to
# coordinates when clicks are unavailable OR when nothing was clicked (for
# example when the notebook is executed non-interactively), so the rest of the
# notebook always describes a real, populated session.
points, labels = (
    picker.add_points_interactive(frame_idx=0, obj_id=2)
    if is_interactive_backend()
    else ([], [])
)
if not points:
    print("No clicks received (non-interactive run) — using coordinates instead.")
    points, labels = [[450, 300]], [1]
    session.add_points(frame_idx=0, obj_id=2, points=points, labels=labels)
print(f"object 2 -> {len(points)} point(s), labels={labels}")
"""
    ),
    md(
        """
### 3.3 — Prompts on later frames

Prompts can live on **any** frame, not just the first. Useful when an object
enters mid-clip, or when you want to correct drift.
"""
    ),
    code(
        """
# Add a further object that starts partway through the clip.
frame_result = session.add_points(
    frame_idx=10,
    obj_id=3,
    points=[[280, 320]],
    labels=[1],
)
print(f"Added object 3 on frame 10. Objects there: {frame_result.object_ids}")
"""
    ),
    md(
        """
### 3.4 — Foreground and background together

Negative points (label `0`) exclude regions from a mask — handy when an object
is similar in colour to its surroundings.
"""
    ),
    code(
        """
# Re-prompt object 1 with a positive point and a negative correction.
frame_result = session.add_points(
    frame_idx=0,
    obj_id=1,
    points=[
        [350, 250],   # foreground
        [355, 240],   # foreground
        [300, 380],   # background — exclude this area
    ],
    labels=[1, 1, 0],
    clear_old=True,   # replace this object's previous prompts on this frame
)
print(f"Object 1 on frame 0: {int(frame_result.masks[1].sum())} px")
"""
    ),
    md(
        """
## 4 — Propagate through the video

With prompts in place, propagate to track every object across frames.
"""
    ),
    code(
        """
# Forward propagation: from the earliest prompted frame to the end.
results = session.propagate()

print(f"Tracked {len(results)} frame(s)")
print(f"Object ids: {sorted(results.object_ids)}")
"""
    ),
    code(
        """
# Write an overlay video: each frame has tracked masks drawn on top.
session.save_overlay("output/tracking.mp4", fps=24, alpha=0.5)
print("Wrote output/tracking.mp4")
"""
    ),
    code(
        """
# Inspect a few frames in a row. Note that iterating VideoResults yields
# (frame_idx, FrameMasks) pairs.
tracked_indices = [idx for idx, _ in results]
mid_idx = len(tracked_indices) // 2
sample_frames = [
    tracked_indices[0],
    tracked_indices[mid_idx],
    tracked_indices[-1],
]

fig, axes = plt.subplots(1, len(sample_frames), figsize=(5 * len(sample_frames), 5))
if len(sample_frames) == 1:
    axes = [axes]

for ax, fidx in zip(axes, sample_frames):
    fm = results[fidx]
    frame_img = session.get_frame(fidx)
    oids = fm.object_ids
    # FrameMasks.masks is a dict {obj_id: mask}, not an array.
    stacked = (
        np.stack([fm.masks[oid] for oid in oids])
        if oids
        else np.zeros((0, *frame_img.shape[:2]), dtype=bool)
    )
    overlay = draw_masks_on_image(frame_img, stacked, alpha=0.5)
    ax.imshow(overlay)
    ax.set_title(f"Frame {fidx} ({len(oids)} object(s))")
    ax.axis("off")

plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
### 4.1 — Bidirectional propagation

If your earliest prompt sits in the middle of the clip, track forward **and**
backward from it.
"""
    ),
    code(
        """
session.reset()

mid = session.num_frames // 2
session.add_points(frame_idx=mid, obj_id=1, points=[[350, 250]], labels=[1])

results_bidir = session.propagate_bidirectional()
tracked = [idx for idx, _ in results_bidir]
print(f"Bidirectional: {len(results_bidir)} frame(s), from {tracked[0]} to {tracked[-1]}")
"""
    ),
    md("### 4.2 — Partial propagation"),
    code(
        """
session.reset()
session.add_points(frame_idx=0, obj_id=1, points=[[350, 250]], labels=[1])

partial = session.propagate(max_frames=30)
print(f"Partial propagation: {len(partial)} frame(s)")
"""
    ),
    md(
        """
## 5 — Inspect results

`VideoResults` supports iteration, indexing, and per-object queries.
"""
    ),
    code(
        """
# Re-run full tracking with two objects for the examples below.
session.reset()
session.add_points(frame_idx=0, obj_id=1, points=[[350, 250]], labels=[1])
session.add_box(frame_idx=0, obj_id=2, box=[400, 150, 600, 400])
results = session.propagate()

# --- Iterate over frames: yields (frame_idx, FrameMasks) pairs ---
for frame_idx, frame_masks in results:
    _ = frame_idx, frame_masks  # process each frame here

# --- Access a single frame by index ---
frame0 = results[0]
print(f"Frame 0 -> objects {frame0.object_ids}")
print(f"Object 1 mask shape: {frame0.masks[1].shape}")

# --- All masks for one object across time: {frame_idx: mask} ---
obj1_masks = results.get_object_masks(obj_id=1)
print(f"Object 1 tracked in {len(obj1_masks)} frame(s)")

# --- All object ids seen anywhere in the video ---
print(f"All object ids: {sorted(results.object_ids)}")
"""
    ),
    code(
        """
# Plot each object's mask area over time — a quick drift sanity check.
fig, ax = plt.subplots(figsize=(11, 4))

for obj_id in sorted(results.object_ids):
    obj_masks = results.get_object_masks(obj_id)
    frames_sorted = sorted(obj_masks)
    areas = [int(obj_masks[f].sum()) for f in frames_sorted]
    ax.plot(frames_sorted, areas, label=f"Object {obj_id}")

ax.set_xlabel("Frame index")
ax.set_ylabel("Mask area (pixels)")
ax.set_title("Object mask area over time")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 6 — Manage objects mid-sequence

You can remove objects, clear individual prompts, and reset the whole session.
"""
    ),
    code(
        """
# Remove object 2 entirely and re-propagate.
session.remove_object(obj_id=2)
results_after_removal = session.propagate()
print(f"After removal -> object ids: {sorted(results_after_removal.object_ids)}")
"""
    ),
    code(
        """
# Clear one object's prompts on one frame (e.g. to fix a bad click),
# without discarding the rest of the object's tracking.
session.clear_frame_prompts(frame_idx=0, obj_id=1)

session.add_points(
    frame_idx=0,
    obj_id=1,
    points=[[352, 252], [345, 260]],
    labels=[1, 1],
)
results_corrected = session.propagate()
print(f"Corrected tracking: {len(results_corrected)} frame(s)")
"""
    ),
    code(
        """
# Full reset: clears all prompts and tracking state.
session.reset()
print("Session reset.")
"""
    ),
    md(
        """
## 7 — Save results

Masks can be written as **PNG**, **NumPy `.npy`**, or **COCO RLE JSON**.
"""
    ),
    code(
        """
# Set up one final tracking run to save.
session.add_points(frame_idx=0, obj_id=1, points=[[350, 250]], labels=[1])
session.add_box(frame_idx=0, obj_id=2, box=[400, 150, 600, 400])
results = session.propagate()
print(f"Tracking {len(results.object_ids)} object(s) over {len(results)} frame(s)")
"""
    ),
    code(
        """
# PNG: output/masks_png/frame_000000/obj_0001.png ...
session.save("output/masks_png", fmt="png")

# NPY: one .npy per object per frame
# session.save("output/masks_npy", fmt="npy")

# COCO RLE: one .json per object per frame
# session.save("output/masks_rle", fmt="coco_rle")

print("Saved masks to output/masks_png")
"""
    ),
    code(
        """
# Overlay frames as individual PNGs (instead of an MP4).
save_video_overlay(session.video_dir, results, "output/overlay_frames", alpha=0.5)
print("Wrote overlay frames to output/overlay_frames")
"""
    ),
    md(
        """
## 8 — Using a mask prompt

Besides points and boxes, an object can be initialised from an **existing binary
mask** — for example from another model or a manual annotation.
"""
    ),
    code(
        """
session.reset()

# Build a simple mask from a previous prediction, then feed it back in.
seed = session.get_frame(0)
seed_result = session.add_points(frame_idx=0, obj_id=1, points=[[350, 250]], labels=[1])
seed_mask = seed_result.masks[1]

session.reset()
mask_result = session.add_mask(frame_idx=0, obj_id=1, mask=seed_mask)
print(f"Initialised object 1 from a {seed_mask.shape} mask -> {mask_result.object_ids}")

results_from_mask = session.propagate()
print(f"Tracked {len(results_from_mask)} frame(s) from the mask prompt")
"""
    ),
    md(
        """
## 9 — Using the sub-components directly

`sam.video(...)` is a shortcut for `VideoTracker(...).new_session(...)`. You can
use the lower-level object directly when you want explicit control.
"""
    ),
    code(
        """
tracker = VideoTracker(
    MODEL_SIZE,
    device="cpu",         # drop this to auto-detect CUDA / MPS / CPU
    vos_optimized=False,  # True compiles the model (CUDA + PyTorch >= 2.5.1)
)

session2 = tracker.new_session(
    VIDEO_PATH,
    max_frames=MAX_FRAMES,        # same cap as the main session
    clean=True,                   # ignore frames left by the earlier cells
    offload_video_to_cpu=True,    # lower GPU memory usage
    offload_state_to_cpu=False,
)
session2.add_points(frame_idx=0, obj_id=1, points=[[350, 250]], labels=[1])
results2 = session2.propagate()
print(f"Direct tracker: {len(results2)} frame(s), device={tracker.device}")
"""
    ),
    md(
        """
## 10 — Standalone interactive helpers

`pick_points_on_image` and `pick_box_on_image` work on **any** image and return
coordinates you can pass to a session yourself.
"""
    ),
    code(
        """
if is_interactive_backend():
    some_image = session.get_frame(0)
    picked_points, picked_labels = pick_points_on_image(some_image, n=3)
    picked_box = pick_box_on_image(some_image)
    print(f"Points: {picked_points}")
    print(f"Labels: {picked_labels}")
    print(f"Box: {picked_box}")

    # Then feed them to the session, e.g.:
    # session.add_points(frame_idx=0, obj_id=1, points=picked_points, labels=picked_labels)
    # session.add_box(frame_idx=0, obj_id=2, box=picked_box)
else:
    print("Non-interactive backend: skipping click-based picking.")
    print("Use preview_frame() and pass coordinates to add_points/add_box manually.")
"""
    ),
    md(
        """
## 11 — Memory and performance tips

| Goal | How |
|------|-----|
| Use a smaller model | `SAM2("tiny")` or `SAM2("small")` |
| Offload frames to CPU | `sam.video(path, offload_video_to_cpu=True)` |
| Offload state to CPU | `sam.video(path, offload_state_to_cpu=True)` |
| Subsample frames | `sam.video("clip.mp4", every_n=3)` |
| Limit propagation | `session.propagate(max_frames=100)` |
| Faster video on CUDA | `SAM2("large", vos_optimized=True)` |
| Free the model | `del sam` or restart the kernel |
"""
    ),
    md(
        """
---

## Summary

You now know how to:

- Load a video from a file or a frame directory
- Place prompts manually or interactively (clicks)
- Track one or many objects, with prompts on any frame
- Propagate forward, backward, or bidirectionally
- Inspect results per frame and per object
- Save masks as PNG, `.npy`, or COCO RLE, and render overlay videos
- Manage objects mid-sequence

For image segmentation and auto-mask workflows, see the
[README](https://github.com/federicopozzi33/easier-sam2#readme):

```python
pred = sam.segment("photo.jpg", points=[[x, y]], labels=[1])
auto = sam.auto_segment("photo.jpg")
```
"""
    ),
]


def main() -> None:
    """Write the notebook to disk."""
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(OUT))
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
