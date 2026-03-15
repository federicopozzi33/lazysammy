# easier-sam2

A simplified, batteries-included wrapper around [Meta's SAM 2](https://github.com/facebookresearch/sam2).

**easier-sam2** provides a single `SAM2` object that gives you access to:

- **Image segmentation** – point, box, mask and multi-box prompts with iterative refinement
- **Video object tracking** – multi-object tracking with prompts at arbitrary frames and bidirectional propagation
- **Automatic mask generation** – zero-prompt "segment everything" with filtering helpers
- **Saving** – PNG, NumPy `.npy`, and COCO RLE (JSON)
- **Visualization** – OpenCV overlays and matplotlib plots

---

## Installation

The project uses [uv](https://docs.astral.sh/uv/) as its package manager.

```bash
# Clone and install
git clone https://github.com/your-org/easier-sam2.git
cd easier-sam2
uv sync

# With visualization extras (matplotlib)
uv sync --extra viz

# Development (linting, testing)
uv sync --extra dev
```

> **Note:** SAM 2 is installed directly from the official GitHub repository. A CUDA GPU is recommended but MPS (Apple Silicon) and CPU are also supported.

---

## Quick start

```python
from easier_sam2 import SAM2

sam = SAM2("large")  # auto-downloads from HuggingFace Hub
```

### Image segmentation

```python
# Segment with a point prompt
pred = sam.segment("photo.jpg", points=[[100, 200]], labels=[1])
print(pred.best_mask.score)       # confidence score
mask_array = pred.best_mask.numpy()  # (H, W) bool array

# Save the best mask
pred.best_mask.save("mask.png")

# Segment with a bounding box
pred = sam.segment("photo.jpg", box=[50, 60, 300, 400])

# One-liner shortcuts
pred = sam.segment_point("photo.jpg", x=100, y=200)
pred = sam.segment_box("photo.jpg", 50, 60, 300, 400)

# Segment multiple objects at once (one box each)
preds = sam.segment_multi_box("photo.jpg", boxes=[[50, 60, 300, 400], [400, 100, 600, 350]])

# Iterative refinement
pred1 = sam.segment_point("photo.jpg", x=100, y=200)
pred2 = sam.refine("photo.jpg", pred1.best_mask.logits, points=[[120, 180]], labels=[0])
```

### Auto-segmentation ("segment everything")

```python
auto = sam.auto_segment("photo.jpg")
print(f"Found {len(auto.masks)} masks")

# Filter by area
large = auto.filter_by_area(min_area=1000)

# Filter by predicted IoU
good = auto.filter_by_iou(min_iou=0.9)

# Save all masks
auto.save("output/auto_masks/")
```

### Video tracking

```python
# Start a session from a directory of JPEG/PNG frames
session = sam.video("path/to/frames/")

# Add prompts on arbitrary frames
session.add_points(frame_idx=0, obj_id=1, points=[[150, 300]], labels=[1])
session.add_box(frame_idx=10, obj_id=2, box=[50, 60, 300, 400])
session.add_mask(frame_idx=5, obj_id=3, mask=my_mask_array)

# Propagate forward from the first prompted frame
results = session.propagate()

# Or propagate in both directions
results = session.propagate_bidirectional()

# Access results
for frame_idx, frame_masks in results:
    for obj_id, mask in frame_masks.masks.items():
        print(f"Frame {frame_idx}, Object {obj_id}: mask shape {mask.shape}")

# Get masks for a specific object across all frames
obj1_masks = results.get_object_masks(obj_id=1)

# Save everything (PNG + NumPy + COCO RLE)
session.save("output/video_masks/")

# Remove an object and re-propagate
session.remove_object(obj_id=2)
results = session.propagate()

# Start over
session.reset()
```

---

## Advanced usage

### Model loading options

```python
# By size alias (downloads from HuggingFace Hub)
sam = SAM2("large")          # also: "tiny", "small", "base_plus"
sam = SAM2("l")              # short aliases work too
sam = SAM2("b+")             # base_plus shorthand

# Local checkpoint
sam = SAM2("large", checkpoint="/path/to/sam2.1_hiera_large.pt")

# Explicit device
sam = SAM2("large", device="cuda:1")
sam = SAM2("large", device="mps")
sam = SAM2("large", device="cpu")

# Video-optimized mode (torch.compile, CUDA only, PyTorch >= 2.5.1)
sam = SAM2("large", vos_optimized=True)
```

### Using sub-components directly

```python
from easier_sam2 import ImageSegmenter, VideoTracker, AutoSegmenter

# Each component can be used independently
img_seg = ImageSegmenter("large", device="cuda")
vid_tracker = VideoTracker("large", device="cuda", vos_optimized=True)
auto_seg = AutoSegmenter("large")
```

### Visualization

```python
from easier_sam2 import (
    draw_masks_on_image,
    draw_points_on_image,
    draw_box_on_image,
    show_image_prediction,
    show_auto_masks,
    show_video_frame,
    save_video_overlay,
)
import cv2

image = cv2.imread("photo.jpg")

# Draw masks on an image (OpenCV)
overlay = draw_masks_on_image(image, pred.best_mask.numpy(), alpha=0.5)
cv2.imwrite("overlay.jpg", overlay)

# Matplotlib-based visualization (requires `viz` extra)
show_image_prediction(image, pred)
show_auto_masks(image, auto)

# Save video tracking overlays
save_video_overlay("path/to/frames/", results, "output/overlays/")
```

### Saving results

```python
from easier_sam2.utils import (
    save_masks_as_png,
    save_masks_as_npy,
    save_masks_as_coco_rle,
    save_video_results,
    save_image_prediction,
    save_auto_mask_result,
)

# Save image prediction in all formats
save_image_prediction(pred, "output/image_pred/")

# Save video results
save_video_results(results, "output/video/")

# Individual mask saving
import numpy as np
mask = pred.best_mask.numpy()
save_masks_as_png({"object_0": mask}, "output/png_masks/")
save_masks_as_npy({"object_0": mask}, "output/npy_masks/")
save_masks_as_coco_rle({"object_0": mask}, "output/rle_masks/")
```

---

## Result types

| Type | Description |
|------|-------------|
| `Mask` | Single binary mask with `.score`, `.logits`, `.numpy()`, `.as_uint8()`, `.save()` |
| `ImagePrediction` | Collection of masks from a single image prediction; `.best_mask` picks the highest-score one |
| `AutoMask` | Mask from auto-generation with extra metadata (area, bbox, stability score, crop box) |
| `AutoMaskResult` | Collection of auto-masks; `.filter_by_area()`, `.filter_by_iou()` |
| `FrameMasks` | Object masks for a single video frame |
| `VideoResults` | All frame masks across a video; iterable, indexable, `.get_object_masks()` |

---

## Examples

See the [examples/](examples/) folder for Jupyter notebooks:

- **[Video Tracking](examples/video_tracking.ipynb)** — multi-object tracking, bidirectional propagation, prompts at arbitrary frames, saving results

---

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Lint
uv run ruff check src/
uv run ruff format --check src/

# Type checking
uv run mypy src/

# Run tests
uv run pytest
```

---

## Model sizes

| Size | Model ID | Parameters |
|------|----------|------------|
| `tiny` | `facebook/sam2.1-hiera-tiny` | ~39M |
| `small` | `facebook/sam2.1-hiera-small` | ~46M |
| `base_plus` | `facebook/sam2.1-hiera-base-plus` | ~81M |
| `large` | `facebook/sam2.1-hiera-large` | ~224M |

---

## License

This project wraps SAM 2 which is licensed under the [Apache 2.0 License](https://github.com/facebookresearch/sam2/blob/main/LICENSE).
