"""Model runtime for the demo: lazy caches, locks, and error policy.

SAM 2 predictors hold GPU memory and are not thread-safe, so construction is
guarded by a lock and every inference call is serialised through it. This keeps
concurrent Gradio requests from corrupting shared session state.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

from lazysammy import SAM2, SAM3

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, SAM2] = {}
# SAM 3 is a separate, optional model (open-vocabulary concepts). It is cached
# under a single key because it has no size variants.
_SAM3_CACHE: dict[str, SAM3] = {}
# Guards model construction only, so loading multi-GB weights never happens
# twice for the same size.
MODEL_LOAD_LOCK = threading.Lock()
# SAM 2 predictors are not thread-safe, so every inference call is serialised.
# This lock is intentionally coarse: it also covers video propagation and MP4
# encoding, which keeps concurrent Gradio requests from corrupting shared state.
INFERENCE_LOCK = threading.Lock()

# Errors the demo surfaces as a status message instead of crashing the request.
DEMO_ERRORS = (ValueError, RuntimeError, IndexError, OSError)

# Output directory for rendered tracking videos. Created once so repeated
# tracks do not leak a temp directory each; Gradio serves the file from here.
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="lazysammy_demo_"))


def get_model(model_size: str) -> SAM2:
    """Return a cached :class:`SAM2` for *model_size*, loading it if needed."""
    with MODEL_LOAD_LOCK:
        if model_size not in _MODEL_CACHE:
            logger.info("Loading SAM2 model: %s", model_size)
            _MODEL_CACHE[model_size] = SAM2(model_size)
        return _MODEL_CACHE[model_size]


def get_sam3_model() -> SAM3:
    """Return the cached :class:`SAM3` model, loading it on first use.

    SAM 3 is an optional extra; if it is not installed the underlying import
    raises ``ImportError``, which the handlers surface as a status message.
    """
    with MODEL_LOAD_LOCK:
        if "sam3" not in _SAM3_CACHE:
            logger.info("Loading SAM3 model")
            _SAM3_CACHE["sam3"] = SAM3()
        return _SAM3_CACHE["sam3"]
