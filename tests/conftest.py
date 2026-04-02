"""
Shared pytest fixtures for heart-beat-tracker tests.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture()
def gray_frame():
    """64×64 uint8 grayscale frame with random texture (Farnebäck needs gradients)."""
    rng = np.random.default_rng(42)
    return rng.integers(40, 220, (64, 64), dtype=np.uint8)


@pytest.fixture()
def bgr_frame():
    """64×64 uint8 BGR frame (mid-grey)."""
    return np.full((64, 64, 3), 100, dtype=np.uint8)


@pytest.fixture()
def shifted_frame(gray_frame):
    """gray_frame shifted 3 pixels to the right — produces measurable flow."""
    return np.roll(gray_frame, shift=3, axis=1)


@pytest.fixture()
def small_video_path():
    """
    Write a tiny 20-frame 64×64 video to a temp file and return its path.
    Frames alternate between two patterns to produce non-zero optical flow.
    """
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
        path = Path(f.name)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (64, 64))

    rng = np.random.default_rng(0)
    base = rng.integers(40, 220, (64, 64, 3), dtype=np.uint8)
    for i in range(20):
        # Shift the textured base left/right each frame — produces real optical flow
        shift = 4 if i % 2 == 0 else -4
        frame = np.roll(base, shift, axis=1).astype(np.uint8)
        writer.write(frame)

    writer.release()
    yield path
    path.unlink(missing_ok=True)
