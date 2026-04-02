"""
visualizer.py — Overlay helpers for the endoscopic heart-beat tracker.

Provides four composable overlays:

    draw_colorbar(frame)
        Renders a circular HSV direction legend in the bottom-right corner
        so the viewer knows which colour maps to which motion direction.

    draw_motion_vectors(frame, flow, step, scale)
        Samples the dense flow field on a regular grid and draws arrows
        showing local motion direction and magnitude.

    draw_phase_indicator(frame, mean_mag, history, threshold)
        Maintains a short history of mean-magnitude values and labels the
        current frame as SYSTOLE (peak motion) or DIASTOLE (low motion).

All functions operate on BGR uint8 frames and modify them in-place.
"""

import collections
from typing import Deque

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Direction colorbar
# ---------------------------------------------------------------------------

def draw_colorbar(
    frame: np.ndarray,
    radius: int = 22,
    margin: int = 8,
) -> None:
    """
    Draw a circular HSV direction legend in the bottom-right corner.

    The wheel shows hue (= motion direction) at full saturation and value so
    the surgeon or engineer can immediately read off which colour means which
    direction.  A small cross marks the centre (= zero motion = black).

    Parameters
    ----------
    frame   : BGR uint8 image to annotate (modified in-place)
    radius  : radius of the colour wheel in pixels
    margin  : gap from the frame edge in pixels
    """
    h, w = frame.shape[:2]
    cx = w - margin - radius
    cy = h - margin - radius
    size = radius * 2 + 1

    # Build the colour wheel as a small square patch
    ys, xs = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    dist = np.sqrt(xs ** 2 + ys ** 2)
    mask = dist <= radius

    angle_map = np.arctan2(ys, xs) % (2 * np.pi)          # [0, 2π)
    hue = (angle_map * 179 / (2 * np.pi)).astype(np.uint8)
    sat = np.full((size, size), 255, dtype=np.uint8)
    val = np.where(mask, 220, 0).astype(np.uint8)          # dim outside circle

    wheel_hsv = np.stack([hue, sat, val], axis=-1)
    wheel_bgr = cv2.cvtColor(wheel_hsv, cv2.COLOR_HSV2BGR)

    # Paste onto frame with a dark circular border
    x0 = cx - radius
    y0 = cy - radius
    roi = frame[y0:y0 + size, x0:x0 + size]

    # Semi-transparent blend: 80% wheel, 20% original inside circle
    blended = cv2.addWeighted(roi, 0.2, wheel_bgr, 0.8, 0)
    blended[~mask] = roi[~mask]                            # keep outside pixels
    frame[y0:y0 + size, x0:x0 + size] = blended

    # Border circle + centre cross
    cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.drawMarker(frame, (cx, cy), (200, 200, 200),
                   cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

    # Cardinal direction labels
    offset = radius + 10
    labels = [("R", (cx + offset, cy + 5)),
              ("L", (cx - offset - 8, cy + 5)),
              ("D", (cx - 4, cy + offset + 12)),
              ("U", (cx - 4, cy - offset))]
    for text, pos in labels:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Motion vector arrows
# ---------------------------------------------------------------------------

def draw_motion_vectors(
    frame: np.ndarray,
    flow: np.ndarray,
    step: int = 20,
    scale: float = 3.0,
    min_mag: float = 0.5,
    color: tuple[int, int, int] = (0, 255, 255),
) -> None:
    """
    Draw a sparse grid of motion arrows on the frame.

    Parameters
    ----------
    frame   : BGR uint8 image (modified in-place)
    flow    : (H, W, 2) float32 array — raw (dx, dy) from cv2.calcOpticalFlowFarneback
    step    : grid spacing in pixels between arrow origins
    scale   : multiply displacement magnitude for visibility
    min_mag : skip arrows shorter than this many pixels (reduces clutter)
    color   : BGR color for arrows
    """
    h, w = frame.shape[:2]
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)

    for y in ys:
        for x in xs:
            dx, dy = flow[y, x]
            mag = np.sqrt(dx * dx + dy * dy)
            if mag < min_mag:
                continue
            end_x = int(x + dx * scale)
            end_y = int(y + dy * scale)
            cv2.arrowedLine(
                frame,
                (x, y),
                (end_x, end_y),
                color,
                thickness=1,
                line_type=cv2.LINE_AA,
                tipLength=0.3,
            )


# ---------------------------------------------------------------------------
# Cardiac phase indicator  (peak-detection based)
# ---------------------------------------------------------------------------

_PHASE_HISTORY_LEN = 30   # ~1 second at 25 fps — enough to see one full beat


def draw_phase_indicator(
    frame: np.ndarray,
    mean_mag: float,
    history: "Deque[float]",
) -> str:
    """
    Label the frame as SYSTOLE or DIASTOLE using derivative-based peak detection.

    Strategy (works in real-time with no look-ahead):
        1. Append the current magnitude to a fixed-length history window.
        2. Smooth the window with a short Gaussian-weighted average.
        3. Compute the first-order difference of the smoothed signal.
        4. If the most recent derivative flipped from positive to negative
           (i.e. we just passed a local maximum) → SYSTOLE peak.
           Otherwise, classify by position relative to the window median:
               above median → rising toward SYSTOLE
               below median → DIASTOLE

    SYSTOLE  → red label (top-right)
    DIASTOLE → green label (top-right)

    Parameters
    ----------
    frame    : BGR uint8 image (modified in-place)
    mean_mag : mean optical flow magnitude for the current frame
    history  : deque(maxlen=_PHASE_HISTORY_LEN) maintained across frames

    Returns
    -------
    phase : "SYSTOLE" or "DIASTOLE"
    """
    history.append(mean_mag)

    if len(history) < 4:
        phase = "DIASTOLE"
    else:
        arr = np.array(history, dtype=np.float32)

        # Gaussian-weighted smooth over last min(10, len) samples
        win = min(10, len(arr))
        weights = np.exp(-0.5 * np.linspace(-2, 2, win) ** 2)
        weights /= weights.sum()
        smoothed = np.convolve(arr, weights, mode="same")

        # First-order derivative of smoothed signal
        diff = np.diff(smoothed)

        # Peak = derivative went from + to - in last two steps
        at_peak = len(diff) >= 2 and diff[-2] > 0 and diff[-1] <= 0

        # Otherwise use position relative to window median
        above_median = float(arr[-1]) >= float(np.median(arr))

        phase = "SYSTOLE" if (at_peak or above_median) else "DIASTOLE"

    color = (0, 60, 220) if phase == "SYSTOLE" else (60, 180, 60)

    h, w = frame.shape[:2]
    text_size, _ = cv2.getTextSize(phase, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    x = w - text_size[0] - 6
    y = 14

    cv2.putText(frame, phase, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, phase, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, color, 1, cv2.LINE_AA)

    return phase


# ---------------------------------------------------------------------------
# Strain map
# ---------------------------------------------------------------------------

def draw_strain_map(
    frame: np.ndarray,
    mag_history: "Deque[np.ndarray]",
    alpha: float = 0.45,
) -> None:
    """
    Overlay a cumulative strain map showing regions of sustained high motion.

    Strain is approximated as the per-pixel mean of recent flow magnitudes.
    High-strain regions (thick, repeatedly moving tissue) appear warm (red);
    low-strain / static regions appear cool (blue) or invisible.

    Parameters
    ----------
    frame       : BGR uint8 image (modified in-place)
    mag_history : deque of (H, W) float32 magnitude arrays — one per recent
                  frame.  Initialise as collections.deque(maxlen=N) and append
                  the magnitude array from compute_flow() each frame.
                  Recommended maxlen: fps (≈1 second window).
    alpha       : blend weight for the strain overlay (0 = invisible, 1 = opaque)
    """
    if not mag_history:
        return

    h, w = frame.shape[:2]

    # Stack and average recent magnitude frames
    stack = np.stack([
        cv2.resize(m, (w, h)) if m.shape[:2] != (h, w) else m
        for m in mag_history
    ], axis=0)
    mean_strain = stack.mean(axis=0)   # (H, W) float32

    # Normalise to [0, 255]
    max_val = mean_strain.max()
    if max_val < 1e-6:
        return
    norm = (mean_strain / max_val * 255).astype(np.uint8)

    # Apply JET colormap: blue=low, green=mid, red=high strain
    strain_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    # Mask: only show overlay where strain is meaningfully above noise
    threshold = 255 * 0.15
    mask = (norm > threshold).astype(np.float32)[..., np.newaxis]

    # Blend only in high-strain regions
    blended = cv2.addWeighted(frame, 1 - alpha, strain_color, alpha, 0)
    frame[:] = (blended * mask + frame * (1 - mask)).astype(np.uint8)
