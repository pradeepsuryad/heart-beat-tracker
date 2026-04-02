"""
utils.py — Video I/O helpers for the endoscopic heart-beat tracker.

    video_info(path)          → dict of metadata (fps, size, frame count, …)
    iter_frames(path, scale)  → generator yielding (index, bgr_frame)
    extract_frames(path, out_dir, step, scale)
                              → save every nth frame as PNG
    magnitude_signal(path, scale)
                              → list of per-frame mean flow magnitudes
                                (useful for plotting the heartbeat signal)
"""

import sys
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def video_info(path: str | Path) -> dict:
    """
    Return a dict of basic video metadata.

    Keys
    ----
    path         : resolved absolute path (str)
    width        : frame width  (px)
    height       : frame height (px)
    fps          : frames per second (float)
    frame_count  : total frames reported by the container (int, may be 0 for
                   some .avi files — use iter_frames to count precisely)
    duration_s   : estimated duration in seconds (float)
    codec        : four-character codec string (e.g. 'XVID', 'H264')

    Raises SystemExit if the file cannot be opened.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {path}")

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    info = {
        "path": str(Path(path).resolve()),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": round(frame_count / fps, 2) if fps else 0.0,
        "codec": codec,
    }
    cap.release()
    return info


# ---------------------------------------------------------------------------
# Frame iteration
# ---------------------------------------------------------------------------

def iter_frames(
    path: str | Path,
    scale: float = 1.0,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, bgr_frame) for every frame in the video.

    Parameters
    ----------
    path  : video file path
    scale : optional resize factor (e.g. 0.5 = half resolution)

    Yields
    ------
    (int, np.ndarray)  — zero-based frame index and BGR uint8 frame
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            h = max(1, int(frame.shape[0] * scale))
            w = max(1, int(frame.shape[1] * scale))
            frame = cv2.resize(frame, (w, h))
        yield idx, frame
        idx += 1

    cap.release()


# ---------------------------------------------------------------------------
# Frame extraction to disk
# ---------------------------------------------------------------------------

def extract_frames(
    path: str | Path,
    out_dir: str | Path,
    step: int = 1,
    scale: float = 1.0,
    fmt: str = "png",
) -> list[Path]:
    """
    Save every `step`-th frame from a video as an image file.

    Parameters
    ----------
    path    : source video path
    out_dir : directory to write images into (created if absent)
    step    : save one frame every `step` frames (1 = every frame)
    scale   : resize factor applied before saving
    fmt     : image format — 'png' or 'jpg'

    Returns
    -------
    List of paths to saved images, in frame order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for idx, frame in iter_frames(path, scale=scale):
        if idx % step != 0:
            continue
        out_path = out_dir / f"frame_{idx:06d}.{fmt}"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)

    print(f"[OK] Saved {len(saved)} frames → {out_dir}")
    return saved


# ---------------------------------------------------------------------------
# Per-frame magnitude signal (heartbeat signal extraction)
# ---------------------------------------------------------------------------

def magnitude_signal(
    path: str | Path,
    scale: float = 0.5,
) -> list[float]:
    """
    Compute the mean optical flow magnitude for every consecutive frame pair.

    This produces a 1-D time series that oscillates with the heartbeat:
    high values during systole (peak contraction), low during diastole.
    Useful for plotting, phase detection, or FFT-based heart rate estimation.

    Parameters
    ----------
    path  : source video path
    scale : resize factor applied before flow computation (0.5 recommended
            for speed — the signal shape is resolution-independent)

    Returns
    -------
    List of float, length = (num_frames - 1), one value per frame transition.
    """
    from tracker import compute_flow   # local import to avoid circular dependency

    signal: list[float] = []
    prev_gray: np.ndarray | None = None

    for idx, frame in iter_frames(path, scale=scale):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            mag, _, _flow = compute_flow(prev_gray, gray)
            signal.append(float(mag.mean()))
        prev_gray = gray

        if (idx + 1) % 100 == 0:
            print(f"\r  magnitude_signal: {idx + 1} frames", end="", flush=True)

    if signal:
        print()

    return signal


# ---------------------------------------------------------------------------
# Rolling BPM estimator
# ---------------------------------------------------------------------------

def estimate_bpm(
    mag_signal: "collections.deque[float]",
    fps: float,
    freq_lo: float = 0.5,
    freq_hi: float = 3.5,
) -> float | None:
    """
    Estimate heart rate in BPM from a short rolling window of mean-magnitude values.

    Uses a real FFT on the deque contents and returns the dominant frequency
    within the physiological range [freq_lo, freq_hi] Hz (30–210 bpm).

    Parameters
    ----------
    mag_signal : rolling deque of recent per-frame mean flow magnitudes
    fps        : video frame rate (needed to convert FFT bins to Hz)
    freq_lo    : lower bound for heart rate search in Hz (default 0.5 = 30 bpm)
    freq_hi    : upper bound for heart rate search in Hz (default 3.5 = 210 bpm)

    Returns
    -------
    BPM as a float, or None if the window is too short to produce a reliable estimate
    (need at least 2 full heartbeat cycles → 2 / freq_lo seconds of data).
    """
    import collections as _collections
    min_frames = int(2.0 / freq_lo * fps)   # need ≥2 full cycles
    if len(mag_signal) < min_frames:
        return None

    arr = np.array(mag_signal, dtype=np.float32)
    arr -= arr.mean()   # detrend

    freqs = np.fft.rfftfreq(len(arr), d=1.0 / fps)
    power = np.abs(np.fft.rfft(arr))

    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if not mask.any():
        return None

    dominant_hz = float(freqs[mask][np.argmax(power[mask])])
    return round(dominant_hz * 60.0, 1)


import collections as _col   # noqa: E402 — needed for type hint above at module level
