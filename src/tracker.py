"""
tracker.py — Dense optical flow tracker using the Farnebäck algorithm.

Clinical context:
    During minimally invasive cardiac surgery, the surgeon views the heart
    through an endoscope. The beating heart moves continuously, making it
    difficult to precisely target tissue. Tracking the dense motion field
    (optical flow) gives a per-pixel velocity estimate that can drive
    motion compensation, tool guidance, or phase detection.

Algorithm — Farnebäck dense optical flow (cv2.calcOpticalFlowFarneback):
    For consecutive frames I₁ and I₂, the algorithm fits a polynomial
    expansion to the neighbourhood of each pixel, then estimates the
    displacement (dx, dy) that best aligns those expansions. The result is
    a 2-channel array flow[y, x] = (dx, dy) for every pixel.

    From (dx, dy) we derive:
        magnitude = √(dx² + dy²)  — how far the pixel moved (pixels/frame)
        angle     = atan2(dy, dx) — direction of motion in radians

Usage (CLI):
    python src/tracker.py --input video.mp4 --output outputs/result.mp4

    Optional flags:
        --scale   FLOAT   resize factor applied before processing (default 1.0)
        --fps     INT     override output FPS (default: match input)
"""

import argparse
import collections
import sys
from pathlib import Path

import cv2
import numpy as np

import visualizer
from visualizer import _PHASE_HISTORY_LEN
from utils import estimate_bpm


# ---------------------------------------------------------------------------
# Core flow computation
# ---------------------------------------------------------------------------

def compute_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Farnebäck dense optical flow between two grayscale frames.

    Parameters
    ----------
    prev_gray : (H, W) uint8   — previous frame in grayscale
    curr_gray : (H, W) uint8   — current  frame in grayscale

    Returns
    -------
    magnitude : (H, W) float32 — per-pixel displacement magnitude (pixels)
    angle     : (H, W) float32 — per-pixel displacement direction (radians,
                                  range 0..2π)

    Farnebäck parameter guide
    -------------------------
    pyr_scale  = 0.5   — each pyramid level is half the size of the previous
    levels     = 3     — number of pyramid levels (more = larger motion range)
    winsize    = 15    — averaging window size; larger = smoother but blurrier
    iterations = 3     — passes per pyramid level
    poly_n     = 5     — neighbourhood size for polynomial expansion (5 or 7)
    poly_sigma = 1.2   — Gaussian std for polynomial expansion weights
    flags      = 0     — no special flags
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    # flow shape: (H, W, 2)  where channel 0 = dx, channel 1 = dy

    # --- Convert Cartesian (dx, dy) → polar (magnitude, angle) ---
    # cv2.cartToPolar returns angle in [0, 2π) when angleInDegrees=False
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)

    return magnitude, angle, flow


# ---------------------------------------------------------------------------
# HSV colour-map visualisation
# ---------------------------------------------------------------------------

def flow_to_hsv_frame(magnitude: np.ndarray, angle: np.ndarray, prev_bgr: np.ndarray) -> np.ndarray:
    """
    Encode flow as an HSV overlay blended onto the original frame.

    Encoding scheme
    ---------------
    Hue   (H) ← angle mapped from [0, 2π] → [0, 179]  (OpenCV hue range)
                 so colour represents direction of motion
    Saturation  fixed at 255 (fully saturated)
    Value (V) ← magnitude normalised to [0, 255]
                 so brightness represents speed

    The resulting HSV image is converted to BGR and alpha-blended with the
    source frame so anatomical structure remains visible.
    """
    h, w = magnitude.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # Hue: map angle [0, 2π] linearly to OpenCV's [0, 179]
    hsv[..., 0] = (angle * 179 / (2 * np.pi)).astype(np.uint8)

    # Saturation: constant — every flowing pixel is fully saturated
    hsv[..., 1] = 255

    # Value: normalise magnitude to [0, 255] so the brightest pixel = max speed
    max_mag = magnitude.max()
    if max_mag > 0:
        # Scale to 0-255; sqrt compresses large values for better perceptual range
        hsv[..., 2] = (np.sqrt(magnitude / max_mag) * 255).astype(np.uint8)
    # else: all zeros → black (no motion)

    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Alpha-blend: 60% original anatomy, 40% flow colour map
    annotated = cv2.addWeighted(prev_bgr, 0.6, flow_bgr, 0.4, 0)

    return annotated


# ---------------------------------------------------------------------------
# Per-frame summary overlay
# ---------------------------------------------------------------------------

def draw_stats(
    frame: np.ndarray,
    frame_idx: int,
    mean_mag: float,
    max_mag: float,
    bpm: float | None = None,
) -> None:
    """
    Burn frame number, motion statistics, and optional BPM into the top-left corner.
    Modifies `frame` in-place.
    """
    lines = [
        f"Frame: {frame_idx}",
        f"Mean disp: {mean_mag:.2f} px",
        f"Max  disp: {max_mag:.2f} px",
    ]
    if bpm is not None:
        lines.append(f"HR: {bpm:.0f} bpm")

    y0 = 14
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (5, y0 + i * 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (4, y0 - 1 + i * 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video(
    input_path: str | Path,
    output_path: str | Path,
    scale: float = 1.0,
    fps_override: int | None = None,
    overlays: bool = True,
) -> None:
    """
    Read an endoscopic video, compute dense optical flow for every consecutive
    frame pair, and write an annotated output video.

    Parameters
    ----------
    input_path   : path to source video (mp4, avi, …)
    output_path  : path for the annotated output video
    scale        : resize factor applied before processing (0 < scale ≤ 1.0
                   recommended for speed; e.g. 0.5 = half resolution)
    fps_override : if given, write output at this FPS instead of source FPS
    overlays     : if True (default), burn in colorbar, motion vectors,
                   phase indicator (SYSTOLE / DIASTOLE), and strain map
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {input_path}")

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = max(1, int(src_w * scale))
    out_h = max(1, int(src_h * scale))
    out_fps   = fps_override or src_fps

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        sys.exit(f"[ERROR] Cannot open VideoWriter for: {output_path}")

    print(f"[INFO] Input : {input_path}  ({src_w}×{src_h} @ {src_fps:.1f} fps, {total} frames)")
    print(f"[INFO] Output: {output_path}  ({out_w}×{out_h} @ {out_fps:.1f} fps)")

    # --- Read and resize the first frame ---
    ret, prev_bgr = cap.read()
    if not ret:
        cap.release()
        writer.release()
        sys.exit("[ERROR] Video has no readable frames.")

    if scale != 1.0:
        prev_bgr = cv2.resize(prev_bgr, (out_w, out_h))

    # Convert to grayscale for the flow algorithm (colour info is discarded;
    # Farnebäck operates on intensity gradients only)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)

    # Write the first frame as-is (no flow yet for frame 0)
    writer.write(prev_bgr)

    phase_history: collections.deque[float] = collections.deque(maxlen=_PHASE_HISTORY_LEN)
    strain_history: collections.deque[np.ndarray] = collections.deque(maxlen=max(1, int(src_fps)))
    bpm_history: collections.deque[float] = collections.deque(maxlen=int(src_fps * 10))
    frame_idx = 1
    while True:
        ret, curr_bgr = cap.read()
        if not ret:
            break  # end of video

        if scale != 1.0:
            curr_bgr = cv2.resize(curr_bgr, (out_w, out_h))

        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        # --- Compute per-pixel flow ---
        magnitude, angle, flow_raw = compute_flow(prev_gray, curr_gray)

        # --- Use original frame as base (no HSV colour overlay) ---
        annotated = curr_bgr.copy()

        # --- Rolling BPM estimate ---
        mean_mag = float(magnitude.mean())
        bpm_history.append(mean_mag)
        bpm = estimate_bpm(bpm_history, src_fps)

        # --- Overlay stats ---
        draw_stats(annotated, frame_idx, mean_mag, float(magnitude.max()), bpm)

        # --- Optional visual overlays ---
        if overlays:
            strain_history.append(magnitude)
            visualizer.draw_strain_map(annotated, strain_history)
            visualizer.draw_motion_vectors(annotated, flow_raw)
            visualizer.draw_phase_indicator(annotated, mean_mag, phase_history)
            visualizer.draw_colorbar(annotated)

        writer.write(annotated)

        # Advance state
        prev_gray = curr_gray
        frame_idx += 1

        if frame_idx % 50 == 0:
            pct = 100 * frame_idx / max(total, 1)
            print(f"[INFO] {frame_idx}/{total} frames processed ({pct:.1f}%)")

    cap.release()
    writer.release()
    print(f"[DONE] Wrote {frame_idx} frames → {output_path}")


# ---------------------------------------------------------------------------
# Live webcam mode
# ---------------------------------------------------------------------------

def run_live(
    camera: int = 0,
    output_path: str | Path | None = None,
    scale: float = 1.0,
    overlays: bool = True,
) -> None:
    """
    Run the tracker in real-time from a webcam or RTSP stream.

    Displays an annotated window.  Press Q to quit.

    Parameters
    ----------
    camera      : OpenCV camera index (0 = default webcam) or RTSP URL string
    output_path : optional path to save the annotated stream as a video file
    scale       : resize factor applied to each frame before processing
    overlays    : enable strain map, phase indicator, BPM, colorbar, vectors
    """
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera: {camera}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w   = max(1, int(src_w * scale))
    out_h   = max(1, int(src_h * scale))

    writer = None
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (out_w, out_h))

    print(f"[INFO] Live mode — camera {camera}  ({src_w}×{src_h} @ {src_fps:.0f} fps)")
    print("[INFO] Press Q to quit.")

    ret, prev_bgr = cap.read()
    if not ret:
        cap.release()
        sys.exit("[ERROR] Could not read from camera.")
    if scale != 1.0:
        prev_bgr = cv2.resize(prev_bgr, (out_w, out_h))
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)

    phase_history: collections.deque[float] = collections.deque(maxlen=_PHASE_HISTORY_LEN)
    strain_history: collections.deque[np.ndarray] = collections.deque(maxlen=max(1, int(src_fps)))
    bpm_history: collections.deque[float] = collections.deque(maxlen=int(src_fps * 10))
    frame_idx = 1

    while True:
        ret, curr_bgr = cap.read()
        if not ret:
            break

        if scale != 1.0:
            curr_bgr = cv2.resize(curr_bgr, (out_w, out_h))
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        magnitude, angle, flow_raw = compute_flow(prev_gray, curr_gray)
        annotated = curr_bgr.copy()

        mean_mag = float(magnitude.mean())
        bpm_history.append(mean_mag)
        bpm = estimate_bpm(bpm_history, src_fps)

        draw_stats(annotated, frame_idx, mean_mag, float(magnitude.max()), bpm)

        if overlays:
            strain_history.append(magnitude)
            visualizer.draw_strain_map(annotated, strain_history)
            visualizer.draw_motion_vectors(annotated, flow_raw)
            visualizer.draw_phase_indicator(annotated, mean_mag, phase_history)
            visualizer.draw_colorbar(annotated)

        cv2.imshow("Heart Beat Tracker  [Q to quit]", annotated)

        if writer is not None:
            writer.write(annotated)

        prev_gray = curr_gray
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"[DONE] Wrote {frame_idx} frames → {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Endoscopic heart-beat tracker via Farnebäck dense optical flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  default=None, help="Path to input endoscopic video (batch mode)")
    parser.add_argument("--output", default=None, help="Path for annotated output video")
    parser.add_argument("--live",   action="store_true", help="Enable live webcam mode")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index for live mode (default: 0)")
    parser.add_argument("--scale",  type=float, default=1.0,
                        help="Frame resize factor before processing (e.g. 0.5 = half-res)")
    parser.add_argument("--fps",    type=int,   default=None,
                        help="Override output FPS (default: match input)")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Disable all visual overlays")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.live:
        run_live(
            camera=args.camera,
            output_path=args.output,
            scale=args.scale,
            overlays=not args.no_overlays,
        )
    else:
        if args.input is None:
            sys.exit("[ERROR] --input is required in batch mode (or use --live)")
        if args.output is None:
            sys.exit("[ERROR] --output is required in batch mode (or use --live)")
        process_video(
            input_path=args.input,
            output_path=args.output,
            scale=args.scale,
            fps_override=args.fps,
            overlays=not args.no_overlays,
        )
