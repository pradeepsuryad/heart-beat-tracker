"""
validate_bpm.py — Self-consistency validation of the rolling BPM estimator.

Runs the optical flow pipeline on one or more videos, collects per-frame
BPM estimates from estimate_bpm(), and reports:

    Mean ± std BPM         — central tendency and spread of the rolling estimate
    Median BPM             — robust central tendency
    CV%                    — coefficient of variation; lower = more stable signal
    Global BPM (FFT)       — dominant frequency from a single FFT over all frames
    SNR (dB)               — ratio of peak power to noise floor in the cardiac band
    Physio-valid %         — fraction of frames whose estimate falls in 50–150 BPM

Saves a two-panel figure per sequence to outputs/bpm_validation.png and prints
a Markdown results table.

Usage:
    python scripts/validate_bpm.py
    python scripts/validate_bpm.py --videos data/hamlyn_seq04.avi data/hamlyn_seq05.avi
    python scripts/validate_bpm.py --scale 0.5 --output outputs/bpm_validation.png
"""

import argparse
import collections
import sys
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import estimate_bpm, magnitude_signal


# ---------------------------------------------------------------------------
# Signal analysis helpers
# ---------------------------------------------------------------------------

def compute_global_bpm_and_snr(
    mag_sig: list,
    fps: float,
    freq_lo: float = 0.5,
    freq_hi: float = 3.5,
) -> tuple[float, float]:
    """
    Compute dominant BPM and SNR from a single FFT over the full magnitude signal.

    SNR = 10 · log10(peak_power / noise_floor)
    where noise_floor = median power of all bins in the cardiac band
    excluding the peak and its two nearest neighbours on each side.

    Returns (dominant_bpm, snr_db).  Returns (None, None) if the band is empty.
    """
    arr = np.array(mag_sig, dtype=np.float64)
    arr -= arr.mean()

    freqs = np.fft.rfftfreq(len(arr), d=1.0 / fps)
    power = np.abs(np.fft.rfft(arr)) ** 2

    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if not mask.any():
        return None, None

    band_freqs = freqs[mask]
    band_power = power[mask]

    peak_idx = int(np.argmax(band_power))
    dominant_bpm = float(band_freqs[peak_idx] * 60.0)
    peak_power = float(band_power[peak_idx])

    # Noise floor: median of band excluding ±2 bins around peak
    noise_mask = np.ones(len(band_power), dtype=bool)
    noise_mask[max(0, peak_idx - 2): peak_idx + 3] = False
    noise_floor = float(np.median(band_power[noise_mask])) if noise_mask.any() else 1e-10
    snr_db = 10.0 * np.log10(peak_power / max(noise_floor, 1e-10))

    return dominant_bpm, snr_db


def rolling_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling median smoothing."""
    out = np.empty_like(values)
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out[i] = np.median(values[lo:hi])
    return out


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def run_validation(
    video_path: str | Path,
    scale: float = 0.5,
    freq_lo: float = 0.5,
    freq_hi: float = 3.5,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list, float]:
    """
    Run the full optical flow + BPM pipeline on one video.

    Returns
    -------
    metrics     : dict of summary statistics
    valid_times : (N,) float array — timestamps of valid BPM estimates (s)
    valid_bpm   : (N,) float array — rolling BPM at each valid frame
    smoothed    : (N,) float array — rolling-median smoothed BPM
    mag_sig     : list of per-frame mean magnitude values
    fps         : video frame rate
    """
    video_path = Path(video_path)
    print(f"\n[INFO] Processing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # --- Compute per-frame magnitude signal ---
    print("[INFO] Computing optical flow magnitude signal …")
    mag_sig = magnitude_signal(video_path, scale=scale)

    # --- Replay rolling BPM estimator frame-by-frame ---
    window_len = int(fps * 10)          # 10-second rolling window
    bpm_history: collections.deque = collections.deque(maxlen=window_len)
    bpm_series: list[float | None] = []
    timestamps: list[float] = []

    for i, mag in enumerate(mag_sig):
        bpm_history.append(mag)
        bpm_series.append(estimate_bpm(bpm_history, fps, freq_lo, freq_hi))
        timestamps.append(i / fps)

    # Filter out frames where estimate_bpm returned None
    valid_pairs = [(t, b) for t, b in zip(timestamps, bpm_series) if b is not None]
    if not valid_pairs:
        raise RuntimeError(f"No valid BPM estimates produced for {video_path.name}")

    valid_times = np.array([p[0] for p in valid_pairs])
    valid_bpm   = np.array([p[1] for p in valid_pairs])

    # --- Smoothed trend for plotting ---
    smooth_win = max(1, int(fps * 5))
    smoothed = rolling_smooth(valid_bpm, smooth_win)

    # --- Global FFT metrics ---
    global_bpm, snr_db = compute_global_bpm_and_snr(mag_sig, fps, freq_lo, freq_hi)

    # --- Physiological validity (50–150 BPM covers porcine and human cardiac) ---
    physio_pct = 100.0 * float(np.mean((valid_bpm >= 50) & (valid_bpm <= 150)))

    metrics = {
        "name":        video_path.stem,
        "frames":      total,
        "duration_s":  round(total / fps, 1),
        "fps":         fps,
        "mean_bpm":    float(np.mean(valid_bpm)),
        "std_bpm":     float(np.std(valid_bpm)),
        "median_bpm":  float(np.median(valid_bpm)),
        "cv_pct":      float(100.0 * np.std(valid_bpm) / max(np.mean(valid_bpm), 1e-6)),
        "global_bpm":  global_bpm,
        "snr_db":      snr_db,
        "physio_pct":  physio_pct,
        "valid_frames": len(valid_bpm),
    }

    print(f"[OK]   Mean BPM: {metrics['mean_bpm']:.1f} ± {metrics['std_bpm']:.1f}  "
          f"| Global FFT: {global_bpm:.1f} BPM  | SNR: {snr_db:.1f} dB  "
          f"| Physio valid: {physio_pct:.1f}%")

    return metrics, valid_times, valid_bpm, smoothed, mag_sig, fps


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_validation(
    results: list[tuple],
    output_path: str | Path,
) -> None:
    """
    Save a two-panel figure per sequence:
      Left  — rolling BPM over time with smoothed trend and ±1 std band
      Right — FFT power spectrum with cardiac peak annotated
    """
    n = len(results)
    fig = plt.figure(figsize=(14, 5 * n))
    gs = gridspec.GridSpec(n, 2, figure=fig, hspace=0.45, wspace=0.32)

    for row, (metrics, times, bpm, smoothed, mag_sig, fps) in enumerate(results):
        # ── Left panel: BPM time series ─────────────────────────────────────
        ax1 = fig.add_subplot(gs[row, 0])

        std = metrics["std_bpm"]
        ax1.fill_between(times, smoothed - std, smoothed + std,
                         alpha=0.18, color="steelblue", label="±1 std")
        ax1.scatter(times, bpm, s=1.5, color="steelblue", alpha=0.35, label="Rolling BPM")
        ax1.plot(times, smoothed, color="steelblue", linewidth=2.0, label="Smoothed (5 s median)")
        ax1.axhline(metrics["mean_bpm"], color="crimson", linestyle="--", linewidth=1.3,
                    label=f"Mean {metrics['mean_bpm']:.1f} BPM")
        ax1.axhspan(50, 150, alpha=0.06, color="green")
        ax1.axhline(50,  color="grey", linestyle=":", linewidth=0.8)
        ax1.axhline(150, color="grey", linestyle=":", linewidth=0.8, label="Physio range (50–150)")

        ax1.set_xlabel("Time (s)", fontsize=11)
        ax1.set_ylabel("BPM", fontsize=11)
        ax1.set_title(f"{metrics['name']}  —  Rolling BPM estimate\n"
                      f"Mean {metrics['mean_bpm']:.1f} ± {metrics['std_bpm']:.1f} BPM  |  "
                      f"CV {metrics['cv_pct']:.1f}%  |  Physio valid {metrics['physio_pct']:.1f}%",
                      fontsize=10)
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(True, alpha=0.3)

        # ── Right panel: FFT power spectrum ─────────────────────────────────
        ax2 = fig.add_subplot(gs[row, 1])

        arr = np.array(mag_sig, dtype=np.float64)
        arr -= arr.mean()
        freqs = np.fft.rfftfreq(len(arr), d=1.0 / fps)
        power = np.abs(np.fft.rfft(arr)) ** 2

        # Show 0.3–5 Hz range (18–300 BPM) for context
        band = (freqs >= 0.3) & (freqs <= 5.0)
        bpm_axis = freqs[band] * 60.0

        ax2.plot(bpm_axis, power[band], color="darkorange", linewidth=1.4)
        ax2.axvline(metrics["global_bpm"], color="crimson", linestyle="--", linewidth=1.5,
                    label=f"Peak {metrics['global_bpm']:.1f} BPM")
        ax2.axvspan(50, 150, alpha=0.08, color="green", label="Physio range")

        ax2.set_xlabel("Frequency (BPM)", fontsize=11)
        ax2.set_ylabel("Power", fontsize=11)
        ax2.set_title(f"{metrics['name']}  —  FFT power spectrum\n"
                      f"Global BPM {metrics['global_bpm']:.1f}  |  SNR {metrics['snr_db']:.1f} dB",
                      fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.suptitle("BPM Self-Consistency Validation — Hamlyn Cardiac Sequences",
                 fontsize=13, y=1.01)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Figure saved: {out.resolve()}")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def build_markdown_table(all_metrics: list[dict]) -> str:
    header = (
        "| Sequence | Frames | Duration (s) | Mean BPM | Std BPM | "
        "Median BPM | CV (%) | Global BPM (FFT) | SNR (dB) | Physio valid (%) |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for m in all_metrics:
        rows.append(
            f"| {m['name']} | {m['frames']} | {m['duration_s']} | "
            f"{m['mean_bpm']:.1f} | {m['std_bpm']:.1f} | {m['median_bpm']:.1f} | "
            f"{m['cv_pct']:.1f} | {m['global_bpm']:.1f} | {m['snr_db']:.1f} | "
            f"{m['physio_pct']:.1f} |"
        )
    return header + "\n".join(rows)


def save_results_md(all_metrics: list[dict], path: str | Path) -> None:
    """Write results/validation_table.md."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table = build_markdown_table(all_metrics)
    content = f"""# BPM Validation Results

## Method

No synchronised ECG ground truth is available in the publicly released
Hamlyn HuggingFace mirror. Validation is therefore conducted as a
**self-consistency analysis** of the rolling FFT BPM estimator.

Metrics reported:

| Metric | Definition |
|---|---|
| **Mean ± Std BPM** | Central tendency and variability of the rolling estimate |
| **Median BPM** | Robust central tendency (insensitive to outliers) |
| **CV (%)** | Coefficient of variation = 100 × std / mean; lower = more stable |
| **Global BPM (FFT)** | Dominant frequency from a single FFT over all frames × 60 |
| **SNR (dB)** | 10 log₁₀(peak power / noise-floor power) in the 0.5–3.5 Hz band |
| **Physio valid (%)** | Fraction of frames with estimate in the physiological range 50–150 BPM |

## Results

{table}

## Notes

- Scale factor 0.5 applied before flow computation (half resolution).
- Rolling window: 10 seconds (250 frames at 25 fps).
- Physiological range: 50–150 BPM (covers ex-vivo porcine cadaver stimulation rates).
- SNR computed on the full-video magnitude signal FFT.
"""
    Path(path).write_text(content, encoding="utf-8")
    print(f"[OK] Results table saved: {Path(path).resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BPM self-consistency validation for the endoscopic heart-beat tracker."
    )
    parser.add_argument(
        "--videos", nargs="+",
        default=["data/hamlyn_seq04.avi", "data/hamlyn_seq05.avi"],
        help="Video paths to validate (default: Hamlyn seq04 and seq05)",
    )
    parser.add_argument("--scale",  type=float, default=0.5,
                        help="Frame resize factor (default 0.5)")
    parser.add_argument("--output", default="outputs/bpm_validation.png",
                        help="Path for the output figure")
    parser.add_argument("--table",  default="results/validation_table.md",
                        help="Path for the Markdown results table")
    args = parser.parse_args()

    all_results: list[tuple] = []
    all_metrics: list[dict]  = []

    for vpath in args.videos:
        p = Path(vpath)
        if not p.exists():
            print(f"[SKIP] Not found: {p}")
            continue
        metrics, times, bpm, smoothed, mag_sig, fps = run_validation(
            p, scale=args.scale,
        )
        all_results.append((metrics, times, bpm, smoothed, mag_sig, fps))
        all_metrics.append(metrics)

    if not all_results:
        sys.exit("[ERROR] No valid videos processed.")

    plot_validation(all_results, args.output)
    save_results_md(all_metrics, args.table)

    print("\n" + "=" * 60)
    print(build_markdown_table(all_metrics))
    print("=" * 60)


if __name__ == "__main__":
    main()
