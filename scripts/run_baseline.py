"""
run_baseline.py  —  Naive frame-differencing ablation baseline.

Compares heartbeat signal quality between two methods on the same videos:

  Method A  (Farneback)  —  cv2.calcOpticalFlowFarneback mean magnitude
  Method B  (Baseline)   —  simple cv2.absdiff mean pixel difference

Reports Global BPM (FFT) and SNR (dB) for both methods side-by-side so that
the delta in SNR justifies the computational cost of dense optical flow.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --videos data/hamlyn_seq04.avi data/hamlyn_seq05.avi
    python scripts/run_baseline.py --scale 0.5 --output outputs/baseline_comparison.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Make src/ and scripts/ importable
_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo / "scripts"))

from utils import iter_frames, magnitude_signal
from validate_bpm import compute_global_bpm_and_snr


# ---------------------------------------------------------------------------
# Baseline signal (frame differencing)
# ---------------------------------------------------------------------------

def differencing_signal(path: str | Path, scale: float = 0.5) -> list[float]:
    """
    Extract heartbeat signal using naive absolute frame differencing.

    Returns a list of per-frame mean |I_t - I_{t-1}| values (one per frame
    transition), analogous to magnitude_signal() but without optical flow.
    """
    signal: list[float] = []
    prev_gray = None

    for idx, frame in iter_frames(path, scale=scale):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            signal.append(float(diff.mean()))
        prev_gray = gray

        if (idx + 1) % 100 == 0:
            print(f"\r  differencing_signal: {idx + 1} frames", end="", flush=True)

    if signal:
        print()
    return signal


# ---------------------------------------------------------------------------
# Per-video comparison
# ---------------------------------------------------------------------------

def compare_video(
    video_path: str | Path,
    scale: float = 0.5,
    fps_override: float | None = None,
) -> dict:
    """
    Run both methods on one video and return a metrics dict.

    Keys
    ----
    name, fps,
    farneback_bpm, farneback_snr,
    baseline_bpm,  baseline_snr,
    snr_delta_db   (Farneback - baseline, positive = Farneback wins)
    """
    p = Path(video_path)
    print(f"\n[INFO] Processing: {p.name}")

    cap = cv2.VideoCapture(str(p))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # Method A: Farneback
    print("[INFO] Method A — Farneback optical flow ...")
    fb_sig = magnitude_signal(p, scale=scale)
    fb_bpm, fb_snr = compute_global_bpm_and_snr(fb_sig, fps)

    # Method B: frame differencing
    print("[INFO] Method B — frame differencing ...")
    fd_sig = differencing_signal(p, scale=scale)
    fd_bpm, fd_snr = compute_global_bpm_and_snr(fd_sig, fps)

    delta = (fb_snr - fd_snr) if (fb_snr is not None and fd_snr is not None) else None

    print(
        f"[OK]  Farneback: {fb_bpm:.1f} BPM, SNR {fb_snr:.1f} dB  |  "
        f"Differencing: {fd_bpm:.1f} BPM, SNR {fd_snr:.1f} dB  |  "
        f"delta SNR: {delta:+.1f} dB"
    )

    return {
        "name":           p.stem,
        "fps":            fps,
        "farneback_bpm":  fb_bpm,
        "farneback_snr":  fb_snr,
        "baseline_bpm":   fd_bpm,
        "baseline_snr":   fd_snr,
        "snr_delta_db":   delta,
        "fb_sig":         fb_sig,
        "fd_sig":         fd_sig,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(results: list[dict], output_path: str | Path) -> None:
    """
    Save a two-panel figure per sequence comparing FFT spectra of both methods.
    Left panel: Farneback.  Right panel: frame differencing.
    """
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.5 * n))
    if n == 1:
        axes = [axes]

    for row, r in enumerate(results):
        fps = r["fps"]
        for col, (sig, label, bpm, snr) in enumerate([
            (r["fb_sig"], "Farneback optical flow", r["farneback_bpm"], r["farneback_snr"]),
            (r["fd_sig"], "Frame differencing",     r["baseline_bpm"],  r["baseline_snr"]),
        ]):
            ax = axes[row][col]
            arr = np.array(sig, dtype=np.float64)
            arr -= arr.mean()
            freqs = np.fft.rfftfreq(len(arr), d=1.0 / fps)
            power = np.abs(np.fft.rfft(arr)) ** 2
            band = (freqs >= 0.3) & (freqs <= 5.0)
            bpm_axis = freqs[band] * 60.0

            color = "steelblue" if col == 0 else "darkorange"
            ax.plot(bpm_axis, power[band], color=color, linewidth=1.3)
            ax.axvline(bpm, color="crimson", linestyle="--", linewidth=1.5,
                       label=f"Peak {bpm:.1f} BPM")
            ax.axvspan(50, 150, alpha=0.07, color="green", label="Physio range")
            ax.set_xlabel("Frequency (BPM)", fontsize=10)
            ax.set_ylabel("Power", fontsize=10)
            ax.set_title(
                f"{r['name']}  —  {label}\nSNR {snr:.1f} dB  |  BPM {bpm:.1f}",
                fontsize=10,
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Ablation: Farneback optical flow vs. naive frame differencing\n"
        "(higher SNR = cleaner heartbeat signal)",
        fontsize=12, y=1.02,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Figure saved: {out.resolve()}")


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    header = (
        "\n| Sequence | Farneback BPM | Farneback SNR (dB) "
        "| Differencing BPM | Differencing SNR (dB) | Delta SNR (dB) |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r['name']} "
            f"| {r['farneback_bpm']:.1f} | {r['farneback_snr']:.1f} "
            f"| {r['baseline_bpm']:.1f}  | {r['baseline_snr']:.1f} "
            f"| {r['snr_delta_db']:+.1f} |"
        )
    print(header + "\n".join(rows))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: Farneback optical flow vs. frame-differencing baseline."
    )
    parser.add_argument(
        "--videos", nargs="+",
        default=["data/hamlyn_seq04.avi", "data/hamlyn_seq05.avi"],
    )
    parser.add_argument("--scale",  type=float, default=0.5)
    parser.add_argument("--output", default="outputs/baseline_comparison.png")
    args = parser.parse_args()

    all_results = []
    for vpath in args.videos:
        p = Path(vpath)
        if not p.exists():
            print(f"[SKIP] Not found: {p}")
            continue
        all_results.append(compare_video(p, scale=args.scale))

    if not all_results:
        sys.exit("[ERROR] No valid videos processed.")

    plot_comparison(all_results, args.output)
    print_table(all_results)


if __name__ == "__main__":
    main()
