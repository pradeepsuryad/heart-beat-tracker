# Endoscopic Heart Beat Tracker

Dense optical flow pipeline for cardiac tissue motion analysis in endoscopic video.
Computes per-pixel displacement using the Farnebäck algorithm, overlays a real-time
strain map, detects cardiac phase (systole/diastole), and estimates heart rate from
the flow signal without ECG hardware.

![Demo — Hamlyn seq04 annotated output](assets/demo.gif)

---

## Contents

- [Clinical Background](#clinical-background)
- [Algorithm](#algorithm)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Datasets](#datasets)
- [Reproducing the Results](#reproducing-the-results)
- [Running the Tracker](#running-the-tracker)
- [Visual Overlays](#visual-overlays)
- [Results](#results)
- [Tests](#tests)
- [References](#references)

---

## Clinical Background

During minimally invasive cardiac procedures (off-pump CABG, valve repair) the heart
continues to beat while the surgeon operates through small ports. Continuous myocardial
motion makes precise instrument placement difficult. A dense motion estimate (one
velocity vector per pixel) enables:

- Motion-compensating robot arms that stabilise a virtual still-point on the beating surface
- Phase gating — triggering surgical actions only during the lowest-motion phase (diastole)
- Strain maps that highlight regions of abnormal wall motion indicative of ischaemia
- Heart rate estimation from video without ECG hardware

---

## Algorithm

Classical sparse optical flow methods (Lucas-Kanade) track a sparse set of corner
features. Cardiac tissue is largely textureless and deforms non-rigidly, so a **dense**
estimate is required.

**Farnebäck dense optical flow** fits a quadratic polynomial to each pixel neighbourhood,
builds a Gaussian pyramid, and solves a linear system at each pyramid level (coarse → fine)
to find the displacement `(dx, dy)` that best aligns successive frames. The result is a
2-channel flow array at every pixel.

From `(dx, dy)` the pipeline derives:

| Quantity | Formula | Use |
|---|---|---|
| Magnitude | `√(dx²+dy²)` | Speed per pixel (px/frame) |
| Angle | `atan2(dy,dx)` | Direction of motion |
| Strain | rolling mean of magnitude | Cumulative tissue deformation |
| BPM | FFT dominant freq × 60 | Heart rate estimate |

See Farnebäck (2003) in [References](#references).

---

## Project Structure

```
heart-beat-tracker/
├── README.md
├── requirements.txt
├── src/
│   ├── tracker.py        ← flow computation, stats overlay, batch & live pipeline
│   ├── visualizer.py     ← strain map, phase indicator, motion vectors, colorbar
│   └── utils.py          ← video I/O, magnitude signal, rolling BPM estimator
├── scripts/
│   ├── download_data.py  ← downloads Hamlyn Dataset 4 & 5 from HuggingFace
│   └── prepare_endovis.py← batch-processes MICCAI EndoVis clips
├── tests/
│   ├── conftest.py
│   ├── test_tracker.py   ← 14 tests
│   ├── test_utils.py     ← 15 tests
│   └── test_visualizer.py← 14 tests
├── assets/               ← figures and GIF used in this README
├── data/                 ← place downloaded videos here (not tracked by git)
├── outputs/              ← annotated output videos (not tracked by git)
└── notebooks/
    └── demo.ipynb        ← end-to-end walkthrough with embedded outputs
```

> **Note:** `data/` and `outputs/` are excluded from git (`.gitignore`).
> Video files are large — regenerate them locally using the steps below.

---

## Setup

```bash
git clone https://github.com/pradeepsuryad/heart-beat-tracker.git
cd heart-beat-tracker

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Datasets

### Hamlyn Centre Endoscopic Video Dataset

In-vivo beating-heart sequences (sequences 4 and 5, 360×288, 25 fps).

```bash
python scripts/download_data.py
# Downloads rectified04.zip and rectified05.zip from Hugging Face,
# extracts image01/ frame sequences, stitches to .avi at 25 fps.
# Output: data/hamlyn_seq04.avi  data/hamlyn_seq05.avi
```

### MICCAI EndoVis Dataset

68 stereo clips across 12 surgical cases (ex-vivo porcine cadaver, in-vivo porcine
abdominal procedures, in-vivo human robotic-assisted partial nephrectomy).
Videos are stereo pairs stacked vertically (1280×2048) or horizontally (720×288).
Each clip includes `info.yaml` (stacking direction, resolution) and `calibration.yaml`
(stereo camera intrinsics).

Place the extracted dataset at any path and point `--src` to the `train/train/` folder:

```bash
python scripts/prepare_endovis.py --src /path/to/train/train
# Reads info.yaml, crops the left-camera half, runs the tracker on every clip.
# Output: outputs/endovis/case_X_clipY.mp4
```

Process specific cases only:

```bash
python scripts/prepare_endovis.py --src /path/to/train/train --cases 1 2 3
```

---

## Reproducing the Results

Follow these steps exactly to go from a fresh clone to all figures and videos.

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Download the Hamlyn dataset

```bash
python scripts/download_data.py
```

Produces `data/hamlyn_seq04.avi` and `data/hamlyn_seq05.avi`.

### Step 3 — Run the tracker on Hamlyn seq04

```bash
python src/tracker.py \
    --input  data/hamlyn_seq04.avi \
    --output outputs/demo_flow.mp4 \
    --scale  0.5
```

Produces `outputs/demo_flow.mp4` — the annotated video with all overlays.

### Step 4 — Run the demo notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

Run all cells in order (Kernel → Restart & Run All). The notebook produces:

| Figure | Content |
|---|---|
| Figure 1 | Original frame / flow magnitude heatmap / HSV direction overlay |
| Figure 2 | Heartbeat magnitude signal (raw + smoothed) with BPM estimate |
| Figure 3 | Strain map (1-second window) — original / heatmap / overlay |

### Step 5 — Run the tests

```bash
python -m pytest tests/ -v
```

All 43 tests should pass.

---

## Running the Tracker

### Batch mode (video file)

```bash
python src/tracker.py --input data/hamlyn_seq04.avi --output outputs/result.mp4
python src/tracker.py --input data/hamlyn_seq04.avi --output outputs/result.mp4 --scale 0.5
python src/tracker.py --input data/hamlyn_seq04.avi --output outputs/result.mp4 --no-overlays
```

### Live mode (webcam)

```bash
# Display only — no file saved
python src/tracker.py --live

# Display + save to file
python src/tracker.py --live --output outputs/live_recording.mp4

# Use a specific camera index
python src/tracker.py --live --camera 1 --output outputs/live_recording.mp4
```

Press **Q** to quit the live window. If `--output` is specified the full annotated
stream is written to that file while the window is open.

### All options

| Flag | Default | Description |
|---|---|---|
| `--input` | — | Path to source video (batch mode) |
| `--output` | — | Path for annotated output video (batch or live) |
| `--live` | off | Enable live webcam mode |
| `--camera` | `0` | Camera index for live mode |
| `--scale` | `1.0` | Resize factor before processing (`0.5` = half-res, ~4× faster) |
| `--fps` | match input | Override output FPS |
| `--no-overlays` | off | Disable all visual overlays |

---

## Visual Overlays

Each output frame carries four overlays (disable with `--no-overlays`):

| Overlay | Position | Description |
|---|---|---|
| Stats + BPM | Top-left | Frame index, mean/max displacement (px), rolling heart rate |
| Phase label | Top-right | SYSTOLE (red) or DIASTOLE (green) |
| Motion vectors | Full frame | Sparse arrow grid — direction and relative magnitude |
| Strain map | Full frame | JET-coloured 1-second mean magnitude |
| Colorbar | Bottom-right | Circular HSV wheel mapping colour → motion direction |

**Sample annotated frame:**

![Annotated output frame](assets/figure_04_video_frame.png)

---

## Results

### Optical Flow Visualisation

Left: raw input frame. Centre: per-pixel flow magnitude (brighter = faster motion).
Right: HSV direction overlay with motion vectors and colorbar.

![Optical flow](assets/figure_01_cell5.png)

### Heartbeat Signal and Heart Rate Estimate

Mean flow magnitude over all 1,573 frames (~63 s). Peaks = systole, troughs = diastole.
Dominant FFT frequency in 0.5–3.5 Hz gives the BPM estimate.

![Heartbeat signal](assets/figure_02_cell7.png)

### Strain Map

Mean flow magnitude over a 1-second sliding window (JET colormap).
Red = high sustained deformation (myocardium), blue = static tissue.

![Strain map](assets/figure_03_cell9.png)

---

## Quantitative Results

Self-consistency validation of the rolling FFT BPM estimator on both Hamlyn sequences.
No ECG ground truth is available in the public mirror; metrics characterise estimator
stability and signal quality.

| Sequence | Frames | Duration (s) | Mean BPM | Std BPM | Median BPM | CV (%) | Global BPM (FFT) | SNR (dB) | Physio valid (%) |
|---|---|---|---|---|---|---|---|---|---|
| hamlyn_seq04 | 1573 | 62.9 | 98.0 | 5.9 | 96.0 | 6.0 | 95.4 | 13.9 | 100.0 |
| hamlyn_seq05 | 899 | 36.0 | 105.6 | 6.9 | 108.0 | 6.5 | 110.2 | 14.4 | 100.0 |

**CV (%)** = coefficient of variation (std / mean × 100) — lower is more stable.
**SNR (dB)** = ratio of FFT peak power to noise-floor power in the 0.5–3.5 Hz cardiac band.
Both sequences report 100% of estimates within the physiological range (50–150 BPM).

![BPM validation — rolling estimate and FFT spectrum](assets/bpm_validation.png)

Reproduce with:
```bash
python scripts/validate_bpm.py
# outputs/bpm_validation.png  +  results/validation_table.md
```

---

## Tests

```bash
python -m pytest tests/ -v
```

43 tests covering all three modules:

| File | Tests | Covers |
|---|---|---|
| `test_tracker.py` | 14 | `compute_flow`, `flow_to_hsv_frame`, `draw_stats`, `process_video` |
| `test_utils.py` | 15 | `video_info`, `iter_frames`, `extract_frames`, `magnitude_signal` |
| `test_visualizer.py` | 14 | `draw_colorbar`, `draw_motion_vectors`, `draw_phase_indicator`, `draw_strain_map` |

---

## References

**Dataset**

> Mountney, P., Stoyanov, D., Davison, A., & Yang, G.-Z. (2010).
> *Simultaneous Stereoscope Localization and Soft-Tissue Mapping for Minimal Invasive Surgery.*
> MICCAI, LNCS vol. 6361, pp. 251–258.
> https://doi.org/10.1007/978-3-642-15705-9_31

> Stoyanov, D., Scarzanella, M. V., Pratt, P., & Yang, G.-Z. (2010).
> *Real-Time Stereo Reconstruction in Robotically Assisted Minimally Invasive Surgery.*
> MICCAI, LNCS vol. 6361, pp. 275–282.
> https://doi.org/10.1007/978-3-642-15705-9_34

> Recasens, A. (2023). *HamlynRectifiedDataset* [Data set]. Hugging Face.
> https://huggingface.co/datasets/Recasens/HamlynRectifiedDataset

**Algorithm**

> Farnebäck, G. (2003).
> *Two-Frame Motion Estimation Based on Polynomial Expansion.*
> SCIA, LNCS vol. 2749, pp. 363–370.
> https://doi.org/10.1007/3-540-45103-X_50

**Libraries**

> Bradski, G. (2000). *The OpenCV Library.* Dr. Dobb's Journal, 25(11), 120–125.
> https://opencv.org

> Harris, C. R. et al. (2020). *Array programming with NumPy.* Nature, 585, 357–362.
> https://doi.org/10.1038/s41586-020-2649-2

> Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment.*
> Computing in Science & Engineering, 9(3), 90–95.
> https://doi.org/10.1109/MCSE.2007.55
