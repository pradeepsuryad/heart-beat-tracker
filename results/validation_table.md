# BPM Validation Results

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

| Sequence | Frames | Duration (s) | Mean BPM | Std BPM | Median BPM | CV (%) | Global BPM (FFT) | SNR (dB) | Physio valid (%) |
|---|---|---|---|---|---|---|---|---|---|
| hamlyn_seq04 | 1573 | 62.9 | 98.0 | 5.9 | 96.0 | 6.0 | 95.4 | 13.9 | 100.0 |
| hamlyn_seq05 | 899 | 36.0 | 105.6 | 6.9 | 108.0 | 6.5 | 110.2 | 14.4 | 100.0 |

## Notes

- Scale factor 0.5 applied before flow computation (half resolution).
- Rolling window: 10 seconds (250 frames at 25 fps).
- Physiological range: 50–150 BPM (covers ex-vivo porcine cadaver stimulation rates).
- SNR computed on the full-video magnitude signal FFT.
