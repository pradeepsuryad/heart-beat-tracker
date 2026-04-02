"""
Tests for src/tracker.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import tracker


class TestComputeFlow:
    def test_returns_three_arrays(self, gray_frame, shifted_frame):
        mag, angle, flow = tracker.compute_flow(gray_frame, shifted_frame)
        assert mag is not None
        assert angle is not None
        assert flow is not None

    def test_output_shapes_match_input(self, gray_frame, shifted_frame):
        h, w = gray_frame.shape
        mag, angle, flow = tracker.compute_flow(gray_frame, shifted_frame)
        assert mag.shape == (h, w)
        assert angle.shape == (h, w)
        assert flow.shape == (h, w, 2)

    def test_zero_motion_gives_low_magnitude(self, gray_frame):
        """Identical frames → magnitude should be near zero."""
        mag, _, _ = tracker.compute_flow(gray_frame, gray_frame)
        assert mag.mean() < 0.5

    def test_shifted_frame_gives_nonzero_magnitude(self, gray_frame, shifted_frame):
        """Shifted frame → measurable motion."""
        mag, _, _ = tracker.compute_flow(gray_frame, shifted_frame)
        assert mag.mean() > 0.1

    def test_angle_range(self, gray_frame, shifted_frame):
        """Angles must be in [0, 2π)."""
        _, angle, _ = tracker.compute_flow(gray_frame, shifted_frame)
        assert angle.min() >= 0.0
        assert angle.max() <= 2 * np.pi + 1e-5

    def test_magnitude_nonnegative(self, gray_frame, shifted_frame):
        mag, _, _ = tracker.compute_flow(gray_frame, shifted_frame)
        assert mag.min() >= 0.0


class TestFlowToHsvFrame:
    def test_output_shape_matches_input(self, gray_frame, shifted_frame, bgr_frame):
        mag, angle, _ = tracker.compute_flow(gray_frame, shifted_frame)
        out = tracker.flow_to_hsv_frame(mag, angle, bgr_frame)
        assert out.shape == bgr_frame.shape

    def test_output_dtype_uint8(self, gray_frame, shifted_frame, bgr_frame):
        mag, angle, _ = tracker.compute_flow(gray_frame, shifted_frame)
        out = tracker.flow_to_hsv_frame(mag, angle, bgr_frame)
        assert out.dtype == np.uint8

    def test_zero_flow_gives_dark_overlay(self, bgr_frame):
        """Zero magnitude → value channel = 0 → overlay is dark."""
        mag = np.zeros((64, 64), dtype=np.float32)
        angle = np.zeros((64, 64), dtype=np.float32)
        out = tracker.flow_to_hsv_frame(mag, angle, bgr_frame)
        # Output should be darker than or equal to input (flow adds no brightness)
        assert out.mean() <= bgr_frame.mean() + 1


class TestDrawStats:
    def test_modifies_frame_inplace(self, bgr_frame):
        original = bgr_frame.copy()
        tracker.draw_stats(bgr_frame, frame_idx=5, mean_mag=1.2, max_mag=4.5)
        assert not np.array_equal(bgr_frame, original)

    def test_does_not_change_shape(self, bgr_frame):
        shape_before = bgr_frame.shape
        tracker.draw_stats(bgr_frame, frame_idx=1, mean_mag=0.0, max_mag=0.0)
        assert bgr_frame.shape == shape_before


class TestProcessVideo:
    def test_output_file_created(self, small_video_path, tmp_path):
        out = tmp_path / "out.mp4"
        tracker.process_video(small_video_path, out, scale=1.0, overlays=False)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_output_with_overlays(self, small_video_path, tmp_path):
        out = tmp_path / "out_overlays.mp4"
        tracker.process_video(small_video_path, out, scale=1.0, overlays=True)
        assert out.exists()

    def test_scale_reduces_resolution(self, small_video_path, tmp_path):
        out = tmp_path / "out_half.mp4"
        tracker.process_video(small_video_path, out, scale=0.5, overlays=False)
        cap = cv2.VideoCapture(str(out))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        assert w == 32
        assert h == 32
