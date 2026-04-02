"""
Tests for src/utils.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import utils


class TestVideoInfo:
    def test_required_keys(self, small_video_path):
        info = utils.video_info(small_video_path)
        for key in ("path", "width", "height", "fps", "frame_count", "duration_s", "codec"):
            assert key in info

    def test_correct_dimensions(self, small_video_path):
        info = utils.video_info(small_video_path)
        assert info["width"] == 64
        assert info["height"] == 64

    def test_fps_positive(self, small_video_path):
        info = utils.video_info(small_video_path)
        assert info["fps"] > 0

    def test_invalid_path_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            utils.video_info(tmp_path / "nonexistent.avi")


class TestIterFrames:
    def test_yields_correct_count(self, small_video_path):
        frames = list(utils.iter_frames(small_video_path))
        assert len(frames) == 20

    def test_yields_tuples(self, small_video_path):
        first = next(utils.iter_frames(small_video_path))
        idx, frame = first
        assert isinstance(idx, int)
        assert frame.ndim == 3

    def test_indices_sequential(self, small_video_path):
        indices = [i for i, _ in utils.iter_frames(small_video_path)]
        assert indices == list(range(20))

    def test_scale_reduces_frame_size(self, small_video_path):
        _, frame = next(utils.iter_frames(small_video_path, scale=0.5))
        assert frame.shape[:2] == (32, 32)


class TestExtractFrames:
    def test_saves_all_frames(self, small_video_path, tmp_path):
        saved = utils.extract_frames(small_video_path, tmp_path)
        assert len(saved) == 20

    def test_step_reduces_count(self, small_video_path, tmp_path):
        saved = utils.extract_frames(small_video_path, tmp_path, step=5)
        assert len(saved) == 4   # frames 0, 5, 10, 15

    def test_files_exist_on_disk(self, small_video_path, tmp_path):
        saved = utils.extract_frames(small_video_path, tmp_path)
        for p in saved:
            assert p.exists()

    def test_jpg_format(self, small_video_path, tmp_path):
        saved = utils.extract_frames(small_video_path, tmp_path, fmt="jpg")
        assert all(p.suffix == ".jpg" for p in saved)


class TestMagnitudeSignal:
    def test_length_is_frames_minus_one(self, small_video_path):
        signal = utils.magnitude_signal(small_video_path, scale=1.0)
        assert len(signal) == 19   # 20 frames → 19 transitions

    def test_values_nonnegative(self, small_video_path):
        signal = utils.magnitude_signal(small_video_path, scale=1.0)
        assert all(v >= 0 for v in signal)

    def test_nonzero_for_moving_content(self, small_video_path):
        """Alternating bar pattern should produce non-trivial flow."""
        signal = utils.magnitude_signal(small_video_path, scale=1.0)
        assert max(signal) > 0.1
