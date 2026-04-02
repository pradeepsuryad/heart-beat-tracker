"""
Tests for src/visualizer.py
"""

import collections
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import visualizer


@pytest.fixture()
def frame():
    return np.full((120, 160, 3), 80, dtype=np.uint8)


@pytest.fixture()
def flow():
    """Constant rightward flow of 2 px/frame."""
    f = np.zeros((120, 160, 2), dtype=np.float32)
    f[..., 0] = 2.0
    return f


class TestDrawColorbar:
    def test_modifies_frame(self, frame):
        original = frame.copy()
        visualizer.draw_colorbar(frame)
        assert not np.array_equal(frame, original)

    def test_shape_unchanged(self, frame):
        shape = frame.shape
        visualizer.draw_colorbar(frame)
        assert frame.shape == shape

    def test_dtype_unchanged(self, frame):
        visualizer.draw_colorbar(frame)
        assert frame.dtype == np.uint8


class TestDrawMotionVectors:
    def test_modifies_frame_with_nonzero_flow(self, frame, flow):
        original = frame.copy()
        visualizer.draw_motion_vectors(frame, flow)
        assert not np.array_equal(frame, original)

    def test_zero_flow_does_not_modify_frame(self, frame):
        zero_flow = np.zeros((120, 160, 2), dtype=np.float32)
        original = frame.copy()
        visualizer.draw_motion_vectors(frame, zero_flow)
        assert np.array_equal(frame, original)

    def test_shape_unchanged(self, frame, flow):
        shape = frame.shape
        visualizer.draw_motion_vectors(frame, flow)
        assert frame.shape == shape


class TestDrawPhaseIndicator:
    def test_returns_string(self, frame):
        history = collections.deque(maxlen=15)
        result = visualizer.draw_phase_indicator(frame, 1.0, history)
        assert isinstance(result, str)

    def test_valid_phase_values(self, frame):
        history = collections.deque(maxlen=15)
        for mag in [0.5, 1.0, 2.0, 0.3]:
            result = visualizer.draw_phase_indicator(frame, mag, history)
            assert result in ("SYSTOLE", "DIASTOLE")

    def test_high_magnitude_is_systole(self, frame):
        history = collections.deque(maxlen=15, iterable=[0.5] * 15)
        # 10× above rolling mean should always be SYSTOLE
        result = visualizer.draw_phase_indicator(frame, 5.0, history)
        assert result == "SYSTOLE"

    def test_low_magnitude_is_diastole(self, frame):
        history = collections.deque(maxlen=15, iterable=[5.0] * 15)
        result = visualizer.draw_phase_indicator(frame, 0.1, history)
        assert result == "DIASTOLE"

    def test_modifies_frame(self, frame):
        history = collections.deque(maxlen=15)
        original = frame.copy()
        visualizer.draw_phase_indicator(frame, 1.0, history)
        assert not np.array_equal(frame, original)


class TestDrawStrainMap:
    def test_modifies_frame(self, frame):
        original = frame.copy()
        mag_history = collections.deque(
            [np.ones((120, 160), dtype=np.float32)] * 5, maxlen=10
        )
        visualizer.draw_strain_map(frame, mag_history)
        assert not np.array_equal(frame, original)

    def test_empty_history_no_crash(self, frame):
        history = collections.deque(maxlen=10)
        visualizer.draw_strain_map(frame, history)   # should not raise

    def test_shape_unchanged(self, frame):
        shape = frame.shape
        mag_history = collections.deque(
            [np.ones((120, 160), dtype=np.float32)] * 3, maxlen=10
        )
        visualizer.draw_strain_map(frame, mag_history)
        assert frame.shape == shape
