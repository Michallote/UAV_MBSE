import numpy as np
import pytest

from src.utils.interpolation import _linear_interpolate  # Replace with actual import


def test_basic_interpolation():
    curve = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    indices = np.array([1.5, 2.5])
    expected = np.array([[1.5, 1.5], [2.5, 2.5]])
    assert np.allclose(_linear_interpolate(curve, indices), expected)


def test_boundary_conditions():
    curve = np.array([[0, 0], [1, 1], [2, 2]])
    indices = np.array([0, 2])
    expected = np.array([[0, 0], [2, 2]])
    assert np.allclose(_linear_interpolate(curve, indices), expected)


def test_non_integer_indices():
    curve = np.array([[0, 0], [1, 1], [4, 4]])
    indices = np.array([1.2, 1.8])
    expected = np.array([[1.2, 1.2], [1.8, 1.8]])
    assert np.allclose(_linear_interpolate(curve, indices), expected)


def test_multidimensional_curve():
    curve = np.random.rand(4, 5, 3)  # 3D curve
    indices = np.array([1, 2])
    results = _linear_interpolate(curve, indices)
    assert results.shape == (2, 5, 3)


def test_empty_curve():
    curve = np.array([])
    indices = np.array([1, 2])
    with pytest.raises(IndexError):
        _linear_interpolate(curve, indices)


# Add more tests as needed
