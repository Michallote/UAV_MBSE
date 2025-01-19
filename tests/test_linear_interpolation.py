import numpy as np
import pytest

from geometry.interpolation import (ndarray_linear_interpolate,
                                    resample_curve_equidistant)


def test_basic_interpolation():
    """Test interpolation on a simple curve."""
    curve = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    indices = np.array([1.5, 2.5])
    expected = np.array([[1.5, 1.5], [2.5, 2.5]])
    assert np.allclose(ndarray_linear_interpolate(curve, indices), expected)


def test_basic_interpolation_3d():
    """Test interpolation on a 3D curve."""
    curve = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    indices = np.array([0.5, 1.5, 2.5])
    expected = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]])
    assert np.allclose(ndarray_linear_interpolate(curve, indices), expected)


def test_boundary_conditions():
    """Test interpolation at the boundaries of the curve."""
    curve = np.array([[0, 0], [1, 1], [2, 2]])
    indices = np.array([0, 2])
    expected = np.array([[0, 0], [2, 2]])
    assert np.allclose(ndarray_linear_interpolate(curve, indices), expected)


def test_non_integer_indices():
    """Test interpolation with non-integer indices."""
    curve = np.array([[0, 0], [1, 1], [4, 4]])
    indices = np.array([1.2, 1.8])
    expected = np.array([[1.6, 1.6], [3.4, 3.4]])
    assert np.allclose(ndarray_linear_interpolate(curve, indices), expected)


def test_multidimensional_curve():
    """Test interpolation on a 3D curve along the same curve."""
    curve = np.random.rand(4, 5, 3)  # 3D curve
    indices = np.array([1, 2])
    results = ndarray_linear_interpolate(curve, indices)
    assert results.shape == (2, 5, 3)


def test_empty_curve():
    """Test that empty curves raise an error."""
    curve = np.array([])
    indices = np.array([1, 2])
    with pytest.raises(IndexError):
        ndarray_linear_interpolate(curve, indices)


def test_multiple_cuves():
    """Test the interpolation of multiple curves."""
    # Add more tests as needed
    curve = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

    curves = np.array([curve + i for i in range(5)])
    indices = np.array([0.5, 1.5, 2.5])

    expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

    expected_m = np.array([expected + result for result in indices])
    assert np.allclose(ndarray_linear_interpolate(curves, indices), expected_m)


def test_matrices():
    """Test the interpolation of 3D matrices."""
    # Add more tests as needed
    curve = np.array(
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ]
    )

    curves = np.array([curve + i for i in range(5)])
    indices = np.array([0.5, 1.5, 2.5])

    expected = curve

    expected_m = np.array([expected + result for result in indices])
    assert np.allclose(ndarray_linear_interpolate(curves, indices), expected_m)


def test_resampling_algorithm_with_element_length():
    """Test the resampling algorithm with a given element length."""
    curve = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    element_length = 0.5
    expected = np.c_[np.repeat(0, 7), np.linspace(0, 3, 7)]
    assert np.allclose(
        resample_curve_equidistant(curve, target_segment_length=element_length),
        expected,
    )
