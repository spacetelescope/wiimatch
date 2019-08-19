"""
A module containing unit tests for the `wcsutil` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
import pytest
import numpy as np

from wiimatch import utils


def test_utils_coordinates():
    image_shape = (3, 5, 4)
    center = (0, 0, 0)
    c = utils.create_coordinate_arrays(image_shape, center=center)
    ind = np.indices(image_shape, dtype=np.float)[::-1]

    assert np.allclose(c[0], ind, rtol=1.e-8, atol=1.e-12)
    assert np.allclose(c[1], center, rtol=1.e-8, atol=1.e-12)


def test_utils_coordinates_no_center():
    image_shape = (3, 5, 4)
    c = utils.create_coordinate_arrays(image_shape, center=None)
    ind = np.indices(image_shape, dtype=np.float)[::-1]

    center = tuple(i // 2 for i in image_shape)
    for orig, cc, i in zip(center, c[0], ind):
        assert(i - orig, cc)

    assert np.allclose(c[1], center, rtol=1.e-8, atol=1.e-12)
