"""
A module containing unit tests for the ``wcsutil`` module.

:License: :doc:`../LICENSE`

"""
import pytest
import numpy as np

from wiimatch import match


@pytest.mark.parametrize('solver', ['RLU', 'PINV'])
def test_match_lsq_solver(solver):
    # simulate background image data:
    c = [1.32, 0.15, 0.62, 0, 0.74, 0, 0, 0]
    im1 = np.zeros((5, 5, 4), dtype=float)
    cbg = c[0] * np.ones_like(im1)  # constand background level image

    # add slope:
    ind = np.indices(im1.shape, dtype=float)
    im3 = cbg + c[1] * ind[0] + c[2] * ind[1] + c[4] * ind[2]

    mask = np.ones_like(im1, dtype=np.int8)
    sigma = np.ones_like(im1, dtype=float)

    p = match.match_lsq(
        [im1, im3], [mask, mask], [sigma, sigma],
        degree=(1, 1, 1), center=(0, 0, 0), solver=solver
    )

    assert np.allclose(-p[0], p[1], rtol=1.e-8, atol=1.e-12)
    assert np.allclose(c, 2 * np.abs(p[0]), rtol=1.e-8, atol=1.e-12)


def test_match_lsq_extended_return():
    # simulate background image data:
    c = [1.32, 0.15, 0.62, 0, 0.74, 0, 0, 0]
    im1 = np.zeros((5, 5, 4), dtype=float)
    cbg = c[0] * np.ones_like(im1)  # constand background level image

    # add slope:
    ind = np.indices(im1.shape, dtype=float)
    im3 = cbg + c[1] * ind[0] + c[2] * ind[1] + c[4] * ind[2]

    mask = np.ones_like(im1, dtype=np.int8)
    sigma = np.ones_like(im1, dtype=float)

    p, a, b, coord_arrays, eff_center, coord_system = match.match_lsq(
        [im1, im3], [mask, mask], [sigma, sigma],
        degree=1, center=(0, 0, 0), ext_return=True
    )

    assert np.allclose(-p[0], p[1], rtol=1.e-8, atol=1.e-12)
    assert np.allclose(c, 2 * np.abs(p[0]), rtol=1.e-8, atol=1.e-12)


@pytest.mark.parametrize('degree', [1, (1, 1, 1)])
def test_match_lsq_num_degree(degree):
    # simulate background image data:
    c = [1.32, 0.15, 0.62, 0, 0.74, 0, 0, 0]
    im1 = np.zeros((5, 5, 4), dtype=float)
    cbg = c[0] * np.ones_like(im1)  # constand background level image

    # add slope:
    ind = np.indices(im1.shape, dtype=float)
    im3 = cbg + c[1] * ind[0] + c[2] * ind[1] + c[4] * ind[2]

    mask = np.ones_like(im1, dtype=np.int8)
    sigma = np.ones_like(im1, dtype=float)

    p = match.match_lsq(
        [im1, im3], [mask, mask], [sigma, sigma],
        degree=degree, center=(0, 0, 0), solver='RLU', ext_return=False
    )

    assert np.allclose(-p[0], p[1], rtol=1.e-8, atol=1.e-12)
    assert np.allclose(c, 2 * np.abs(p[0]), rtol=1.e-8, atol=1.e-12)
