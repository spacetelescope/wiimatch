"""
This module provides utility functions for use by :py:mod:`wiimatch` module.

:Author: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`../LICENSE`

"""
from __future__ import (absolute_import, division, unicode_literals,
                        print_function)

import numpy as np


__all__ = ['create_coordinate_arrays']


def create_coordinate_arrays(image_shape, center=None, image2world=None):
    """
    Create a list of coordinate arrays/grids for each dimension in the image
    shape. This function is similar to `numpy.indices` except it returns the
    list of arrays in reversed order. In addition, it can center image
    coordinates to a provided ``center`` and also convert image coordinates to
    world coordinates using provided ``image2world`` function.

    Parameters
    ----------
    image_shape : sequence of int
        The shape of the image/grid.

    center : iterable, None
        An iterable of length equal to the number of dimensions in
        ``image_shape`` that indicates the center of the coordinate system
        in **image** coordinates. When ``center`` is `None` then ``center`` is
        set to the middle of the "image" as ``center[i]=image_shape[i]//2``.
        If ``image2world`` is not `None`, then center will first be converted
        to world coordinates.

    image2world : function, None
        Image-to-world coordinates transformation function. This function
        must be of the form ``f(x,y,z,...)`` and accept a number of arguments
        `numpy.ndarray` arguments equal to the dimensionality of the image.

    Returns
    -------
    coord_arrays : list
        A list of `numpy.ndarray` coordinate arrays each of ``image_shape``
        shape.

    Examples
    --------
    >>> import wiimatch
    >>> wiimatch.utils.create_coordinate_arrays((3,5,4))
    (array([[[-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.]],
            [[-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.]],
            [[-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.],
             [-1.,  0.,  1.,  2.]]]),
     array([[[-2., -2., -2., -2.],
             [-1., -1., -1., -1.],
             [ 0.,  0.,  0.,  0.],
             [ 1.,  1.,  1.,  1.],
             [ 2.,  2.,  2.,  2.]],
            [[-2., -2., -2., -2.],
             [-1., -1., -1., -1.],
             [ 0.,  0.,  0.,  0.],
             [ 1.,  1.,  1.,  1.],
             [ 2.,  2.,  2.,  2.]],
            [[-2., -2., -2., -2.],
             [-1., -1., -1., -1.],
             [ 0.,  0.,  0.,  0.],
             [ 1.,  1.,  1.,  1.],
             [ 2.,  2.,  2.,  2.]]]),
     array([[[-2., -2., -2., -2.],
             [-2., -2., -2., -2.],
             [-2., -2., -2., -2.],
             [-2., -2., -2., -2.],
             [-2., -2., -2., -2.]],
            [[-1., -1., -1., -1.],
             [-1., -1., -1., -1.],
             [-1., -1., -1., -1.],
             [-1., -1., -1., -1.],
             [-1., -1., -1., -1.]],
            [[ 0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.]]]))

    """
    if center is None:
        # set the center at the center of the image array:
        center = tuple([float(i//2) for i in image_shape])

    else:
        if len(center) != len(image_shape):
            raise ValueError("Number of coordinates of the 'center' must "
                             "match the dimentionality of the image.")

    ind = np.indices(image_shape, dtype=np.float)[::-1]

    if image2world is None:
        coord_arrays = tuple([i - c for (i, c) in zip(ind, center)])

    else:
        # convert image's center from image to world coordinates:
        center = tuple(map(float, image2world(center)))

        # convert pixel indices to world coordinates:
        w = image2world(*coord_arrays)

        coord_arrays = tuple([i - c for i in zip(w, center)])

    return coord_arrays
