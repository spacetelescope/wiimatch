=========
CHANGELOG
=========

v0.1.2 (12-June-2017)
=====================

Added
^^^^^

- Several functions now return more values that can be used to analyse returned
  results:

  - :py:func:`wiimatch.utils.create_coordinate_arrays` now returns effective
    ``center`` values used in generating coordinate array and coordinate system
    type (``'image'`` or ``'world'``);

  - :py:func:`wiimatch.lsq_optimizer.build_lsq_eqs` now returns coordinate
    arrays, effective ``center`` values used in generating coordinate array,
    and the coordinate system type of coordinates in addition to coefficients
    of linear equations;

  - :py:func:`wiimatch.match.match_lsq` now optionally returns coefficients
    of linear equations, coordinate arrays, effective ``center`` values used
    in generating coordinate array, and the coordinate system type of
    coordinates in addition to optimal solution to the matching problem.
    New parameter ``ext_return`` indicates to return extended information.


v0.1.1 (06-June-2017)
=====================

Added
^^^^^

- ``center_cs`` parameter to :py:func:`wiimatch.utils.create_coordinate_arrays`
  :py:func:`wiimatch.match.match_lsq` and
  :py:func:`wiimatch.lsq_optimizer.build_lsq_eqs` in order to allow
  specification of the coordinate system of the center
  (``'image'`` or ``'world'``) when it is explicitly set.

Fixed
^^^^^

- Broken logic in :py:func:`wiimatch.utils.create_coordinate_arrays` code
  for generating coordinate arrays.


v0.1.0 (09-May-2017)
====================

Initial release.
