=========
CHANGELOG
=========

v0.1.1 (06-June-2017)
=====================

Added
^^^^^

- ``center_cs`` parameter to :py:func:`wiimatch.utils.create_coordinate_arrays`
  :py:func:`wiimatch.utils.match_lsq` and
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
