"""
Data containers for accessing image data (i.e., `numpy.ndarray`)
uniformly whether they are kept in memory, as memory mapped files (load),
or stored to/loaded from a file as whole arrays.

:Author: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`LICENSE`

"""

import abc
import tempfile

import numpy as np


__all__ = ['WMData', 'WMInMemoryData', 'WMMappedData', 'WMMemMappedData']


class WMData(abc.ABC):
    """ Base class for all data containers. Provides a common interface to
        access data.
    """
    kind = 'mapped'
    """ Hints to how data are stored: ``'mapped'``, ``'file'``, or
        ``'in-memory'``. May be used by code for performance optimization. """

    @property
    @abc.abstractmethod
    def data(self):
        """ Sets/Gets linked data.

        Parameters
        ----------
        data : object
            Data to be set.

        """
        pass

    @data.setter
    @abc.abstractmethod
    def data(self, data):
        pass

    @property
    @abc.abstractmethod
    def shape(self):
        """ Returns a tuple describing the shape of linked data. """
        pass


class WMInMemoryData(WMData):
    """ Acessor for in-memory `numpy.ndarray` data. """

    kind = 'in-memory'
    """ Hints to how data are stored: ``'mapped'``, ``'file'``, or
        ``'in-memory'``. May be used by code for performance optimization. """

    def __init__(self, data):
        super().__init__()
        self.data = data

    @property
    def data(self):
        """ Sets/gets linked `numpy.ndarray`.

        Parameters
        ----------
        data : object
            Data to be set.

        """
        return self._data

    @data.setter
    def data(self, data):
        self._data = np.asarray(data)

    @property
    def shape(self):
        """ Returns a tuple describing the shape of linked data. """
        return np.shape(self._data)


class WMMappedData(WMData):
    """ Data container for arrays stored in temporary files. This is best
        suited when array data are needed in memory all at once and when array
        is not needed - it can be stored to a file.

        To access small segments of data, use cls:`WMMemMappedData`.
    """

    kind = 'file'
    """ Hints to how data are stored: ``'mapped'``, ``'file'``, or
        ``'in-memory'``. May be used by code for performance optimization. """

    def __init__(self, data, tmpfile=None, prefix='tmp_wiimatch_',
                 suffix='.npy', tmpdir=''):
        super().__init__()
        if tmpfile is None:
            self._close = True
            self._tmp = tempfile.NamedTemporaryFile(
                prefix=prefix,
                suffix=suffix,
                dir=tmpdir
            )
            if not self._tmp:
                raise RuntimeError("Unable to create temporary file.")
        else:
            # temp file managed by the caller
            self._close = False
            self._tmp = tmpfile

        self.data = data

    @property
    def data(self):
        """ Sets/gets linked `numpy.ndarray`.

        Parameters
        ----------
        data : object
            Data to be set.

        """
        self._tmp.seek(0)
        return np.load(self._tmp)

    @data.setter
    def data(self, data):
        data = np.asarray(data)
        self._data_shape = data.shape
        self._tmp.seek(0)
        np.save(self._tmp, data)

    def __del__(self):
        if self._close:
            self._tmp.close()

    @property
    def shape(self):
        """ Returns a tuple describing the shape of linked data. """
        return self._data_shape


class WMMemMappedData(WMData):
    """ Data container for arrays stored in temporary files. This is best
        suited when array data are needed in memory all at once and when array
        is not needed - it can be stored to a file.

        To access entire data arrays, use cls:`WMMappedData`.
    """

    kind = 'mapped'
    """ Hints to how data are stored: ``'mapped'``, ``'file'``, or
        ``'in-memory'``. May be used by code for performance optimization. """

    def __init__(self, data, tmpfile=None, prefix='tmp_wiimatch_',
                 suffix='.npy', tmpdir=''):
        super().__init__()
        if tmpfile is None:
            self._close = True
            self._tmp = tempfile.NamedTemporaryFile(
                prefix=prefix,
                suffix=suffix,
                dir=tmpdir
            )
            if not self._tmp:
                raise RuntimeError("Unable to create temporary file.")
        else:
            # temp file managed by the caller
            self._close = False
            self._tmp = tmpfile

        self.data = data

    @property
    def data(self):
        """ Sets/gets linked `numpy.ndarray`.

        Parameters
        ----------
        data : object
            Data to be set.

        """
        return self._data

    @data.setter
    def data(self, data):
        data = np.asarray(data)
        self._data_shape = data.shape
        self._tmp.seek(0)
        self._data = np.memmap(self._tmp, dtype=data.dtype.type, mode='w+',
                               shape=data.shape)
        self._data[...] = data[...]

    def __del__(self):
        if self._close:
            self._data = None
            self._tmp.close()

    @property
    def shape(self):
        """ Returns a tuple describing the shape of linked data. """
        return self._data_shape
