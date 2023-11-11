"""wiimatch"""


__docformat__ = 'restructuredtext en'
__author__ = 'Mihai Cara'


from importlib.metadata import PackageNotFoundError, version
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = 'UNKNOWN'


from . import match  # noqa: F401
from . import lsq_optimizer  # noqa: F401
from . import utils  # noqa: F401
from . import containers  # noqa: F401
