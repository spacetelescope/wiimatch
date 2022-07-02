"""wiimatch"""
from pkg_resources import get_distribution, DistributionNotFound


__docformat__ = 'restructuredtext en'
__author__ = 'Mihai Cara'


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = 'UNKNOWN'


from . import match  # noqa: F401
from . import lsq_optimizer  # noqa: F401
from . import utils  # noqa: F401
from . import containers  # noqa: F401
