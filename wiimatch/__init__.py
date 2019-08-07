"""wiimatch"""
import os
from pkg_resources import get_distribution, DistributionNotFound


__docformat__ = 'restructuredtext en'
__author__ = 'Mihai Cara'


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = 'UNKNOWN'


from . import match
from . import lsq_optimizer
from . import utils
