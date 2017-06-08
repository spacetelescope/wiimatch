"""wiimatch"""

from __future__ import (absolute_import, division, unicode_literals,
                        print_function)
import os


__docformat__ = 'restructuredtext en'
__version__ = '0.1.1'
__version_date__ = '08-June-2017'
__author__ = 'Mihai Cara'


from .version import *

from . import match
from . import lsq_optimizer
from . import utils
