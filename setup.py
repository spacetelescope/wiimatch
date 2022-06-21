#!/usr/bin/env python
import os
from setuptools import setup, find_packages

try:
    from distutils.config import ConfigParser
except ImportError:
    from configparser import ConfigParser

conf = ConfigParser()
conf.read(['setup.cfg'])

# Get some config values
metadata = dict(conf.items('metadata'))
PACKAGENAME = metadata.get('package_name', 'wiimatch')
DESCRIPTION = metadata.get('description', 'A package for optimal "matching" '
                           'N-dimentional image background using '
                           '(multivariate) polynomials')
LONG_DESCRIPTION = metadata.get('long_description', 'README.rst')
LONG_DESCRIPTION_CONTENT_TYPE = metadata.get('long_description_content_type',
                                             'text/x-rst')
AUTHOR = metadata.get('author', 'Mihai Cara')
AUTHOR_EMAIL = metadata.get('author_email', 'help@stsci.edu')
URL = metadata.get('url', 'https://github.com/spacetelescope/wiimatch')
LICENSE = metadata.get('license', 'BSD-3-Clause')

# load long description
this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, LONG_DESCRIPTION), encoding='utf-8') as f:
    long_description = f.read()

PACKAGE_DATA = {
    '': [
        'README.rst',
        'LICENSE.txt',
        'CHANGELOG.rst',
        '*.fits',
        '*.txt',
        '*.inc',
        '*.cfg',
        '*.csv',
        '*.yaml',
        '*.json'
    ],
}

INSTALL_REQUIRES = [
    'numpy',
    'scipy',
]

SETUP_REQUIRES = [
    'setuptools_scm',
]

TESTS_REQUIRE = [
    'pytest',
    'pytest-cov',
    'pytest-doctestplus',
    'codecov',
]

DOCS_REQUIRE = [
    'numpydoc',
    'graphviz',
    'sphinx',
    'sphinx-rtd-theme',
    'stsci-rtd-theme',
    'sphinx_automodapi',
]

OPTIONAL_DEP = [
    'scipy',
]

setup(
    name=PACKAGENAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Development Status :: 3 - Alpha',
    ],
    use_scm_version=True,
    setup_requires=SETUP_REQUIRES,
    python_requires='>=3.5',
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    ext_modules=[],
    extras_require={
        'docs': DOCS_REQUIRE,
        'test': TESTS_REQUIRE,
        'all': OPTIONAL_DEP,
    },
    project_urls={
        'Bug Reports': 'https://github.com/spacetelescope/wiimatch/issues/',
        'Source': 'https://github.com/spacetelescope/wiimatch/',
        'Help': 'https://hsthelp.stsci.edu/',
    },
)
