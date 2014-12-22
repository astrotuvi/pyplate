from setuptools import setup
import pyplate

setup(
    name = 'pyplate',
    version = pyplate.__version__,
    url = 'https://github.com/astrotuvi/pyplate',
    license = 'Apache License, Version 2.0',
    author = 'Taavi Tuvikene',
    author_email = 'taavi.tuvikene@to.ee',
    description = 'A Python package for processing astronomical photographic plates',
    classifiers = ['Programming Language :: Python',
                   'Programming Language :: Python :: 2.7',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Astronomy'
                  ],
    packages = ['pyplate'],
    install_requires = ['numpy>=1.7',
                        'astropy>=0.2.3',
                        'ephem'
                       ]
)

