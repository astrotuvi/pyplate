from setuptools import setup

setup(
    name = 'pyplate',
    version = open('pyplate/_version.py').readlines()[-1].split()[-1].strip('"\''),
    url = 'https://github.com/astrotuvi/pyplate',
    license = 'Apache License, Version 2.0',
    author = 'Taavi Tuvikene',
    author_email = 'taavi.tuvikene@to.ee',
    description = 'A Python package for processing astronomical photographic plates',
    classifiers = ['Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: Apache Software License',
                   'Topic :: Scientific/Engineering :: Astronomy'
                  ],
    packages = ['pyplate'],
    install_requires = ['numpy>=1.7',
                        'astropy>=1.0',
                        'ephem',
                        'pytimeparse',
                        'pytz',
                        'unicodecsv',
                        'unidecode',
                        'Pillow',
                        'Deprecated'
                       ],
    extras_require = {'solve': ['scipy', 'statsmodels', 'healpy', 'esutil'],
                      'mysql': ['MySQL-python']}
)

