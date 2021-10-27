from setuptools import setup, find_packages

setup(
    name = 'pyplate',
    version = open('pyplate/_version.py').readlines()[-1].split()[-1].strip('"\''),
    url = 'https://github.com/astrotuvi/pyplate',
    license = 'Apache License, Version 2.0',
    author = 'Taavi Tuvikene',
    author_email = 'taavi.tuvikene@ut.ee',
    description = 'A Python package for processing astronomical photographic plates',
    classifiers = ['Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: Apache Software License',
                   'Topic :: Scientific/Engineering :: Astronomy'
                  ],
    packages = find_packages(),
    include_package_data = True,
    install_requires = ['numpy>=1.7',
                        'astropy',
                        'scipy',
                        'statsmodels',
                        'healpy',
                        'astroquery',
                        'pyvo',
                        'ephem',
                        'pytimeparse',
                        'pytz',
                        'unicodecsv',
                        'unidecode',
                        'PyYAML',
                        'Pillow',
                        'Deprecated'
                       ],
    extras_require = {'ml': ['scikit-learn', 'tensorflow'],
                      'pgsql': ['psycopg2-binary'],
                      'mysql': ['pymysql']}
)
