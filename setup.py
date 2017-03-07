#!/usr/bin/env python

from __future__ import absolute_import, print_function

import warnings
from setuptools import setup, Extension, find_packages

publish=True
if publish:
    from distutils.command.build_ext import build_ext
    keyboardpyx = Extension(
        'word2keypress/_keyboard',
        sources=['src/word2keypress/_keyboard.c'],
    )
else:
    from Cython.Distutils import build_ext
    keyboardpyx = Extension(
        'word2keypress/_keyboard',
        sources=['src/word2keypress/_keyboard.pyx'],
    )


VERSION = '1.0.8'
GITHUB_URL = 'https://github.com/rchatterjee/word2keypress/tarball/'
configuration = dict(
    name = 'word2keypress',
    version = VERSION,
    description = 'Convert word to keypress sequence',
    author = 'Rahul Chatterjee',
    author_email = 'rahul@cs.cornell.edu',
    url = 'https://github.com/rchatterjee/word2keypress.git',
    download_url = "{}/v{}".format(GITHUB_URL, VERSION),

    packages = find_packages('src'),
    package_dir={'': 'src'},
    include_package_data = True,
    package_data = {
        '': ['*.json', 'LICENSE', 'README.md']
    },

    long_description = 'See README.md',
    ext_modules = [keyboardpyx],
    cmdclass = {'build_ext': build_ext},

    # include_dirs = [np.get_include()],

    # install_requires = ["numpy >= 1.12"],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest'],

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License"
    ],
    license = 'MIT License',
    zip_safe = False,
)


setup(**configuration)
