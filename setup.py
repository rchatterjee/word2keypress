from distutils.core import setup
from Cython.Build import cythonize
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='word2keypress',
    version='1.0',
    description='Convert word to keypress sequence',
    author='Rahul Chatterjee',
    author_email='rahul@cs.cornell.edu',
    url='https://github.com/rchatterjee/word2keypress.git',
    download_url = 'https://github.com/rchatterjee/word2keypress/tarball/v1.0',
    install_requires=[
        'cython',
        'numpy',
        'python-levenshtein'
    ],
    long_description='See README.md',
    packages=['word2keypress'],
    ext_modules=cythonize('word2keypress/_keyboard.pyx'),  # accepts a glob pattern
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License"
    ],
)
