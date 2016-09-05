import warnings
try:
    from Cython.Distutils import build_ext
    from setuptools import setup, Extension
    HAVE_CYTHON = True
except ImportError as e:
    HAVE_CYTHON = False
    warnings.warn(e.message)
    from distutils.core import setup, Extension
    from distutils.command import build_ext

keyboardpyx = Extension(
    '_keyboard',
    sources = ['word2keypress/_keyboard.pyx'],
    include_dirs = ['word2keypress/']
)

configuration = dict(
    name='word2keypress',
    version='1.1',
    description='Convert word to keypress sequence',
    author='Rahul Chatterjee',
    author_email='rahul@cs.cornell.edu',
    url='https://github.com/rchatterjee/word2keypress.git',
    download_url = 'https://github.com/rchatterjee/word2keypress/tarball/v1.1',
    install_requires=[
        'numpy',
        'python-levenshtein'
    ],
    long_description='See README.md',
    packages=['word2keypress'],
    ext_modules=[keyboardpyx], # cythonize('word2keypress/_keyboard.pyx'),  # accepts a glob pattern
    cmdclass={'build_ext': build_ext},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License"
    ]
)


if not HAVE_CYTHON:
    keyboardpyx.sources[0] = '_keyboard.c'


setup(**configuration)


# from setuptools import setup, extension
# from Cython.Build import cythonize

# try:
#     from Cython.Distutils import build_ext
# except ImportError:
#     use_cython = False
# else:
#     use_cython = True

# import os

# def read(fname):
#     return open(os.path.join(os.path.dirname(__file__), fname)).read()
# ext = extension.Extension(
#     'word2keypress/_keyboard', 
#     ['word2keypress/_keyboard.pyx']
# )

# setup(
# )
