import warnings
from setuptools import setup, Extension

publish=True
if publish:
    from distutils.command.build_ext import build_ext
    keyboardpyx = Extension(
        'word2keypress/_keyboard',
        sources = ['word2keypress/_keyboard.c'],
    )
else:
    from Cython.Distutils import build_ext    
    keyboardpyx = Extension(
        'word2keypress/_keyboard',
        sources = ['word2keypress/_keyboard.pyx'],
    )



configuration = dict(
    name='word2keypress',
    version='0.2',
    description='Convert word to keypress sequence',
    author='Rahul Chatterjee',
    author_email='rahul@cs.cornell.edu',
    url='https://github.com/rchatterjee/word2keypress.git',
    download_url = 'https://github.com/rchatterjee/word2keypress/tarball/v0.2',
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


setup(**configuration)
