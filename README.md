# Convert word to keypress-sequence
[![Build Status](https://travis-ci.org/rchatterjee/word2keypress.svg?branch=master)](https://travis-ci.org/rchatterjee/word2keypress)

I made this as a submodule for my project on correcting password typos. I found
this very cool, and might be useful in many scenarios where one has to find
vairations of word due to mistyping.

Here we define typo as purely mistyping a key, and not due to ignorance. Hence,
finding possible typos of a word require converting the word into sequence of
key presses. This module provides those functionality to convert word into and
from sequence of key presses.  Also, given a word find the possible typo of that
word tuned to the typos due to mistyping.

## Dependency
Right now it needs cython. I will remove it in future.
* Cython.

## Install
```bash
$ pip install word2keypress

# or

$ python setup.py install
```

## How to Use?

Refer to the `HowToRun.ipynb` file.

```python
from word2keypress import distance, Keyboard
kb = Keyboard(u'US') # making unicode is mandatory (weird Cython)
kseq = kb.word_to_keyseq('Password')
print "\nRaw sequence:", repr(kseq)

print "\nReadable sequence:", repr(kb.print_keyseq(kseq))

print "\nkeyseq->word:", kb.keyseq_to_word(kseq)

print "\ndistance:", distance('Password1', 'PASSWORD1')
```
