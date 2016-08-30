# Convert word to keypress-sequence

I made this as a submodule for my project on correcting password typos. I found
this very cool, and might be useful in many scenarios where one has to find
vairations of word due to mistyping. 

Here we define typo as purely mistyping a key, and not due to ignorance. Hence,
finding possible typos of a word require converting the word into sequence of
key presses. This module provides those functionality to convert word into and
from sequence of key presses.  Also, given a word find the possible typo of that
word tuned to the typos due to mistyping. 

## dependency
Right now it needs cython. I will remove it in future.
* Cython.

## Install

```bash
$ python setup.py build_ext --inplace
```

```ipython
In [1]: from keyboard import Keyboard

In [2]: kb = Keyboard()

In [3]: kb.print_keyseq(kb.word_to_keyseq('Oasdfasd!3'))
Out[3]: u'<s>oasdfasd<s>13'

In [4]: kb.word_to
kb.word_to_keyseq  kb.word_to_typos   

In [4]: kb.word_to_typos('Password1!')
Out[4]: <generator at 0x104439408>

In [5]: s = kb.word_to_keyseq('PassD<o>rd1!')

In [6]: kb.keyseq_to_word(s) == w
Out[6]: True
```
