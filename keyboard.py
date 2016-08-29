import pyximport; pyximport.install()
from _keyboard import *
import time

if __name__ == '__main__':
    s = time.time()
    kb = Keyboard('US')
    ks = u'{s}wo{c}rd123{s}{c}'.format(c=CAPS_KEY, s=SHIFT_KEY)
    pw1 = u'Password123'
    p1 = kb.word_to_key_presses(pw1)
    pw11 = kb.key_presses_to_word(p1)
    print ("{!r} -> {!r} --> {!r}".format(pw1, p1, pw11))
    print ("{!r} -> {!r}".format(ks, kb.key_presses_to_word(ks)))
    print "Time take: {}s".format(time.time()-s)
