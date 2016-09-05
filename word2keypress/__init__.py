from _keyboard import Keyboard
import Levenshtein as lv

kb = Keyboard(u'US')
def distance(w1, w2):
    # print kb.print_keyseq(kb.word_to_keyseq(unicode(w1))),
    # print kb.print_keyseq(kb.word_to_keyseq(unicode(w2)))
    return lv.distance(
        kb.word_to_keyseq(unicode(w1)),
        kb.word_to_keyseq(unicode(w2))
    )


__all__ = [
    Keyboard,
    distance
]
