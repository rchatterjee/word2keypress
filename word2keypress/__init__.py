from _keyboard import Keyboard
kb = Keyboard(u'US')

try:
    from pyxdameraulevenshtein import \
        normalized_damerau_levenshtein_distance as _distance
except ImportError:
    from Levenshtein import distance as _distance

def distance(w1, w2):
    """
    Computes the edit (DL edit) distance between two strings @s1 and @s2.
    Edits are: [insert, delete, replace, transpose]
    """
    return _distance(
        kb.word_to_keyseq(unicode(w1)),
        kb.word_to_keyseq(unicode(w2))
    )


__all__ = [
    Keyboard,
    distance
]
