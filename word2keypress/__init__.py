from ._keyboard import Keyboard
kb = Keyboard('US')

from pyxdameraulevenshtein import \
    damerau_levenshtein_distance as _distance
# from Levenshtein import distance as _distance (To aid
# transposition as distance 1)

def distance(w1, w2):
    """
    Computes the edit (DL edit) distance between two strings @s1 and @s2.
    Edits are: [insert, delete, replace, transpose]
    """
    # Some quick replies.
    if w1==w2: return 0
    if w1.swapcase() == w2: return 1
    return _distance(
        kb.word_to_keyseq(w1),
        kb.word_to_keyseq(w2)
    )


__all__ = [
    Keyboard,
    distance
]
