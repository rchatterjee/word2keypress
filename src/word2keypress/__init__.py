from ._keyboard import Keyboard
kb = Keyboard('qwerty')

# _ROOT = os.path.dirname(os.path.abspath(__file__))
# def _get_data_path(path):
#     return os.path.join(_ROOT, path)

def distance(w1, w2):
    """
    Computes the edit (DL edit) distance between two strings @s1 and @s2.
    Edits are: [insert, delete, replace, transpose]
    """
    # Some quick replies.
    if w1==w2: return 0
    if w1.swapcase() == w2: return 1
    return kb.dl_distance(w1, w2)


__all__ = [
    Keyboard,
    distance
]
