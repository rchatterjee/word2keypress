
from __future__ import print_function
import os, sys, re, string
from collections import defaultdict
import itertools
import numpy as np
try:
    from word2keypress.weight_matrix import WEIGHT_MATRIX
    from word2keypress import Keyboard
except ImportError:
    from weight_matrix import WEIGHT_MATRIX
    import pyximport; pyximport.install()
    from _keyboard import Keyboard
import random
####### Extra Key codes #######
# 1 (\x01) : Start of a string
# 2 (\x02) : End of a string
# 3 (\x03) : Shift key
# 4 (\x04) : Caps lock key
# 5 (\x05) : Blank, bottom
#
# So, \x05 -> 'a' means inserting 'a'
#    ###################


KB = Keyboard("qwerty")
SHIFT_KEY = chr(3) # [u'\x03', "<s>"][user_friendly]
CAPS_KEY = chr(4) # [u'\x04', "<c>"][user_friendly]

ALLOWED_KEYS = set("`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./ {}{}"\
    .format(SHIFT_KEY, CAPS_KEY))
ALLOWED_CHARACTERS = set(string.printable[:-5])
STARTSTR, ENDSTR, BLANK = chr(1), chr(2), chr(5)
MAX_ALLOWED_EDITS = 3
UNWEIGHTED = True
DEBUG = False
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Weight type is how we calculate the cost of each edit.
# 1 : consider context, the WEIGHT_MATRIX should have probabilities
# 2 : Take 1/f as weight of each edit, where f is the frequency of
#     occurance of that edit.
WEIGHT_TYPE = 2

def dp(**kwargs):
    print(kwargs)


M = {}

def load_weight_matrix():
    global M, WEIGHT_MATRIX
    WEIGHT_MATRIX = dict(WEIGHT_MATRIX)
    for (l, r), f in WEIGHT_MATRIX.items():
        if l not in M:
            M[l] = {}
        if r in (ENDSTR, STARTSTR): continue
        M[l][r] = f


CONFUSING_SETS = [{'1', 'l', 'I'},
                  {'o', '0', 'O'}]
def confusions(w, s):
    for cset in CONFUSING_SETS:
        if w in cset and s in cset:
            return True
    return False

def _get_cost(s, w, i, N):
    l = min(len(s), len(w))
    prob = 0.0
    for j in range(-N, 1):  # start anywhere between i-N to i
        for k in range(1, max(3, N + 2)):  # end anywhere between i to i+N
            if i+j >= 0 and i+k <= l:
                tl, tr = s[i+j:i+k], w[i+j:i+k]
                if tl==tr: continue
                prob += M.get(tl, {}).get(tr, 0.0)
    return prob

def _delete(s, i, N):
    """The cost of deleting 'i'-th character from s. Also takes into
    account the context.
    """
    if WEIGHT_TYPE == 1:
        w = s[:i] + BLANK + s[i+1:]
        return _get_cost(s, w, i, N)
    elif WEIGHT_TYPE == 2:
        return WEIGHT_MATRIX.get((s[i-1], BLANK), 1)

def _insert(s, i, N):
    """The cost of inserting a character at 'i'-th character from s. Also
    takes into account the context.
    """
    if WEIGHT_TYPE == 1:
        w = s[:i] + BLANK + s[i+1:]
        return _get_cost(s, w, i, N)
    elif WEIGHT_TYPE == 2:
        return 1/WEIGHT_MATRIX.get((BLANK, s[i-1]), 1)

def _replace(s, i, c, N):
    if WEIGHT_TYPE == 1:
        w = s[:i] + c + s[i+1:]
        return _get_cost(s, w, i, N)
    elif WEIGHT_TYPE == 2:
        if s[i-1] == c:
            return 0.0
        else:
            return 1/WEIGHT_MATRIX.get((s[i-1], c), 1)

def _transposition(s, i, N):
    if WEIGHT_TYPE == 1:
        w = s[:i-2] + s[i] + s[i-1] + s[i+1:]
        return _get_cost(s, w, i, N)
    elif WEIGHT_TYPE == 2:
        return 1/WEIGHT_MATRIX.get((s[i-1:i+1], s[i-1:i+1:-1]), 1)

def weditdist(s1, s2, N=1):
    """This is actual weghted edit dist which takes WEIGHT_MATRIX into
    consideration.
    """
    # remove if some part of the prefix of the string is common
    s1, s2 = KB.word_to_keyseq(s1), KB.word_to_keyseq(s2)
    i = 0
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        if c1 != c2:
            break
    s1 = s1[i:]
    s2 = s2[i:]

    n1, n2 = len(s1) + 1, len(s2) + 1
    A = np.zeros(shape=(n1, n2))

    A[0, 0] = 0
    for i in range(1, n1):
        A[i, 0] = A[i-1, 0] + _delete(s1, i-1, N)
    for j in range(1, n2):
        A[0, j] = A[0, j-1] + _insert(s2, j-1, N)

    def _transposition_cost(i, j):
        if all((i > 1, j > 1, s1[i-1] != s2[j-1],
                s1[i-2] == s2[j-1], s1[i-1] == s2[j-2])):
            return A[i-2, j-2] + _transposition(s1, i, N)
        else:
            return A[i-1, j-1] + 2

    for i in range(1, n1):
        for j in range(1, n2):
            if s1[i-1] == s2[j-1]:
                A[i, j] = A[i-1, j-1]
            else:
                A[i, j] = min(
                    A[i-1, j] + _delete(s1, i, N),
                    A[i, j-1] + _insert(s2, j, N),
                    A[i-1, j-1] + _replace(s1, i, s2[j-1], N),
                    _transposition_cost(i, j)
                )
    if DEBUG:
        dp(n1=n1, n2=n2, i=i, j=j)
        print('\n'.join(str(a) for a in A))
    return A[-1, -1]

def _editdist(s1, s2, limit=2, is_get_edits=True):
    """
    Edit distance to convert s1 to s2. So basically, what I have to
    insert/delete/replace to *CREATE s2 FROM s1*

     \s2
    s1\-------------------
      | (i-1,j-1)   (i-1,j)
      | (i  ,j-1)   (i  ,j)

    It also returns all the possible alignments.

    """
    n1, n2 = len(s1) + 1, len(s2) + 1
    if abs(n1-n2) > limit:
        return (limit, [])

    allowed_edits = ('', 'd', 'i', 'r', 't')
    op_dict = {c: i for i, c in enumerate(allowed_edits)}
    index_changes = [(-1, -1), (-1, 0), (0, -1), (-1, -1), (-2, -2)]
    t_edits = np.array([op_dict[e] for e in ('d', 'i', 'r', 't', '')])
    A = np.zeros(shape=(n1, n2))
    B = np.zeros(shape=(n1, n2, len(op_dict)))

    A[0, 0] = 0
    B[0, 0, op_dict['']] = 1
    for i in range(1, n1):
        A[i, 0] = A[i-1, 0] + 1
        B[i, 0, op_dict['d']] = 1
    for j in range(1, n2):
        A[0, j] = A[0, j-1] + 1
        B[0, j, op_dict['i']] = 1

    def _transposition_cost(i, j):
        if all((i > 1, j > 1, s1[i-1] != s2[j-1],
                s1[i-2] == s2[j-1], s1[i-1] == s2[j-2])):
            return A[i-2, j-2] + 1
        else:
            return A[i-1, j-1] + 2

    for i in range(1, n1):
        for j in range(1, n2):
            costs = (
                A[i-1, j] + 1, # deletion
                A[i, j-1] + 1, # insertion
                A[i-1, j-1] + 1, # replace
                _transposition_cost(i, j),  # transposition
                A[i-1, j-1] + (1 if s1[i-1] != s2[j-1] else 0), # nothing
            )
            min_cost = min(costs)
            possible_edits = np.array([
                e for ei, e in enumerate(t_edits) if costs[ei] <= min_cost
            ])
            A[i, j] = min_cost
            B[i, j, possible_edits] = 1

    if DEBUG:
        dp(n1=n1, n2=n2, i=i, j=j)
        print('\n'.join(str(a) for a in A))
    ret = []
    def get_all_edits(r, c, w1='', w2=''):
        e = B[r, c]
        for ei in e.nonzero()[0]:
            ic = index_changes[ei]
            iprime, jprime = r + ic[0], c + ic[1]
            w1prime = s1[r-1] + w1 if ei != op_dict['i'] else BLANK + w1
            w2prime = s2[c-1] + w2 if ei != op_dict['d'] else BLANK + w2
            if iprime > 0 or jprime > 0:
                get_all_edits(iprime, jprime, w1prime, w2prime)
            else:
                ret.append((w1prime, w2prime))

    w = A[n1-1, n2-1]
    if w > limit or not is_get_edits:
        return min(limit, w), []
    if w > 0:
        get_all_edits(n1-1, n2-1)
    else:
        ret = [(s1, s2)]
    ret = list(set(ret))
    if DEBUG:
        if w == 1 and len(ret) > 1:
            x = set(
                (c1, c2)
                for r1, r2 in ret
                for c1, c2 in zip(r1, r2) if c1 != c2 and not confusions(c1, c2)
            )
            if len(x) > 1:
                print("OMG!!!! <<ed={}>>> {}\n{}\n".format(w, ret, x), 
                      file=sys.stderr)
    return w, ret


def align(s1, s2, limit=2):
    """
    Aligns s1 and s2 according to the best alignment it can think of.
    :param s1: a str
    :param s2: a str
    :param try_breaking_replace: boolean, denoting whether or not to consider
    replace as a delete and insert
    :return: pair of strings of equal length, with added blanks for 
             insert and delte.
    """
    assert isinstance(s1, str) and isinstance(s2, str), \
        "The input to align should be two strs only it will apply the " \
        "keypress function by itself."

    # print("{!r} <--> {!r}".format(s1, s2))
    s1 = ''.join(c for c in s1 if c in ALLOWED_CHARACTERS)
    s2 = ''.join(c for c in s2 if c in ALLOWED_CHARACTERS)
    s1k = KB.word_to_keyseq(s1)
    s2k = KB.word_to_keyseq(s2)
    if s1.swapcase() == s2:  # I don't know how else to fix this
        s2k = CAPS_KEY + s1k
    w, edits = _editdist(s1k, s2k, limit=limit)
    return w, edits

def is_series_insertion(s, t):
    #  s,t = ''.join(s), ''.join(t)
    regex = '^(?P<front>{0}{{2,}})|(?P<back>{0}{{2,}})$'\
        .format(re.escape(BLANK))
    m = re.search(r'%s' % regex, s)
    if m:
        front, back = m.groups(['front', 'back'])
        t_priv = t[m.start():m.end()]
        if front:
            s_ = s[m.end():]
            t_ = t[m.end():]
        elif back:
            s_ = s[:m.start()]
            t_ = t[:m.start()]
        else:
            s_, t_ = '', ''
        return s_, t_, t_priv
    return s, t, None


def _all_pairs(i, contextsize, arrlen):
    """
    i: index in the array
    contextsize: size of the context around i
    arrlen: length of the array

    Returns iterator for index-tuples near i in a list of size s with
    context size @contextsize.  Context of k around i-th index means all
    substrings/subarrays of a[i-N:i+N+1] containing a[i]. 
    """
    return [ 
        (l,r) 
        # start anywhere between i - contextsize to i
        for l in range(max(i-contextsize, 0), i+1)
        # end anywhere between i to i + contextsize
        for r in range(i+1, min(i + 1 + contextsize, arrlen))
    ]


def _extract_edits(w, s, N=1):
    """
    It extracts the edits between two strings @w and @s.
    @N denotes the amount of history and future that will
    be used while considering edits. E.g.,
    @w = 'raman', @s = 'Ramn'
    First will align the texts, 'raman' <--> 'Ram*n'
    Edits:
        r->R, ^r -> ^R, ^ra -> ^Ra, a->*, ma -> m, man-> m*n
        an -> *n
    @returns: an array of edit tuple, e.g. [('r', 'R'), ('^r', '^R')..] etc.
    NOTE: ^ and $ are used to denote start and end of a string.
    In code we use "\1" and "\2".
    """
    assert isinstance(w, str) and isinstance(s, str),\
        "w ({!r}) or s ({!r}) is not string".format(w, s)
    assert len(w) == len(s), \
        "Length of w (%s - %d) and s (%s - %d) are not equal."\
        % (w, len(w), s, len(s))
    assert N >= 0, "N must be >= 0"
    if w == s:  # ******* REMEMBER THIS CHECK **********
        print("w ({!r}) == s ({!r})".format(w, s))
        return []  # No edit if strings are equal

    w = STARTSTR + w + ENDSTR
    s = STARTSTR + s + ENDSTR
    wl = len(w)  # or len(s) they are equal
    E = [
        (w[l:r], s[l:r]) 
        for i, (c, d) in enumerate(zip(w, s))
        for (l, r) in _all_pairs(i, N, wl)
        if c != d
    ]
    # if c == d or confusions(c, d): continue
    # for j in range(-N, 1):  # start anywhere between i-N to i
    #     for k in range(1, max(3, N + 2)):  # end anywhere between i to i+N
    #         if i+j >= 0 and i+k <= l:
    #             E.append((w[i + j:i + k], s[i + j:i + k]))
    return E


def all_edits(orig, typed, N=1, edit_cutoff=2):
    if DEBUG:
        print("{0:*^30} --> {1:*^30}".format(orig, typed))
    w, alignments = align(orig, typed)
    if w > edit_cutoff:
        print("Ignoring: {!r} <--> {!r}\t: {}"
              .format(orig, typed, w, ),  file=sys.stderr)

        return []   # ignore
    return [
        pair
        for a in alignments
        for pair in _extract_edits(a[0], a[1], N=N)
    ]

    # # IGNORE THE FOLLOWING
    # # modify series insertion
    # s, t, typed_priv = is_series_insertion(s, t)
    # s_priv, t_priv = '', ''
    # if DEBUG:
    #     print("Typed Extra:", typed_priv)
    # if typed_priv:
    #     s_priv, t_priv = align(orig, typed_priv)
    #     # remove_end_deletes
    #     s_priv, t_priv, _ = is_series_insertion(s_priv, t_priv)
    #     num_edits = len([1 for x, y in zip(s_priv, t_priv) \
    #                      if x != y and not confusions(x, y)])
    #     if num_edits > MAX_ALLOWED_EDITS:
    #         s_priv = ''
    #         t_priv = ''
    # # print("%s\t\t%s\n%s\t\t%s" % (s_priv, s, t_priv, t))
    # return _extract_edits(s_priv, t_priv, N=N) + _extract_edits(s, t, N=N)


def count_edits(L):
    W_Matrix = defaultdict(list)
    for i, (orig, typed) in enumerate(L):
        if DEBUG:
            print(orig, typed,)
        for c, d in all_edits(orig, typed, N=1):
            if DEBUG:
                print(str((c, d)),)
            # W_Matrix[c][d] = W_Matrix[c].get(d, []).append((orig, typed))
            W_Matrix[(c, d)].append(i)
    # for x, d in W_Matrix.items():
    #     f = float(sum(d.values()))
    #     for k in d:
    #         d[k] /= f
    return W_Matrix


def generate_weight_matrix(L):
    E = {k: len(set(v)) for k, v in count_edits(L).items()}
    E = sorted(E.items(), key=lambda x: x[1], reverse=True)
    with open('{}/weight_matrix.py'.format(THIS_FOLDER), 'w') as f:
        f.write('WEIGHT_MATRIX = {}'.format(repr(E)))



def sample_typos(rpw, n):
    """
    Samples 'n' typos for 'rpw' according to the distribution induced
    by WEIGHT_MATRIX.
    The samplig procedure is as follows.
    Pick a random location in the string, pick possible edits with
    left hand side contining the character. Checkout the "possible_edits"
    variable bellow for more details.
    """
    if n<=0: return []
    w = KB.word_to_keyseq(rpw)
    pos_i = random.randint(0, len(w))
    w = STARTSTR + w + ENDSTR
    c = w[pos_i+1]
    possible_edits = {c, BLANK + c, c + BLANK, w[pos_i:pos_i + 2],
                      w[pos_i + 1:pos_i + 3], w[pos_i:pos_i + 3]}

    allowed_arr = [
        ((l, r), f)
        for l in possible_edits
        for r, f in M.get(l, {}).items()
        # ((l,r), v) for (l, r), v in WEIGHT_MATRIX.items()
        # if all((l in possible_edits, ENDSTR not in r, STARTSTR not in r))
    ]
    if not allowed_arr:
        return []
    edits, freqs = zip(*allowed_arr)
    # keys = np.array(keys)
    pdist = np.array(freqs)/float(sum(freqs))
    indxs = np.arange(pdist.shape[0], dtype=int)
    actual_n = n
    n = min(pdist.shape[0], n)
    ret = ['' for _ in range(n)]
    samples = np.random.choice(
        indxs, n, p=pdist,
        replace=False
    )
    sorted_samples = indxs[np.argsort(pdist[samples])]

    for i, ti in enumerate(sorted_samples):
        l, r = edits[ti]
        j = (pos_i + 1) if l[0] == c else pos_i
        k = j + len(l.replace(BLANK, ''))
        # print("{!r} -> {!r}".format(l, r))
        tpw = w[:j] + r + w[k:]
        tpw = tpw.replace(BLANK, '').lstrip(STARTSTR).rstrip(ENDSTR)
        tpw = KB.keyseq_to_word(tpw)
        ret[i] = tpw
    if n<actual_n:
        return ret + sample_typos(rpw, actual_n-n)
    else:
        return ret

def clean_str(s):
    assert isinstance(s, str), "{} -> {}".format(s, type(s))
    return ''.join(
        c for c in s
        if ord(c)>7 and ord(c) <= 255
    )


if __name__ == '__main__':
    # unittest.main()
    import pandas as pd
    USAGE = """Usage:
$ {} [options] [arguments]

-train <fname>: trains the model, and generates weigt_matrix.py
-sample <password>: samples typos for the password from the model
-prob <rpw> <tpw>: probability of rpw -> tpw
-random: Do random things! (Not recommended).
-edist <rpw> <tpw> : Returns the edist between rpw and tpw
        """.format(sys.argv[0])
    if (len(sys.argv)<2):
        print(USAGE)
        exit(1)
    load_weight_matrix()
    if sys.argv[1] == '-train':
        train_f = sys.argv[2]
        d = pd.read_csv(train_f, skipinitialspace=False)\
              .astype(str)
        d_tr = d[d.rpw != d.tpw].sample(int(0.8*len(d.index)), random_state=345)
        d_ts = d[d.rpw != d.tpw].sample(int(0.2*len(d.index)))
        L = zip(d_tr.rpw.apply(clean_str), d_tr.tpw.apply(clean_str))
        print("Got {} typos from {}".format(len(L), sys.argv[1]))
        generate_weight_matrix(L)
    elif sys.argv[1] == '-sample':
        pw = sys.argv[2]
        print("{} --> {}".format(pw, '\n'.join(sample_typos(pw, 10))))
    elif sys.argv[1] == '-random':
        L = [
            ('!!1303diva', '!!1303DIVA'),
            ('principle', 'prinncipal'),
            ('IL0VEMIKE!', 'ILOVEMIKE!'),
            ('MICHAEL', 'michale'),
        ]
        for w1, w2 in L:
            print("{} <--> {}: {}".format(w1, w2, align(w1, w2)))
        w, s = align(
            'GARFIELD',
            'garfied'
        )
        print(w)
        print(s)
        print(all_edits(w,s))

        for rpw, tpw in L:
            s1 = ''.join(c for c in rpw if c in ALLOWED_KEYS)
            s2 = ''.join(c for c in tpw if c in ALLOWED_KEYS)
            s1k = KB.word_to_keyseq(s1)
            s2k = KB.word_to_keyseq(s2)
            print("{} <--> {}: {}".format(rpw, tpw, weditdist(s1k, s2k, N=1)))
        # print('\n'.join(str(x) for x in L))
        # E = {str((k,x)): v[x] for k,v in count_edits(L).items()
        #     for x in v if x.lower()!=k.lower()}
    elif sys.argv[1] == "-edist":
        print(_editdist(sys.argv[2], sys.argv[3]))
        print(KB.dl_distance(sys.argv[2], sys.argv[3]))
        print([KB.change_shift(ord(ch)) for ch in sys.argv[2]])
        print(KB.loc(ord(b'q')))
    else:
        print(USAGE)
