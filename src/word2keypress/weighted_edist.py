#!/usr/bin/python
import sys, re, string, csv
from pprint import pprint
from word2keypress import Keyboard
from collections import defaultdict
import numpy as np
import random
try:
    from word2keypress.weight_matrix import WEIGHT_MATRIX
except ImportError:
    from weight_matrix import WEIGHT_MATRIX


KB = Keyboard('qwerty')
SHIFT_KEY = chr(3) # [u'\x03', "<s>"][user_friendly]
CAPS_KEY = chr(4) # [u'\x04', "<c>"][user_friendly]

ALLOWED_KEYS = b"`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./ {}{}"\
    .format(SHIFT_KEY, CAPS_KEY)
ALLOWED_CHARACTERS = string.printable[:-5]
_allowed_key_set = set(ALLOWED_KEYS)
BLANK, STARTSTR, ENDSTR = chr(1), chr(2), chr(5)
MAX_ALLOWED_EDITS = 3
UNWEIGHTED = True
DEBUG = False

def dp(**kwargs):
    print kwargs


CONFUSING_SETS = [set(['1', 'l', 'I']),
                  set(['o', '0', 'O'])]
def confusions(w, s):
    for cset in CONFUSING_SETS:
        if w in cset and s in cset:
            return True
    return False


def fill_weight_matrix():
    for k in WEIGHT_MATRIX:
        s, d = k
        WEIGHT_MATRIX[k] = 0 if s == d else \
            1.1 if KB.is_keyboard_close(s, d) \
                else 2.1


def update_weight_matrix(weight_matrix):
    for rc in ALLOWED_KEYS:
        total = sum(weight_matrix.get((rc, x), 0.0)+WEIGHT_MATRIX[(rc, x)]
                    for x in ALLOWED_KEYS)
        for tc in ALLOWED_CHARACTERS:
            WEIGHT_MATRIX[(rc, tc)] = total/weight_matrix[(rc, tc)]


def _get_cost(s, w, i, N):
    l = min(len(s), len(w))
    prob = 0.0
    for j in xrange(-N, 1):  # start anywhere between i-N to i
        for k in xrange(1, max(2, N + 1)):  # end anywhere between i to i+N
            if i+j>=0 and i+k<=l:
                prob += WEIGHT_MATRIX.get(s[i + j:i + k], {})\
                                     .get(w[i + j:i + k], 0.0)
    return prob
    
def _delete(s, i, N):
    """The cost of deleting 'i'-th character from s. Also takes into
    account the context.
    """
    w = s[:i] + BLANK + s[i+1:]
    return _get_cost(s, w, i, N)

def _insert(s, i, N):
    """The cost of inserting a character at 'i'-th character from s. Also
    takes into account the context.
    """
    w = s[:i] + BLANK + s[i+1:]
    return _get_cost(s, w, i, N)

def _replace(s, i, c, N):
    w = s[:i] + c + s[i+1:]
    return _get_cost(s, w, i, N)

def weditdist(s1, s2, N=1):
    """This is actual weghted edit dist which takes WEIGHT_MATRIX into
    consideration.
    """
    # remove if some part of the prefix of the string is common 
    i = 0
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        if c1!=c2:
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

    for i in range(1, n1):
        for j in range(1, n2):
            if s1[i-1] == s2[j-1]:
                A[i, j] = A[i-1, j-1]
            else:
                A[i, j] = min(
                    A[i-1, j] + _delete(s1, i, N),
                    A[i, j-1] + _insert(s2, j, N),
                    A[i-1, j-1] + _replace(s1, i, s2[j-1], N)
                )
    if DEBUG:
        dp(n1=n1, n2=n2, i=i, j=j)
        print '\n'.join(str(a) for a in A)
    return A[-1, -1]

def _editdist(s1, s2):
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
    A = np.zeros(shape=(n1, n2))
    B = np.zeros(shape=(n1, n2, 4))
    op_dict = {'': 0, 'd': 1, 'i': 2, 'r': 3}
    index_changes = [(-1, -1), (-1, 0), (0, -1), (-1, -1)]
    t_edits = np.array([op_dict[e] for e in ('d', 'i', 'r', '')])

    A[0, 0] = 0
    B[0, 0, op_dict['']] = 1
    for i in range(1, n1):
        A[i, 0] = A[i-1, 0] + 1
        B[i, 0, op_dict['d']] = 1
    for j in range(1, n2):
        A[0, j] = A[0, j-1] + 1
        B[0, j, op_dict['i']] = 1

    for i in range(1, n1):
        for j in range(1, n2):
            costs =  (
                (A[i-1, j] + 1), 
                (A[i, j-1] + 1), 
                (A[i-1, j-1] + 1),
                (A[i-1, j-1] + (1 if s1[i-1]!=s2[j-1] else 0))
            )
            min_cost = min(costs)
            possible_edits = np.array([
                e for ei, e in enumerate(t_edits) if costs[ei]<=min_cost
            ])
            A[i, j] = min_cost
            B[i, j, possible_edits] = 1

    if DEBUG:
        dp(n1=n1, n2=n2, i=i, j=j)
        print '\n'.join(str(a) for a in A)
    ret = []

    def get_all_edits(r, c, w1='', w2=''):
        e = B[r, c]
        for ei in e.nonzero()[0]:
            ic = index_changes[ei]
            iprime, jprime = r + ic[0], c + ic[1]
            w1prime = s1[r-1] + w1 if ei!=op_dict['i'] else BLANK + w1
            w2prime = s2[c-1] + w2 if ei!=op_dict['d'] else BLANK + w2
            if iprime>0 or jprime>0:
                get_all_edits(iprime, jprime, w1prime, w2prime)
            else:
                ret.append((w1prime, w2prime))

    w = A[n1-1, n2-1]
    if w>0:
        get_all_edits(n1-1, n2-1)
    else:
        ret = [(s1, s2)]
    if w==1 and len(ret)>1:
        x = set((c1, c2)
                for r1, r2 in ret
                for c1, c2 in zip(r1, r2) if c1 != c2 and not confusions(c1, c2))
        if len(x)>1:
            sys.stderr.write("OMG!!!! <<ed={}>>> {}\n{}\n".format(w, ret, x))
    return w, ret

    
def align(s1, s2):
    """
    Aligns s1 and s2 according to the best alignment it can think of.
    :param s1: a basestring object
    :param s2: a basestring object
    :param try_breaking_replace: boolean, denoting whether or not to consider
    replace as a delete and insert
    :return: pair of strings of equal length, with added blanks for insert and delte.
    """
    assert isinstance(s1, basestring) and isinstance(s2, basestring), \
        "The input to align should be two basestrings only it will apply the keypress" \
        "function by itself."

    # print "{!r} <--> {!r}".format(s1, s2)
    s1 = ''.join(c for c in s1 if c in _allowed_key_set)
    s2 = ''.join(c for c in s2 if c in _allowed_key_set)
    s1k = KB.word_to_keyseq(s1)
    s2k = KB.word_to_keyseq(s2)

    w, edits = _editdist(s1k, s2k)
    return w, edits

def is_series_insertion(s, t):
    #  s,t = ''.join(s), ''.join(t)
    regex = '^(?P<front>{0}{{2,}})|(?P<back>{0}{{2,}})$'.format(re.escape(BLANK))
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

        return s_, t_, t_priv
    return s, t, None


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
    assert len(w) == len(s), \
        "Length of w (%s-%d) and s (%s - %d) are not equal." % (w, len(w), s, len(s))
    assert N >= 0, "N must be >= 0"
    if w == s:  # ******* REMEMBER THIS CHECK **********
        return []  # No edit if strings are equal

    w = STARTSTR + w + ENDSTR
    s = STARTSTR + s + ENDSTR
    l = min(len(w), len(s))
    E = []
    e_count = 0
    for i, t in enumerate(zip(w, s)):
        c, d = t
        if c != d and not confusions(c, d):
            e_count += 1
            for j in xrange(-N, 1):  # start anywhere between i-N to i
                for k in xrange(1, max(2, N + 1)):  # end anywhere between i to i+N
                    if i+j>=0 and i+k<=l:
                        E.append((w[i + j:i + k], s[i + j:i + k]))

    return E if e_count < 5 else []


def all_edits(orig, typed, N=1, edit_cutoff=2):
    if DEBUG:
        print "{0:*^30} --> {1:*^30}".format(orig, typed)
    w, alignments = align(orig, typed)
    if w>edit_cutoff:
        return []   # ignore 
    return [
        pair 
        for a in alignments
        for pair in _extract_edits(a[0], a[1])
    ]

    # IGNORE THE FOLLOWING
    # modify series insertion
    s, t, typed_priv = is_series_insertion(s, t)
    s_priv, t_priv = '', ''
    if DEBUG:
        print "Typed Extra:", typed_priv
    if typed_priv:
        s_priv, t_priv = align(orig, typed_priv)
        # remove_end_deletes
        s_priv, t_priv, _ = is_series_insertion(s_priv, t_priv)
        num_edits = len([1 for x, y in zip(s_priv, t_priv) if x != y and not confusions(x, y)])
        if num_edits > MAX_ALLOWED_EDITS:
            s_priv = ''
            t_priv = ''
    # print "%s\t\t%s\n%s\t\t%s" % (s_priv, s, t_priv, t)
    return _extract_edits(s_priv, t_priv, N=N) + _extract_edits(s, t, N=N)


def count_edits(L):
    W_Matrix = defaultdict(list)
    for i, (orig, typed) in enumerate(L):
        if DEBUG:
            print orig, typed,
        for c, d in all_edits(orig, typed, N=1):
            if DEBUG:
                print str((c, d)),
            # W_Matrix[c][d] = W_Matrix[c].get(d, []).append((orig, typed))
            W_Matrix[(c, d)].append(i)
    # for x, d in W_Matrix.items():
    #     f = float(sum(d.values()))
    #     for k in d:
    #         d[k] /= f

    return W_Matrix


import unittest


class Test_Editdistance(unittest.TestCase):
    def test__editdistance_basic(self):
        global UNWEIGHTED
        old_uw = UNWEIGHTED
        UNWEIGHTED = True
        inp_res_map = [(('', ''), (0.0, [])),
                       (('', 'a'), (2.5, [(-1, 0, 'i')])),
                       (('a', 'a'), (0.0, [])),
                       (('a', 'b'), (3.0, [(0, 0, 'r')])),
                       (('aaa', 'aa'), (2.0, [(1, 0, 'd')])),
                       (('a', 'aa'), (1.5, [(0, 1, 'i')])),
                       (('abcd', 'a'), (6.0, [(1, 0, 'd'), (2, 0, 'd'), (3, 0, 'd')])),
                       (('abcd', 'Abc'), (2.2, [(0, 0, 'r'), (3, 2, 'd')])),
                       ]

        for inp, res in inp_res_map:
            res_ = _editdist(*inp)
            self.assertEqual(res_, res, "%s --> %s {{%s}}" % (inp, res, res_))
        UNWEIGHTED = old_uw

    def test_replace(self):
        global UNWEIGHTED
        old_uw = UNWEIGHTED
        UNWEIGHTED = False
        inp_res_map = [(('a', 'a'), (0)),
                       (('a', 'A'), (0.2))
                       ]
        for inp, res in inp_res_map:
            self.assertEqual(_replace(*inp), res)  # , "%s --> %s" %(str(inp),res))
        UNWEIGHTED = old_uw

    def test_aligned_text(self):
        inp_res_map = [(('', 'a'), (BLANK, 'a')),
                       (('a', ''), ('a', BLANK)),
                       (('abcd', 'ABCD'), ('abcd', 'ABCD')),
                       (('aAaA', 'AaA'), ('aAaA', '\0AaA')),
                       ]

        for n in xrange(6):
            a = list(ALLOWED_CHARACTERS)
            random.shuffle(a)
            t = a[:n + 1]
            s = t[:]
            for _ in xrange(random.randint(1, n + 1)):  # how many edit
                loc = random.randint(0, len(s))
                if random.randint(0, 1) == 0:  # what edit -- 0: insert
                    s.insert(loc, a.pop())
                    t.insert(loc, BLANK)
                else:  # 1: delete
                    t.insert(loc, a.pop())
                    s.insert(loc, BLANK)
            s, t = ''.join(s), ''.join(t)
            inp_res_map.append(((s.replace(BLANK, ''),
                                 t.replace(BLANK, ''), True), (s, t)))
        for inp, res in inp_res_map:
            try:
                self.assertEqual(align(*inp), res)
            except:
                if len(res[1]) > 5:
                    continue
                else:
                    print align(*inp)
                    print res

def generate_weight_matrix(L):
    E = {k: len(set(v)) for k, v in count_edits(L).items()}
    E = sorted(E.items(), key=lambda x: x[1], reverse=True)
    with open('weight_matrix___.py', 'w') as f:
        f.write(repr(E))

    
if __name__ == '__main__':
    # fill_weight_matrix()
    # unittest.main()
    w1 = 'principle'
    w2 = 'prinncipal'
    print "{} <--> {}: {}".format(w1, w2, weditdist(w1, w2))
    # w, s = aligned_text(KB.key_presses('GARFIELD'), KB.key_presses('garfied'))
    # print w
    # print s
    # print all_edits(w,s
    L = [(row['rpw'], row['tpw']) for row in csv.DictReader(open(sys.argv[1], 'rb'))]
    for rpw, tpw in L:
        s1 = ''.join(c for c in rpw if c in _allowed_key_set)
        s2 = ''.join(c for c in tpw if c in _allowed_key_set)
        s1k = KB.word_to_keyseq(s1)
        s2k = KB.word_to_keyseq(s2)
        print "{} <--> {}: {}".format(rpw, tpw, weditdist(s1k, s2k, N=1))
    # print '\n'.join(str(x) for x in L)
    # E = {str((k,x)): v[x] for k,v in count_edits(L).items()
    #     for x in v if x.lower()!=k.lower()}
