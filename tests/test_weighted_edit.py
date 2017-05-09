
import unittest
import random
from word2keypress.weighted_edist import (
    _editdist, BLANK, UNWEIGHTED, _replace,
    ALLOWED_CHARACTERS, align, load_weight_matrix,
    CAPS_KEY, SHIFT_KEY, KB
)

class Test_Editdistance(unittest.TestCase):
    def test__editdistance_basic(self):
        global UNWEIGHTED
        old_uw = UNWEIGHTED
        UNWEIGHTED = True
        inp_res_map = [
            (('', ''), (0.0, [('', '')])),
            (('', 'a'), (1, [(BLANK, 'a')])),
            (('a', 'a'), (0.0, [('a', 'a')])),
            (('a', 'b'), (1.0, [('a', 'b')])),
            (('aaa', 'aa'), (1.0, [
                ('aaa', '{}aa'.format(BLANK)),
                ('aaa', 'a{}a'.format(BLANK)),
                ('aaa', 'aa{}'.format(BLANK)),
            ])),
            (('a', 'ab'), (1, [('a{}'.format(BLANK), 'ab')])),
            (('abcd', 'a'), (3.0, [('abcd', 'a{0}{0}{0}'.format(BLANK))]))
        ]

        for inp, res in inp_res_map:
            res_ = _editdist(inp[0], inp[1], limit=5)
            self.assertEqual(res[0], res_[0])
            self.assertEqual(set(res[1]), set(res_[1]))

        UNWEIGHTED = old_uw

    def random_typos(self):
        for n in range(6):
            a = list(ALLOWED_CHARACTERS)
            random.shuffle(a)
            t = list(KB.word_to_keyseq(''.join(a[:n + 1])))
            s = t[:]
            for _ in range(random.randint(1, n + 1)):  # how many edit
                loc = random.randint(0, len(s))
                # 0: insert, 1: delete, 2: replace, 3: transposition
                edit = random.randint(0, 4)
                ch = KB.word_to_keyseq(a.pop())
                if edit == 0: # insert
                    s.insert(loc, ch)
                    t.insert(loc, BLANK*len(ch))
                elif edit == 1:  # 1: delete
                    t.insert(loc, ch)
                    s.insert(loc, BLANK*len(ch))
                elif edit == 2:
                    s[loc] = ch
                elif n>1 and edit == 3:
                    if loc >= n: loc = n-2
                    t[loc], t[loc-1] = t[loc-1], t[loc]
            s, t = ''.join(s), ''.join(t)
            yield (
                (s.replace(BLANK, ''), t.replace(BLANK, '')),
                (s, t)
            )

    def test_replace(self):
        global UNWEIGHTED
        old_uw = UNWEIGHTED
        UNWEIGHTED = False
        inp_res_map = [
            (('a', 1, 'b', 1), (0)),
            (('a', 1, 'a', 1), (0.2))
        ]
        load_weight_matrix()
        res = [_replace(*inp) for inp, res in inp_res_map]
        assert res[0] > res[1]
        UNWEIGHTED = old_uw

    def test_aligned_text(self):
        inp_res_map = [
            (('', 'a'), (BLANK, 'a')),
            (('a', ''), ('a', BLANK)),
            (('abcd', 'ABCD'), (BLANK + 'abcd', CAPS_KEY + 'abcd')),
            (('aAaA', 'AaA'),
                ('a{s}aa{s}a'.format(s=SHIFT_KEY),
                '{b}{s}aa{s}a'.format(b=BLANK, s=SHIFT_KEY))
             ),
        ]

        for inp, res in inp_res_map:
            try:
                self.assertIn(res, align(*inp)[1])
            except Exception as e:
                raise e
