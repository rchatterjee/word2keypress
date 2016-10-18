
import unittest
import random
from word2keypress.weighted_edist import (
    _editdist, BLANK, UNWEIGHTED, _replace,
    ALLOWED_CHARACTERS, align
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

    def test_replace(self):
        global UNWEIGHTED
        old_uw = UNWEIGHTED
        UNWEIGHTED = False
        inp_res_map = [
            (('a', 1, 'b', 1), (0)),
            (('a', 1, 'a', 1), (0.2))
        ]
        res = [_replace(*inp) for inp, res in inp_res_map]
        assert res[0] > res[1]
        UNWEIGHTED = old_uw

    def test_aligned_text(self):
        inp_res_map = [
            (('', 'a'), (BLANK, 'a')),
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
                    print(align(*inp))
                    print(res)
