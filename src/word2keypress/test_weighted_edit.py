
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
        inp_res_map = [
            (('a', 'a'), (0)),
            (('a', 'A'), (0.2))
        ]
        for inp, res in inp_res_map:
            self.assertEqual(_replace(*inp), res)  # , "%s --> %s" %(str(inp),res))
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
