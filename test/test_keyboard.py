#!/usr/bin/python
import pyximport; pyximport.install()
import os, sys, json, csv, re
import socket
import random
import pytest
import array
from word2keypress import Keyboard
import time
SHIFT_KEY = 3 # [u'\x03', "<s>"][user_friendly]
CAPS_KEY = 4 # [u'\x04', "<c>"][user_friendly]


kb = Keyboard('qwerty')
def test_keyboard_type():
    with pytest.raises(AssertionError):
        Keyboard('US')
        Keyboard('asdfadsf')
    kb = Keyboard('qwerty')
    kb = Keyboard('dvorak')


class TestKeyboard():
    def test_loc(self):
        inp_res_map = [(('t'), (1,5,0)),
                      (('T'), (1,5,1)),
                      (('a'), (2,1,0)),
                      (('('), (0,9,1)),
                      (('M'), (3,7,1))]
        for q, r in inp_res_map:
            assert kb.loc(ord(*q)) == r

    def test_keyboard_dist(self):
        inp_res_map = [(('t', 't'), (0)),
                       (('t', 'T'), (0.8)),
                       (('a', 'S'), (1.8)),
                       (('a', 'w'), (2)),
                       (('w', '$'), (3.8)),
                       (('<', '>'), (1))
                   ]
        for q, r in inp_res_map:
            q = [ord(x) for x in q]
            assert abs(kb.keyboard_dist(*q)-r)<0.0001

    def test_is_typable(self):
        assert kb.is_typable('adfasdf')
        assert not kb.is_typable(b'\xf3a35')

    @pytest.mark.parametrize(('inp', 'res'), [('a', 'QWSZqwsz'),
                                              ('g', 'TYHBVFtyhbvf'),
                                              ('r', '$%TFDE45tfde')])
    def test_key_prox_chars(self, inp, res):
        ret = [c for c in kb.keyboard_nearby_chars(inp)]
        assert set(ret) == set(res)

    @pytest.mark.skip(reason="cpython function. Not available outside")
    def test_key_prox_keys(self):
        for inp, res in [('a', 'aqwsxz'),
                         ('t', 'tr456yhgf'),
                         (';', ";lop['/.")]:
            ret = kb.keyboard_prox_key(inp)
            assert set(ret) == set(res)

    def test_keypress_to_w(self):
        for inp, res in [('wor{c}d123', 'worD123'),
                         ('{c}pass{c}wo{c}rd{c}', 'PASSwoRD')]:
            w = kb.keyseq_to_word(
                inp.format(s=chr(SHIFT_KEY), c=chr(CAPS_KEY))
            )
            assert w == res


key = {'c': chr(CAPS_KEY),
       's': chr(SHIFT_KEY)}


@pytest.mark.parametrize(
    ('inp', 'res'),
    [('Pa', '{s}pa'),
     ('PAasWOrd', '{s}p{s}aas{s}w{s}ord'),
     ('password', 'password'),
     ('Password', '{s}password'),
     ('P@ssword12', '{s}p{s}2ssword12'),
     ('@!asdASDads', '{s}2{s}1asd{c}asd{c}ads'),
     # There is this small issue, what if there
     # is a shit in the middle of a password
     ('PASSwoRD', '{c}pass{c}wo{c}rd{c}')]
)
class TestKeyPresses():
    @pytest.mark.skip(reason="cpython function. Not available outside")
    def test_word_to_keyseq(self, inp, res):
        t1 = kb.word_to_keyseq(inp)
        t2 = res.format(**key)
        assert t1 == t2, "{!r} <--> {!r}".format(t1, t2)

    def test_keyseq_to_word(self, inp, res):
        assert inp == kb.keyseq_to_word(res.format(**key))

    def test_other_keyseq_to_word(self, inp, res):
        kw = kb.keyseq_to_word('{c}asdf{s}1{c}sdf'.format(**key))
        assert 'ASDFasdf' == kb.keyseq_to_word('{c}asdf{s}a{c}sdf'.format(**key))
        assert 'ASDF!sdf' == kb.keyseq_to_word('{c}asdf{s}1{c}sdf'.format(**key))

    def test_keyseq(self, inp, res):
        inp_res_map = [(('|'), (1, 13, 1))]
        for q, r in inp_res_map:
            assert kb.loc(ord(*q)) == r

    def test_part_keyseq(self, inp, res):
        res = res.format(**key)
        i = random.randint(0, len(res))
        pre_word, shift, caps = kb.part_keyseq_string(res[:i])
        post_word, shift, caps = kb.part_keyseq_string(res[i:], shift, caps)
        assert inp == pre_word + post_word

    def test_sub_word_table(self, inp, res):
        res = res.format(**key)
        A = kb._sub_word_table(res)
        for i in xrange(len(res)):
            pre_w, shift, caps = A[i][0]
            post_w = A[i][2*shift + caps + 1][0]
            assert pre_w + post_w == inp

    @pytest.mark.skip(reason="Not available outside")
    def test_keyseq_insert_edits(self, inp, res):
        inp_res_map = [(
            (
                b'{s}pa'.format(**key),
                [chr(CAPS_KEY), chr(SHIFT_KEY), 'a'],
                [chr(CAPS_KEY), 't']
            ),
            (
                'pA', 'Pa', 'aPa', 'pa', 'PA', 'tpa', # j=0
                'pA', 'Pa', 'Apa', 'A', 'a', 'Ta',   # j=1
                'PA', 'PA', 'Paa', 'P', 'P', 'Pt',   # j=2
                'Paa'
            )
        )]
        for inp, res in inp_res_map:
            inp = (kb.keyseq_to_word(inp[0]), inp[1], inp[2])
            for i, r in enumerate(kb.word_to_typos(*inp)):
                assert r == res[i]

def str_cython_char_array(s):
    return array.array('c', s)

allowed_keys = b"`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./ "
import string
allowed_chars = list(string.printable[:-5])

def test_word_edits(capsys):
    rand_string = ''.join(random.choice(allowed_chars) for _ in range(12))
    # for pw in kb.word_to_typos(rand_string):
    #     print repr(pw)
    _s = set(allowed_chars)
    assert all(ch in _s for ch in rand_string), '{} :: {}'\
        .format(repr(rand_string), set(rand_string)-_s)
    s_t = time.time()
    total_typo_computed = 0
    for i, tpw in enumerate(
            set(kb.word_to_typos(rand_string, allowed_keys, allowed_keys))
    ):
        total_typo_computed += 1
        # print repr(rand_string), repr(tpw)
        assert all(ch in _s for ch in tpw), \
            '{}. {!r} --> {!r} :: {}'.format(i, rand_string, tpw, set(tpw)-_s)
        total_typo_computed += len(set(kb.word_to_typos(tpw, allowed_keys, allowed_keys)))
    e_t = time.time()
    # with capsys.disabled():
    print "\nNumber of typos of {!r}: {:,}".format(rand_string, i)
    print "Total typos covered: {:,}".format(total_typo_computed)
    print ">> Time taken: {:.3f} s".format(e_t-s_t)

def test_edit_distance():
    from word2keypress import distance
    word_pairs = [
        ('password', 'PASSWORD'),
        ('Password', 'password'),
        ('Password', 'PASSWORD'),
        ('Password1', 'Password!'),
        ('pASSWORD', 'Password'),  # This is not good!
        ('P@ssword', 'P2ssword')]
    for w1, w2 in word_pairs:
        assert distance(w1, w2)<2


# class TestPWLogging:
#     def test_logging(self):
#         HOST, PORT = "localhost", 9999
#         sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         sock.settimeout(1)  # wait only 1 msec
#         DB= [('rahulc', 'qwerty'), ('user1', 'Password'), ('user2', 'password'),
#             ('abcd@xyz.com', 'abcd123')]
#         #  clear log file
#         for uid, pw in DB:
#             data = {'uid': uid, 'password': pw, 'useragent': "User-Agent", 'isValid': -1}
#             try:
#                 sock.sendto(json.dumps(data) + "\n", (HOST, PORT))
#                 recvd = sock.recv(1024)
#             except socket.timeout:
#                 print "Cannot reach the logging server."
#             #  TODO - write this test
