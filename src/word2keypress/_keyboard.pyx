# cython: boundscheck=False, c_string_type=str
# cython: infer_types=True, c_string_encoding=utf-8
# cython: cdivision=True, profile=True

from __future__ import print_function

import re, os, sys
from cpython cimport array
import array
import json
cimport cython
from libc.stdlib cimport calloc, free
from word2keypress.adjacency_graphs import adj_graph
cdef int user_friendly = 0
cdef char SHIFT_KEY = 3 # [u'\x03', "<s>"][user_friendly]
cdef char CAPS_KEY = 4 # [u'\x04', "<c>"][user_friendly]
cdef str KEYBOARD_TYPE = 'qwerty'
import itertools
cdef int debug = 0

layout_matrix = {
    "qwerty": ([
        b"`1234567890-=",
        b"~!@#$%^&*()_+",
        b" qwertyuiop[]\\",
        b" QWERTYUIOP{}|",
        b" asdfghjkl;'\n"[:-1],  # removing \n
        b" ASDFGHJKL:\"\n"[:-1], # removing \n
        b" zxcvbnm,./",
        b" ZXCVBNM<>?",
        b"         ",
        b"         "
    ], 2),

    "MOBILE_ANDROID": ([
        b"qwertyuiop",
        b"QWERTYUIOP",
        b"1234567890",
        b"~`|......",
        b"asdfghjkl",
        b"ASDFGHJKL",
        b"@#$%&-+()",
        b"....^.={}",
        b"zxcvbnm",
        b"ZXCVBNM",
        b"*\"':;!?",
        b"\....[]",
        b"/      .",
        b",_    /.",
        b",<    >.",
        b",<    >."], 4),

    "dvorak": ([
        b"`1234567890[]",
        b"~!@#$%^&*(){}",
        b"',.pyfgcrl/=\\",
        b"\"<>PYFGCRL?+|",
        b"aoeuidhtns-",
        b"AOEUIDHTNS_",
        b";qjkxbmwvz",
        b":QJKXBMWVZ"], 2)
}

cdef unicode _chr(char c):
    return unicode(chr(c))

cdef str safe_chr(char c):
    if c>0 and c<128:
        return chr(c)
    else:
        return ''

cdef Py_ssize_t TWO_AGO = 0
cdef Py_ssize_t ONE_AGO = 1
cdef Py_ssize_t THIS_ROW = 2
cpdef unsigned int _dl_distance(s1, s2):
    """
    Computes the DL distance between two strings s1 and s2.
    This part of the code is copied form
    https://github.com/gfairchild/pyxDamerauLevenshtein/
    """
    # possible short-circuit if words have a lot in common at the
    # beginning (or are identical)
    cdef Py_ssize_t first_differing_index = 0
    while first_differing_index < len(s1) and \
          first_differing_index < len(s2) and \
              s1[first_differing_index] == s2[first_differing_index]:
        first_differing_index += 1

    s1 = s1[first_differing_index:]
    s2 = s2[first_differing_index:]

    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # Py_ssize_t should be used wherever we're dealing with an array index or length
    cdef Py_ssize_t i, j
    cdef Py_ssize_t offset = len(s2) + 1
    cdef unsigned long delete_cost, add_cost, subtract_cost, edit_distance

    # storage is a 3 x (len(s2) + 1) array that stores TWO_AGO, ONE_AGO, and THIS_ROW
    cdef unsigned long * storage = <unsigned long * >calloc(3 * offset,
                                                            sizeof(unsigned long))
    if not storage:
        raise MemoryError()

    try:
        # initialize THIS_ROW
        for i in range(1, offset):
            storage[THIS_ROW * offset + (i - 1)] = i

        for i in range(len(s1)):
            # swap/initialize vectors
            for j in range(offset):
                storage[TWO_AGO * offset + j] = storage[ONE_AGO * offset + j]
                storage[ONE_AGO * offset + j] = storage[THIS_ROW * offset + j]
            for j in range(len(s2)):
                storage[THIS_ROW * offset + j] = 0
            storage[THIS_ROW * offset + len(s2)] = i + 1

            # now compute costs
            for j in range(len(s2)):
                delete_cost = storage[ONE_AGO * offset + j] + 1
                add_cost = storage[THIS_ROW * offset + (j - 1 if j > 0 else len(s2))] + 1
                subtract_cost = storage[ONE_AGO * offset + (j - 1 if j > 0 else len(s2))] + (s1[i] != s2[j])
                storage[THIS_ROW * offset + j] = min(delete_cost, add_cost, subtract_cost)
                # deal with transpositions
                if (i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j] and s1[i] != s2[j]):
                    storage[THIS_ROW * offset + j] = min(
                        storage[THIS_ROW * offset + j],
                        storage[TWO_AGO * offset + j - 2 if j > 1 else len(s2)] + 1
                    )
        edit_distance = storage[THIS_ROW * offset + (len(s2) - 1)]
    finally:
        # free dynamically-allocated memory
        free(storage)

    return edit_distance

_t_keys = b"`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./ {}{}"
if sys.version_info >= (3,):
    _t_keys += bytes([SHIFT_KEY, CAPS_KEY])
else:
    _t_keys += chr(SHIFT_KEY) + chr(CAPS_KEY)

ALLOWED_KEYS = array.array('B', _t_keys)

assert all(map(
    lambda c: (c>=20 and c<128) or c in [3,4], ALLOWED_KEYS
)), ALLOWED_KEYS

cdef class Keyboard(object):
    cdef str _keyboard_type
    # some random parameters, need to learn later
    cdef float _shift_discount
    cdef int _num_shift
    cdef list _keyboard
    cdef dict _loc_map, _adj_map
    cdef set _printables
    def __init__(self,  str _type='qwerty', float shift_discount=0.8):
        self._adj_map = adj_graph
        assert _type in self._adj_map
        self._keyboard_type = _type
        try:
            self._keyboard, self._num_shift = layout_matrix[self._keyboard_type]
        except KeyError as e:
            print("KeyError: {}\n{}".format(self._keyboard_type, e))
            raise(e)
        self._shift_discount = shift_discount
        self._loc_map = {}
        self._printables = set(''.join(k.decode("utf-8")for k in self._keyboard))

        # generated by Dropbox/python-zxcvbn
        assert len(self._keyboard) % self._num_shift==0, "Problem in Keyboard layout. "\
            "Expecting the size of the keyboard ({}) to be multiple of num_shift ({})."\
            .format(len(self._keyboard), self._num_shift)

    def char2key(self, char _char, int capslock_state):
        assert self._keyboard_type == 'qwerty', \
            "Not yet supported non-qwerty keyboards: {}".format(self._keyboard_type)
        cdef int r, c, shift
        r, c, shift = self.loc(_char)
        if capslock_state == 1 and chr(_char).isalpha():   # caps lock is on
            shift = (shift+1) % 2  # No need to press shift
        return shift, self._keyboard[r*self._num_shift][c]

    def remove_shift(self, char _char):
        # only valid chars
        assert _char >= 20 and _char < 128, "_char: {!r}".format(_char)
        cdef int r, c, shift
        r, c, shift = self.loc(_char)
        if shift:
            _char = self.loc2char(r*self._num_shift, c)
        assert _char>=20, "_char: {!r}".format(_char)
        return _char, shift

    def add_shift(self, char _char):
        # only valid chars
        assert _char >= 20 and _char < 128, "_char: {!r}".format(_char)
        cdef int r, c, shift
        r, c, shift = self.loc(_char)
        if not shift:
            _char = self.loc2char(r*self._num_shift+1, c)
        assert _char>=20, "_char: {!r}".format(_char)
        return _char, shift

    def change_shift(self, char _char):
        # only valid chars
        assert _char >= 20 and _char < 128, "_char: {!r}".format(_char)
        if not chr(_char).isalpha(): return _char
        cdef int r, c, shift
        r, c, shift = self.loc(_char)
        cdef nshift = (shift+1) % self._num_shift
        _char = self.loc2char(r*self._num_shift + nshift, c)
        assert _char>=0, "_char: {!r}".format(_char)
        return _char, shift

    def loc(self, char _char):
        """
        return location of a key, the row, column and shift on
        or off.
        """
        assert _char > 0 and _char < 128 # only valid chars
        cdef int i, j, num_shift
        KM, num_shift = self._keyboard, self._num_shift
        if not self._loc_map:
            def map_chr_2to3(ch):
                return ord(ch) if sys.version_info < (3,) else ch

            self._loc_map = {map_chr_2to3(ch): (i/num_shift, j, i % num_shift)
                             for i, row in enumerate(KM)
                             for j, ch in enumerate(row)}
            self._loc_map[map_chr_2to3(b' ')] = (3, 0, 0)
        if _char not in self._loc_map:
            raise ValueError("Could not find location of: {!r} <{}, {}>"\
                             .format(_char, type(_char), repr(chr(_char))))

        return self._loc_map.get(_char, (-1, -1, -1))

    cdef char loc2char(self, int r, int c):
        """
        Given loc (r,c) returns the actual character
        """
        if r>=0 and r<len(self._keyboard):
            if c>=0 and c<len(self._keyboard[r]):
                return ord(self._keyboard[r][c]) if sys.version_info < (3,) \
                    else self._keyboard[r][c]
        print("Got a weird r={} and c={}".format(r,c))
        return 0

    cdef int num_shift(self):
        return self._num_shift

    def is_typable(self, s):
        return len(set(s) - self._printables) == 0

    def keyboard_dist(self, char key_o, char key_t):
        """
        Returns how close the keys are in keyboard
        key_o = original key
        key_w = typed key
        (Though the output is order agnostic :P)
        """
        cdef oi, oj, oshift, ti, tj, tshift
        oi, oj, oshift = self.loc(key_o)
        ti, tj, tshift = self.loc(key_t)
        if debug>2:
            print (key_o, oi, oj, oshift, '>>><<<<',)
            print (ti, tj, tshift, key_t)

        return abs(oi-ti) + abs(oj-tj) + \
            self._shift_discount*abs(oshift-tshift)

    def is_keyboard_prox(self, char s, char d):
        """
        Checks whether two words are close in terms of keys
        :param s: character
        :param d: character
        :return: weight
        """
        return self.keyboard_dist(s, d) < 0

    def keyboard_nearby_chars(self, _char):
        """Returns the closed by characters of character @c in the keyboard.
        :param c: character
        :return: a list of characters
        """
        # Should be inside _keyboard_nearby_keys
        # assert all((ret<128) & (ret>=20)), "{}  (char: {})".format(ret, _char)

        # using the Dropbox adjacency graph
        return ''.join(
            c
            for c in self._adj_map[self._keyboard_type].get(_char, [])
            if c
        )
        # # Old code below. Need to test!!
        # cdef int i, j, shift, sh, r, c
        # cdef char ch
        # i, j, shift = self.loc(ord(_char))
        # ret = []
        # KM, num_shift = self._keyboard, self._num_shift
        # for sh in range(num_shift):
        #     for r in range(i-1, i+2):
        #         for c in range(j-1, j+2):
        #             ch = self.loc2char(r*num_shift+sh, c)
        #             if ch>0 and ch != ord(' ') and ch != _char:
        #                 ret.append(chr(ch))
        # return ret

    def dl_distance(self, w1, w2):
        w1 = self._word_to_keyseq(str(w1))
        w2 = self._word_to_keyseq(str(w2))
        return _dl_distance(w1, w2)

    cdef array.array _keyboard_nearby_keys(self, char _char):
        """Returns the closed by characters of character @c in standard qwerty
        Keyboard.
        :param c: character
        :return: a list of keys
        """
        if _char == SHIFT_KEY:
            return array.array('B', [CAPS_KEY])
        elif _char == CAPS_KEY:
            return array.array('B', [SHIFT_KEY])
        return array.array('B', self.keyboard_nearby_chars(chr(_char)))

        # cdef int i, j, shift, num_shift, r, c
        # cdef char ch
        # i, j, shift = self.loc(_char)
        # KM, num_shift = self._keyboard, self._num_shift
        # ret = array.array('c', list(filter(
        #     lambda x: x and x!=ord(' '),
        #     (self.loc2char(r*num_shift, c)
        #      for r in range(i-1, i+2)
        #      for c in range(j-1, j+2))
        # )))
        # return ret

    def word_to_keyseq(self, word):
        return self._word_to_keyseq(str(word))

    cdef str _word_to_keyseq(self, str word):
        """
        Converts a @word into a key press sequence for the keyboard KB.
        >>> KB = Keyboard('qwerty')
        >>> KB.word_to_keyseqes('Password12!@')
        <s>password12<s>1<s>2
        >>> KB.word_to_keyseqes('pASSWORD')
        <c><s>password
        >>> KB.word_to_keyseqes('PASSword!@')
        <c>pass</c>word<s>1<s>2
        >>> KB.word_to_keyseqes('PAasWOrd') # this is not what it should but close!
        <s>p<s>aas<s>w<s>ord
        <c>pa</c>as<c>wo</c>rd
        """
        caps_key = chr(CAPS_KEY)
        shift_key = chr(SHIFT_KEY)
        assert KEYBOARD_TYPE == 'qwerty', "Not implemented for {!r}"\
            .format(KEYBOARD_TYPE)
        # Add shift keys
        cdef int i, shift
        cdef char ch
        # Add caps in the beginning
        cdef str nword = re.sub(
            r'([A-Z][^a-z]{2,})',
            lambda m: caps_key + m.group(0).lower() + caps_key,
            word
        )
        def unshifted(ch):
            """Takes a str or byte, and return a string"""
            ich = ord(ch) if isinstance(ch, str) else ch
            if ich < 20 or ich >= 128:
                return chr(ich)
            try:
                ich, shift = self.remove_shift(ich)
                return shift_key + chr(ich) if shift else chr(ich)
            except (AssertionError, ValueError) as e:
                return ''
        new_str = ''.join(unshifted(ch) for ch in nword)
        # # finding continuous use of shift and replace that with capslock
        # for s in re.findall(r'(({0}[a-z]){{3,}})'.format(shift_key), new_str):
        #     o_s, _ = s
        #     n_s = re.sub(r'{0}([a-z])'.format(shift_key), r'\1'.format(caps_key), o_s)
        #     new_str = re.sub(re.escape(o_s), '{0}{1}{0}'.format(caps_key, n_s), new_str)

        # # drop <c>a<c> to <s>a
        # new_str = re.sub(r'{0}(.){0}'.format(caps_key),
        #                  r'{}\1'.format(shift_key),
        #                  new_str)

        # # move the last capslock to the end
        # # PASSOWRD123 -> <c>password<c>123 -> <c>password123<c>
        # new_str = re.sub(r'{0}([^a-z]+)$'.format(caps_key),
        #                  r'\1{0}'.format(caps_key),
        #                  new_str)

        # # convert last sequence of shift into caps sequence
        # # passwoRD123 -> passwo<s>r<s>d123 -> passwo<c>rd123<c>
        # # r'(<s>[a-z][^a-z]*)+{2,}$ ->
        # m = re.match(r'.*?(?P<endshifts>({0}[a-z][^a-z{0}]*){{2,}}({0}.[^a-z]*)*)$'\
        #              .format(shift_key), new_str)
        # if m:
        #     s = m.group('endshifts')
        #     ns = caps_key + \
        #          re.sub(r'{0}([a-z])'.format(shift_key), r'\1', s) + \
        #          caps_key
        #     # print m.groups(), ns, s
        #     new_str = new_str.replace(s, ns)

        # drop last <c> before sending
        return new_str.rstrip(caps_key)

    def print_keyseq(self, str keyseq):
        """print the @key_str as the human readable format.
        """
        return keyseq.replace(chr(SHIFT_KEY), '<s>').replace(chr(CAPS_KEY), '<c>')

    def part_keyseq_string(self, str keyseq, shift=False, caps=False):
        """
        returns the word for a part of the keyseq, and returns (word, shift, caps)
        """
        assert KEYBOARD_TYPE == 'qwerty', "Not implemented for mobile"
        cdef str ret = ''
        cdef int i = 0
        cdef char a
        while i<len(keyseq):
            a = ord(keyseq[i])
            if a == CAPS_KEY:
                caps = caps^True
            elif a == SHIFT_KEY:
                shift = True
            else:
                if chr(a).isalpha():
                    a = self.add_shift(a)[0] if caps^shift else a
                else:
                    a = self.add_shift(a)[0] if shift else a
                shift = False
                ret += chr(a)
            i += 1
        return ret, shift, caps

    cdef char _apply_shift_caps(self, char c, shift, caps):
        assert c>=20 and c<128, "Got: {!r}".format(c)
        if chr(c).isalpha() and shift^caps:
            c = self.add_shift(c)[0]
        elif shift:
            c = self.add_shift(c)[0]
        assert c>=20, "Returning: {!r}".format(c)
        return c

    def _sub_word_table(self, str keyseq):
        """
        keyseq must be pure (that is output of _word_to_keyseq function.
        n = len(word), returns an 2-D array,
        TT = shift-caps, both true, TF, FT and FF are similar
        i/j  0     1      2     3       4
        0  [:0] [0:]FF [:0]FT  [:0]TF  [:0]TT
        1  [:1] [1:]FF [:1]FT  [:1]TF  [:1]TT
        .
        .
        n  [:n] [n:]FF [:n]FT  [:n]TF  [:n]TT

        """
        cdef int n = len(keyseq)
        A = [[(), (), (), (), ()] for i in xrange(n+1)]
        A[0] = [('', False, False),
                self.part_keyseq_string(keyseq, False, False),# FF
                self.part_keyseq_string(keyseq, False, True), # FT
                self.part_keyseq_string(keyseq, True, False), # TF
                self.part_keyseq_string(keyseq, True, True)]  # TT
        A[n] = [A[0][1], ('', False, False), ('', False, True), ('', True, False), ('', True, True)]
        cdef int j
        spcl_keys = (SHIFT_KEY, CAPS_KEY)
        cdef char c, nc, shifted_nc
        for j in xrange(1, n):
            last_row = A[j-1]
            row = A[j];
            c = ord(keyseq[j-1]) # current char
            nc = ord(keyseq[j]) # if j<n else 0 # next char
            shifted_nc = self.add_shift(nc)[0] if nc not in spcl_keys else 0
            if c == SHIFT_KEY:
                # case 0: only pre
                row[0] = (last_row[0][0], True, last_row[0][2])
                # case 1: shift-caps = FF, remove the shift from next char
                row[1] = (chr(nc) + last_row[1][0][1:], last_row[1][1], last_row[1][2])
                # case 2: shift-caps = FT
                row[2] = ((safe_chr(shifted_nc) if chr(nc).isalpha() else chr(nc)) + last_row[2][0][1:], last_row[2][1], last_row[2][2])
                # case 2: shift-caps = TF
                row[3] = last_row[3]
                # case 2: shift-caps = TT
                row[4] = last_row[4]
            elif c == CAPS_KEY:
                # case 0: only pre
                row[0] = (last_row[0][0], last_row[0][1], last_row[0][2]^True)
                # shift-caps = FF
                row[1] = (last_row[1][0].swapcase(), last_row[1][1], last_row[1][2]^True)
                # shift-caps = FT
                row[2] = last_row[1]
                # shift-caps = TF
                row[3] = (last_row[3][0].swapcase(), last_row[3][1], last_row[3][2]^True)
                # shift-caps = TT
                row[4] = last_row[3]
            else: # c is not SHIT or CAPS
                row[0] = (
                    last_row[0][0] + \
                    chr(self._apply_shift_caps(
                        c, last_row[0][1], last_row[0][2])
                    ),
                    False, last_row[0][2]
                )
                row[1] = (last_row[1][0][1:], last_row[1][1], last_row[1][2]) # shift-caps = FF
                row[2] = (last_row[2][0][1:], last_row[2][1], last_row[2][2]) # shift-caps = FT
                row[3] = (safe_chr(shifted_nc) + last_row[3][0][2:] if shifted_nc else last_row[3][0][1:],
                          last_row[3][1], last_row[3][2]) # shift-caps = TF
                row[4] = (chr(nc) + last_row[4][0][2:] if chr(nc).isalpha() else
                          safe_chr(shifted_nc) + last_row[4][0][2:],
                          last_row[4][1], last_row[4][2]) # shift-caps = TT
        return A

    def keyseq_insert_edits(self, str keyseq, _insert_keys=[], _replace_keys=[]):
        """It will insert/replace/delete one key at a time from the
        keyseq. And return a set of words. Which keys to insert is
        specified by the @insert_keys parameter.
        :param pos: int, position of the edit, pos=0..len(keyseq):
        insert, delete and replace.
                    if pos=len(keyseq)+1, then only insert
        """
        spcl_keys = array.array('B', [SHIFT_KEY, CAPS_KEY])
        sub_words = self._sub_word_table(keyseq)
        # print(">>>>>", _insert_keys, _replace_keys)
        cdef array.array insert_keys = array.array('B')
        cdef array.array replace_keys = array.array('B')
        insert_keys.fromstring(_insert_keys)
        replace_keys.fromstring(_replace_keys)
        cdef int smart = not (insert_keys or len(insert_keys)>0 or \
                              replace_keys or len(replace_keys)>0)
        cdef int i, t
        cdef char c, k
        cdef str pre_w
        while i<len(keyseq):
            c = ord(keyseq[i])
            if smart:
                # if replacing a caps or shift, replace with everything
                if c in spcl_keys:
                    replace_keys = spcl_keys # ALLOWED_KEYS
                    insert_keys = spcl_keys # ALLOWED_KEYS
                else: # else use only the closed by keys or spcl keys
                    replace_keys = array.copy(self._keyboard_nearby_keys(c))
                    if i==0:
                        insert_keys = replace_keys + spcl_keys
                    else:
                        insert_keys = array.array('B', set.union(
                            set(self._keyboard_nearby_keys(ord(keyseq[i-1]))),
                            set(replace_keys),
                            set([c, SHIFT_KEY])
                        ))
            # print(chr(c), '--->', replace_keys.tostring(), insert_keys.tostring())
            assert all(map(
                lambda ch: 128>ch>=20 or ch in [3,4],
                replace_keys+insert_keys)), \
                "Replace: {}\nInsert: {}".format(replace_keys, insert_keys)

            pre_w, shift, caps  = sub_words[i][0]
            assert ('\x00' not in pre_w), pre_w
            t = 2*shift + caps + 1
            if debug>1:
                print("inserting @", i, pre_w, chr(c), repr(sub_words[i][2*shift+2][0]))
            # insert
            for k in insert_keys:
                if k == SHIFT_KEY:
                    yield pre_w + sub_words[i][3+caps][0]
                elif k == CAPS_KEY:
                    yield pre_w + sub_words[i][2*shift+2][0]
                else:
                    yield pre_w + \
                        chr(self._apply_shift_caps(k, shift, caps)) + \
                        sub_words[i][caps+1][0]
                    # if i == 0:
                    #     yield chr(self._apply_shift_caps(k, True, caps)) + \
                    #         sub_words[i][1][0]
            if debug>1:
                print("deleting @", i)
            # delete
            if c == SHIFT_KEY:
                yield pre_w + sub_words[i+1][1+caps][0]
            elif c == CAPS_KEY:
                yield pre_w + sub_words[i+1][2*shift+2][0]
            else:
                yield pre_w + sub_words[i+1][t][0]
            # replace
            if debug>1:
                print("replacing @", i)
            for k in replace_keys:
                if k == SHIFT_KEY:
                    yield pre_w + sub_words[i+1][3+caps][0]
                elif k == CAPS_KEY:
                    # If already caps, then this will cancel that
                    yield pre_w + sub_words[i+1][2*shift + 2 - caps][0]
                else:
                    yield pre_w + \
                        chr(self._apply_shift_caps(k, shift, caps)) + \
                        sub_words[i+1][1+caps][0]
            i += 1
        # For inserting at the end
        pre_w, shift, caps = sub_words[-1][0]
        if smart:
            insert_keys = ALLOWED_KEYS
        for k in insert_keys:
            if k not in spcl_keys:
                yield pre_w + chr(self._apply_shift_caps(k, shift, caps))
                # yield pre_w + chr(self._apply_shift_caps(k, True, caps))

    def word_to_typos(self, word, _insert_keys=b'', _replace_keys=b''):
        """
        Most highlevel and useful function.
        All possible edits around the word.
        Insert all keys, delete all keys, replace only keys that are close.
        """
        keypress = self._word_to_keyseq(word)
        assert isinstance(keypress, str), "Keypress is expected to be string"\
            "got {}".format(type(keypress))
        insert_keys = array.array('B', _insert_keys)
        replace_keys = array.array('B', _replace_keys)
        return self.keyseq_insert_edits(
            keypress, insert_keys, replace_keys
        )

    def keyseq_to_word(self, keyseq):
        return self._keyseq_to_word(keyseq)

    def _keyseq_to_word(self, str keyseq):
        """This is the same function as word_to_keyseq, just trying to
        make it more efficient. Remeber the capslock and convert the
        shift.
        """
        return self.part_keyseq_string(keyseq)[0]

    def keyseq_to_word_slow(self, char[] keyseq):
        """
        Converts a keypress sequence to a word
        """
        caps_key = CAPS_KEY
        shift_key = SHIFT_KEY
        assert KEYBOARD_TYPE == 'qwerty', "Not implemented for mobile"

        word = keyseq
        def caps_change(m):
            return ''.join(self.change_shift(c)[0] if c !=shift_key else shift_key
                           for c in m.group(1))

        def shift_change(m):
            return ''.join(self.add_shift(c)[0] if c != caps_key else caps_key
                           for c in m.group(1))

        word = re.sub(r'({0})+'.format(shift_key), r'\1', word)
        word = re.sub(r'({0})+'.format(caps_key), r'\1', word)
        # only swap <s><c> to <c><s>
        word = re.sub(r'({1}{0})+([a-zA-Z])'.format(caps_key, shift_key),
                      r'{0}{1}\2'.format(caps_key, shift_key),
                      word)

        if word.count(caps_key)%2 == 1:
            word += caps_key

        try:
            # apply all shift keys
            word = re.sub(r'{0}+([\w\W])'.format(shift_key),
                          shift_change, word)
            # apply all capslocks
            word = re.sub(r'{0}(.*?){0}'.format(caps_key),
                          caps_change, word)
        except Exception, e:
            print (">>>> I could not figure this out: {!r}, stuck at {!r}".format(keyseq, word))
            raise e
        word = word.strip(shift_key).strip(caps_key)
        return word
