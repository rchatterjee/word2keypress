#cython: language_level=3, boundscheck=False, c_string_type=str
#cython: infer_types=True, c_string_encoding=utf-8
#cython: cdivision=True, profile=True
import Levenshtein as lv
import re
import numpy as np
cimport numpy as np
cimport cython

cdef int user_friendly = 0
cdef char SHIFT_KEY = 3 # [u'\x03', "<s>"][user_friendly]
cdef char CAPS_KEY = 4 # [u'\x04', "<c>"][user_friendly]
cdef str KEYBOARD_TYPE = 'US'

cdef int debug = 0

layout_matrix = {
    "US": ([
        "`1234567890-=",
        "~!@#$%^&*()_+",
        " qwertyuiop[]\\",
        " QWERTYUIOP{}|",
        " asdfghjkl;'\n"[:-1],  # removing \n
        " ASDFGHJKL:\"\n"[:-1], # removing \n
        " zxcvbnm,./",
        " ZXCVBNM<>?",
        "         ",
        "         "
    ], 2),
    "MOBILE_ANDROID": ([
        "qwertyuiop",
        "QWERTYUIOP",
        "1234567890",
        "~`|......",
        "asdfghjkl",
        "ASDFGHJKL",
        "@#$%&-+()",
        "....^.={}",
        "zxcvbnm",
        "ZXCVBNM",
        "*\"':;!?",
        "\....[]",
        "/      .",
        ",_    /.",
        ",<    >.",
        ",<    >."], 4)
}

cdef unicode _chr(char c):
    return unicode(chr(c))

cdef np.ndarray str_to_numpyarray(str s):
    return np.array([ord(k) for k in s])

cdef str safe_chr(char c):
    if c>0 and c<128:
        return _chr(c)
    else:
        return u''

_t_keys = "`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./ {}{}"\
    .format(chr(SHIFT_KEY), chr(CAPS_KEY))
ALLOWED_KEYS = str_to_numpyarray(_t_keys)
assert all(map(lambda c: (c>=20 and c<128) or c in [3,4], ALLOWED_KEYS)), \
    ALLOWED_KEYS

cdef class Keyboard(object):
    cdef str _keyboard_type 
    # some random parameters, need to learn later
    cdef float _shift_discount
    cdef int _num_shift 
    cdef list _keyboard
    cdef dict _loc_map
    def __init__(self,  str _type='US', float shift_discount=0.8):
        self._keyboard_type = _type
        self._keyboard, self._num_shift = layout_matrix[self._keyboard_type]
        self._shift_discount = shift_discount
        self._loc_map = {}
        assert len(self._keyboard) % self._num_shift==0, "Problem in Keyboard layout. "\
            "Expecting the size of the keyboard ({}) to be multiple of num_shift ({})."\
            .format(len(self._keyboard), self._num_shift)

    def char2key(self, char _char, int capslock_state):
        assert self._keyboard_type == 'US', "Not yet supported non-US keyboards"
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
            self._loc_map = {ord(ch): (i/num_shift, j, i % num_shift)
                             for i, row in enumerate(KM)
                             for j, ch in enumerate(row)}
            self._loc_map[' '] = (3, 0, 0)
        if _char not in self._loc_map:
            raise ValueError("Could not find location of: <{}>"\
                             .format(repr(chr(_char))))
            
        return self._loc_map.get(_char, (-1, -1, -1))

    cdef char loc2char(self, int r, int c):
        """
        Given loc (r,c) returns the actual character
        """
        if r>=0 and r<len(self._keyboard):
            if c>=0 and c<len(self._keyboard[r]):
                return ord(self._keyboard[r][c])
        return 0

    def num_shift(self):
        return self._num_shift

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

    def keyboard_nearby_chars(self, char _char):
        """Returns the closed by characters of character @c in standard US
        Keyboard.
        :param c: character
        :return: a list of characters
        """
        cdef int i, j, shift, sh, r, c
        cdef char ch
        i, j, shift = self.loc(_char)
        ret = []
        KM, num_shift = self._keyboard, self._num_shift
        for sh in xrange(num_shift):
            for r in range(i-1, i+2):
                for c in range(j-1, j+2):
                    ch = self.loc2char(r*num_shift+sh, c)
                    if ch>0 and ch != ord(' ') and ch != _char:
                        ret.append(ch)
        return ret

    cdef np.ndarray _keyboard_nearby_keys(self, char _char):
        """Returns the closed by characters of character @c in standard US
        Keyboard.
        :param c: character
        :return: a list of keys
        """
        if _char == SHIFT_KEY: 
            return np.array([CAPS_KEY], dtype='<I')
        elif _char == CAPS_KEY:
            return np.array([SHIFT_KEY], dtype='<I')
        cdef int i, j, shift, num_shift, r, c
        cdef char ch
        i, j, shift = self.loc(_char)
        KM, num_shift = self._keyboard, self._num_shift
        ret = np.array(filter(
            lambda x: x and x!=ord(' '),
            (self.loc2char(r*num_shift, c)
             for r in range(i-1, i+2)
             for c in range(j-1, j+2))
        ), dtype='<I')
        assert all((ret<128) & (ret>=20)), "{}  (char: {})".format(ret, _char)
        return ret

    def word_to_keyseq(self, word):
        return self._word_to_keyseq(unicode(word))

    cdef str _word_to_keyseq(self, str word):
        """
        Converts a @word into a key press sequence for the keyboard KB.
        >>> KB = Keyboard('US')
        >>> KB.word_to_keyseqes('Password12!@')
        <s>password12<s>1<s>2
        >>> KB.word_to_keyseqes('PASSword!@')
        <c>pass</c>word<s>1<s>2
        >>> KB.word_to_keyseqes('PAasWOrd') # this is not what it should but close!
        <s>p<s>aas<s>w<s>ord
        <c>pa</c>as<c>wo</c>rd
        """
        caps_key = chr(CAPS_KEY)
        shift_key = chr(SHIFT_KEY)
        assert KEYBOARD_TYPE == 'US', "Not implemented for mobile"
        cdef str new_str = ''
        # Add shift keys
        cdef int i, shift
        cdef char ch
        for i in xrange(len(word)):
            ch = word[i]
            try:
                ch, shift = self.remove_shift(ch)
            except ValueError as e:
                print("Bad word found!!", e, repr(word), repr(ch), repr(shift))
                raise e
            if shift:
                new_str += shift_key + chr(ch)
            else:
                new_str += chr(ch)

        # finding continuous use of shift and replace that with capslock
        for s in re.findall(r'(({0}[a-z]){{3,}})'.format(shift_key), new_str):
            o_s, _ = s
            n_s = re.sub(r'{0}([a-z])'.format(shift_key), r'\1'.format(caps_key), o_s)
            new_str = re.sub(re.escape(o_s), '{0}{1}{0}'.format(caps_key, n_s), new_str)

        
        # drop <c>a<c> to <s>a
        new_str = re.sub(r'{0}(.){0}'.format(caps_key),
                         r'{}\1'.format(shift_key),
                         new_str)  

        # move the last capslock to the end
        # PASSOWRD123 -> <c>password<c>123 -> <c>password123<c>
        new_str = re.sub(r'{0}([^a-z]+)$'.format(caps_key),
                         r'\1{0}'.format(caps_key),
                         new_str)  
        
        # convert last sequence of shift into caps sequence
        # passwoRD123 -> passwo<s>r<s>d123 -> passwo<c>rd123<c>
        # r'(<s>[a-z][^a-z]*)+{2,}$ ->
        m = re.match(r'.*?(?P<endshifts>({0}[a-z][^a-z{0}]*){{2,}}({0}.[^a-z]*)*)$'.format(shift_key), new_str)
        if m:
            s = m.group('endshifts')
            ns = caps_key + re.sub(r'{0}([a-z])'.format(shift_key), r'\1', s) + caps_key
            # print m.groups(), ns, s
            new_str = new_str.replace(s, ns)

        return new_str

    def print_keyseq(self, keyseq):
        """print the @key_str as the human readable format.
        """
        return keyseq.replace(SHIFT_KEY, '<s>').replace(CAPS_KEY, '<c>')

    def part_keyseq_string(self, str keyseq, shift=False, caps=False):
        """
        returns the word for a part of the keyseq, and returns (word, shift, caps)
        """
        assert KEYBOARD_TYPE == 'US', "Not implemented for mobile"
        cdef str ret = ''
        cdef int i = 0
        cdef char a
        while i<len(keyseq):
            a = ord(keyseq[i])
            if keyseq[i] == CAPS_KEY:
                caps = caps^True
            elif keyseq[i] == SHIFT_KEY:
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
        cdef char c, nc, shifted_nc
        for j in xrange(1, n):
            last_row = A[j-1]
            row = A[j];
            c = ord(keyseq[j-1]) # current char
            nc = ord(keyseq[j]) # if j<n else 0 # next char
            shifted_nc = self.add_shift(nc)[0] if nc not in [SHIFT_KEY, CAPS_KEY] \
                         else 0
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

    def keyseq_insert_edits(self, str keyseq, 
                               np.ndarray insert_keys=np.array([]), 
                               np.ndarray replace_keys=np.array([])):
        """It will insert/replace/delete one key at a time from the
        keyseq. And return a set of words. Which keys to insert is
        specified by the @insert_keys parameter. 
        :param pos: int, position of the edit, pos=0..len(keyseq): insert,delete and replace.
                    if pos=len(keyseq)+1, then only insert
        """
        spcl_keys = np.array([SHIFT_KEY, CAPS_KEY])
        sub_words = self._sub_word_table(keyseq)
        # print '\n'.join(str(r) for r in sub_words)
        cdef int smart = len(insert_keys)==0 and len(replace_keys)==0
        cdef int i, t
        cdef char c, k
        cdef str pre_w
        for i, c in enumerate(keyseq):
            if smart:
                if c in spcl_keys: # if replacing a caps or shift, replace with everything
                    replace_keys = ALLOWED_KEYS
                    insert_keys = ALLOWED_KEYS
                else: # else use only the closed by keys or spcl keys
                    replace_keys = np.concatenate(
                        (self._keyboard_nearby_keys(c), spcl_keys)
                    )
                    if i>0:
                        insert_keys = np.unique(np.concatenate(
                            (replace_keys, 
                            self._keyboard_nearby_keys(keyseq[i-1]))
                        ))
            assert all(map(lambda c: (c>=20 and c<128) or c in [3,4], 
                           np.concatenate((replace_keys, insert_keys)))), \
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

        # For inserting at the end
        pre_w, shift, caps = sub_words[-1][0]
        if smart:
            insert_keys = ALLOWED_KEYS
        for k in insert_keys:
            if k not in spcl_keys:
                yield pre_w + chr(self._apply_shift_caps(k, shift, caps))
                # yield pre_w + chr(self._apply_shift_caps(k, True, caps))

    def word_to_typos(self, word, insert_keys='', replace_keys=''):
        """
        Most highlevel and useful function.
        All possible edits around the word.
        Insert all keys, delete all keys, replace only keys that are close.
        """
        keypress = self._word_to_keyseq(unicode(word))
        insert_keys = np.array([ord(c) for c in insert_keys])
        replace_keys = np.array([ord(c) for c in replace_keys])
        return self.keyseq_insert_edits(
            keypress, insert_keys, replace_keys
        )

    def keyseq_to_word(self, keyseq):
        return self._keyseq_to_word(unicode(keyseq))

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
        assert KEYBOARD_TYPE == 'US', "Not implemented for mobile"

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
