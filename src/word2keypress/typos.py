from __future__ import print_function
import os, sys, re, json
from collections import defaultdict
import numpy as np
import pandas as pd
try:
    from word2keypress.weight_matrix import WEIGHT_MATRIX
    from word2keypress.weighted_edist import (
        STARTSTR, ENDSTR, KB, BLANK, SHIFT_KEY, CAPS_KEY, all_edits, _editdist)
except ImportError:
    from weight_matrix import WEIGHT_MATRIX
    from weighted_edist import (
        STARTSTR, ENDSTR, KB, BLANK, SHIFT_KEY, CAPS_KEY, all_edits, _editdist)




EDIT_DIST_CUTOFF = 1
WEIGHT_MATRIX = [
    (e,w) for e, w in WEIGHT_MATRIX
    if _editdist(e[0], e[1], limit=EDIT_DIST_CUTOFF)[1]
]

# giant_regex = re.complie('|'.join(
#     re.escape(l) for ((l,r),w) in WEIGHT_MATRIX))

def allowed_edits(pw_key_str):
    """
    Returns all the edits that are allowed for @pw.
    An edit=(l -> r) is allowed if @l is in @pw
    returns the filtered WEIGHT_MATRIX
    """
    if not pw_key_str.startswith(STARTSTR):
        pw_key_str = STARTSTR + pw_key_str + ENDSTR
        # print(pw_key_str)
    return sorted(
        [((l,r), w) for ((l,r),w) in WEIGHT_MATRIX
         if l.replace(BLANK, '') in pw_key_str], 
        key=lambda x: x[1], reverse=True
    )

def edit_locs(pw_key_str, l):
    matched_indexes = [
        (m.start(), m.end())
        for m in re.finditer('({})'.format(re.escape(l.replace(BLANK, ''))), 
                             pw_key_str)
        if m.start()<len(pw_key_str) and m.end()>0
    ]
    return matched_indexes


def apply_edit(pw_key_str, e):
    """
    Applies edits on the pw_key_str, whereever the edit e is possible
    If there are multiple location, then apply only at one location.
    """
    l, r = e
    matched_indexes = edit_locs(pw_key_str, l)
    assert matched_indexes, "Wow!! matched index is empty for pw={}, e={}"\
        .format(pw, e)
    # Choose one index at random from the possible options
    # i = np.random.randint(0, len(matched_indexes))
    # pos_s, pos_e = matched_indexes[i]
    # if BLANK in l: 
    #     typo_key_str = _insert_edit(pw_key_str, l, r, pos_s, pos_e)
    # else:
    for m in matched_indexes:
        pos_s, pos_e = m
        typo_key_str = pw_key_str[:pos_s] + r + pw_key_str[pos_e:]
        yield typo_key_str.replace(BLANK, ''), 1.0/len(matched_indexes)

    
def get_prob(rpw, tpw):
    """
    Probability that rpw is mistyped to tpw,
    get all the edit locations. sum their probabiliries
    """
    edits = set(all_edits(rpw, tpw, N=1, edit_cutoff=1))
    pw_key_str = STARTSTR + KB.word_to_keyseq(rpw) + ENDSTR
    E = allowed_edits(pw_key_str)
    s = float(sum(x[1] for x in E))
    if(s==0): return 0.0
    # print("s = {} (len(E)={})".format(s, len(E)))
    # print(edits)
    f = 0.0
    for e, w in E:
        if e not in edits: continue
        for typo_key_str, w_frac in apply_edit(pw_key_str, e):
            typo_key_str = typo_key_str.strip(STARTSTR).strip(ENDSTR)
            typo = KB.keyseq_to_word(typo_key_str)
            if typo == tpw:
                f += w*w_frac
    return f/s

            
def get_topk_typos(pw, k=10):
    """
    Returns top k typos of the word pw
    """
    pw_key_str = STARTSTR + KB.word_to_keyseq(pw) + ENDSTR
    E = sorted(allowed_edits(pw_key_str), key=lambda x: x[1]*len(x[0][0]),
               reverse=True)
    tt = defaultdict(float)
    s = float(sum(x[1] for x in E))
    # print("s = {} (len(E)={})".format(s, len(E)))
    i = 0
    debug_pw = set([pw.swapcase()])
    while len(tt)<k*len(pw)*10 and i <len(E):
        e, w = E[i]
        for typo_key_str, w_frac in apply_edit(pw_key_str, e):
            typo_key_str = typo_key_str.strip(STARTSTR).strip(ENDSTR)
            typo = KB.keyseq_to_word(typo_key_str)
            tt[typo] += w * w_frac/s
            # if typo in debug_pw:
            #     print("{!r} ->> {} ({}, {})".format(typo_key_str, e, w, w*w_frac/s))
        i += 1
    return sorted(tt.items(), key=lambda x: x[1], reverse=True)[:k]


def test_model(train_f):
    from pyxdameraulevenshtein import damerau_levenshtein_distance as dldist
    d = pd.read_csv(train_f, skipinitialspace=False)\
          .astype(str)
    d_ts = d[d.rpw != d.tpw].sample(int(0.03*len(d.index)), random_state=435)
    a = np.array([get_prob(rpw, tpw)
            for rpw, tpw in zip(d_ts.rpw, d_ts.tpw) 
            if dldist(rpw.lower(), tpw.lower())<=1])
    a = a[a>0]
    rank = []
    for rpw, tpw in  zip(d_ts.rpw, d_ts.tpw):
        if dldist(rpw.lower(), tpw.lower())>1: continue
        k = 20
        typos = [tp for tp, w in get_topk_typos(rpw, k)]
        if tpw in typos:
            rank.append(typos.index(tpw)+1)
        else:
            rank.append(k)
    print("Avg_rank: ", sum(rank)/float(len(rank)*k))
    print(d_ts.shape, a.shape, a.mean(), a.std())
    return  a
    
if __name__ == '__main__':
    USAGE = """Usage:
$ {} [options] [arguments]
-allowed-edits <rpw> : returns the allowed edits of rpw
-sample <password>: samples typos for the password from the model
-prob <rpw> <tpw>: probability of rpw -> tpw
-topktypos <rpw> [<n>]  : returns n (default 10) typos of rpw 
-test : Tests the efficacy of the model 
        """.format(sys.argv[0])
    if len(sys.argv)<=1:
        print(USAGE)
        exit(1)
    if sys.argv[1] == '-allowed-edits':
        pw = KB.word_to_keyseq(sys.argv[2])
        (l,r),w  = WEIGHT_MATRIX[0]
        assert l.replace(BLANK, '') in pw, "{!r}, {} {}"\
                .format(l.replace(BLANK, ''), pw, w)
        print(allowed_edits(pw))
    elif sys.argv[1] == '-sample':
        pw = sys.argv[2]
        print("{} --> {}".format(pw, len(set((sample_typos(pw, 100))))))
    elif sys.argv[1] == '-topktypos':
        pw = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv)>3 else 10
        typos = get_topk_typos(pw, n)
        print('\n'.join("{}: {:.5f}".format(x,y) for x, y in typos))
        print("{} --> {}".format(pw, len(typos)))
    elif sys.argv[1] == '-prob':
        print(get_prob(sys.argv[2], sys.argv[3]))
    elif sys.argv[1] == '-test':
        (test_model(sys.argv[2]))
    else:
        print(USAGE)
