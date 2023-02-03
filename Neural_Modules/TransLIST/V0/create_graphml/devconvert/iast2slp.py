#!/usr/bin/env python3

# Double characters mapping to single characters
_dbl = dict({
    'ai' : 'E',
    'au' : 'O',
    'kh' : 'K',
    'gh' : 'G',
    'ch' : 'C',
    'jh' : 'J',
    'ṭh' : 'W',
    'ḍh' : 'Q',
    'th' : 'T',
    'dh' : 'D',
    'ph' : 'P',
    'bh' : 'B'})

# One to one mapping
_oth = dict({
    'ā' : 'A',
    'ī' : 'I',
    'ū' : 'U',
    'ṛ' : 'f',
    'ṝ' : 'F',
    'ḷ' : 'x',
    'ḹ' : 'X',
    'ṃ' : 'M',
    'ḥ' : 'H',
    'ṁ' : '~',
    'ṅ' : 'N',
    'ñ' : 'Y',
    'ṭ' : 'w',
    'ḍ' : 'q',
    'ṇ' : 'R',
    'ś' : 'S',
    'ṣ' : 'z'})

def convert(src):
    '''
    Converts International Alphabet for Sanskrit Transliteration (IAST) scheme to
    Sanskrit Library Phonetic Basic notation
    '''
    tgt = ''
    inc = 0
    while inc < len(src):
        now = src[inc]
        nxt = src[inc+1] if inc < len(src) - 1 else ''
        if now + nxt in _dbl:
            tgt += _dbl[now + nxt]
            inc += 1
        elif now in _oth:
            tgt += _oth[now]
        else:
            tgt += now
        inc += 1
    return tgt
