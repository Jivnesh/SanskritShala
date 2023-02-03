#!/usr/bin/env python3

# SLP to IAST map
_si_dict = dict({
    'A' : 'ā',
    'I' : 'ī',
    'U' : 'ū',
    'f' : 'ṛ',
    'F' : 'ṝ',
    'x' : 'ḷ',
    'X' : 'ḹ',
    'E' : 'ai',
    'O' : 'au',
    'M' : 'ṃ',
    'H' : 'ḥ',
    '~' : 'ṁ',
    'K' : 'kh',
    'G' : 'gh',
    'N' : 'ṅ',
    'C' : 'ch',
    'J' : 'jh',
    'Y' : 'ñ',
    'w' : 'ṭ',
    'W' : 'ṭh',
    'q' : 'ḍ',
    'Q' : 'ḍh',
    'R' : 'ṇ',
    'T' : 'th',
    'D' : 'dh',
    'P' : 'ph',
    'B' : 'bh',
    'S' : 'ś',
    'z' : 'ṣ'})
_si_map = str.maketrans(_si_dict)

def convert(src):
    '''
    Converts Sanskrit Library Phonetic Basic notation to
    International Alphabet for Sanskrit Transliteration scheme
    '''
    return src.translate(_si_map)
