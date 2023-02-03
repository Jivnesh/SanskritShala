#!/usr/bin/env python3

# SLP to (La)TeX map
_st_dict = dict({
    'A' : '\={a}',
    'I' : '\={\i}',
    'U' : '\={u}',
    'f' : '\d{r}',
    'F' : '\={\d{r}}',
    'x' : '\d{l}',
    'X' : '\={\d{l}}',
    'E' : 'ai',
    'O' : 'au',
    'M' : '\d{m}',
    'H' : '\d{h}',
    '~' : '\.{m}',
    'K' : 'kh',
    'G' : 'gh',
    'N' : '\.{n}',
    'C' : 'ch',
    'J' : 'jh',
    'Y' : '\~{n}',
    'w' : '\d{t}',
    'W' : '\d{t}h',
    'q' : '\d{d}',
    'Q' : '\d{d}h',
    'R' : '\d{n}',
    'T' : 'th',
    'D' : 'dh',
    'P' : 'ph',
    'B' : 'bh',
    'S' : '\\\'{s}',
    'z' : '\d{s}'})
_st_map = str.maketrans(_st_dict)

def convert(src):
    '''
    Converts Sanskrit Library Phonetic Basic notation to
    (La)TeX friendly notation for diacritic IAST
    '''
    return src.translate(_st_map)
