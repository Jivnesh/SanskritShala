#!/usr/bin/env python3

# Vowels and vowel marks
_vow = dict({
    'a' : ['अ', ''],
    'A' : ['आ', 'ा'],
    'i' : ['इ', 'ि'],
    'I' : ['ई', 'ी'],
    'u' : ['उ', 'ु'],
    'U' : ['ऊ', 'ू'],
    'f' : ['ऋ', 'ृ'],
    'F' : ['ॠ', 'ॄ'],
    'x' : ['ऌ', 'ॢ'],
    'X' : ['ॡ', 'ॣ'],
    'e' : ['ए', 'े'],
    'E' : ['ऐ', 'ै'],
    'o' : ['ओ', 'ो'],
    'O' : ['औ', 'ौ']})

# Consonants
_con = dict({
    'k' : 'क',
    'K' : 'ख',
    'g' : 'ग',
    'G' : 'घ',
    'N' : 'ङ',
    'c' : 'च',
    'C' : 'छ',
    'j' : 'ज',
    'J' : 'झ',
    'Y' : 'ञ',
    'w' : 'ट',
    'W' : 'ठ',
    'q' : 'ड',
    'Q' : 'ढ',
    'R' : 'ण',
    't' : 'त',
    'T' : 'थ',
    'd' : 'द',
    'D' : 'ध',
    'n' : 'न',
    'p' : 'प',
    'P' : 'फ',
    'b' : 'ब',
    'B' : 'भ',
    'm' : 'म',
    'y' : 'य',
    'r' : 'र',
    'l' : 'ल',
    'v' : 'व',
    'S' : 'श',
    'z' : 'ष',
    's' : 'स',
    'h' : 'ह'})

# Others
_oth = dict({
    'M' : 'ं',
    'H' : 'ः',
    '~' : 'ँ',
    '\'' : 'ऽ',
    '0' : '०',
    '1' : '१',
    '2' : '२',
    '3' : '३',
    '4' : '४',
    '5' : '५',
    '6' : '६',
    '7' : '७',
    '8' : '८',
    '9' : '९'})

def convert(src):
    '''
    Converts Sanskrit Library Phonetics Basic notation to
    Devanagari characters
    '''
    tgt = ''
    boo = False
    inc = 0
    while inc < len(src):
        now = src[inc]
        nxt = src[inc+1] if inc < len(src) - 1 else ''
        if now in _con:
            tgt += _con[now]
            if nxt == 'a':
                inc += 1
            elif nxt in _vow:
                boo = True
            else:
                tgt += '्'
        elif now in _vow:
            if boo:
                tgt += _vow[now][1]
                boo = False
            else:
                tgt += _vow[now][0]
        elif now in _oth:
            tgt += _oth[now]
        elif now == '.':
            if nxt == '.':
                tgt += '॥'
                inc += 1
            else:
                tgt += '।'
        else:
            tgt += now
        inc += 1
    return tgt
