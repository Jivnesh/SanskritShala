#!/usr/bin/env python3

# Consonants
_con = dict({
    'क' : 'k',
    'ख' : 'K',
    'ग' : 'g',
    'घ' : 'G',
    'ङ' : 'N',
    'च' : 'c',
    'छ' : 'C',
    'ज' : 'j',
    'झ' : 'J',
    'ञ' : 'Y',
    'ट' : 'w',
    'ठ' : 'W',
    'ड' : 'q',
    'ढ' : 'Q',
    'ण' : 'R',
    'त' : 't',
    'थ' : 'T',
    'द' : 'd',
    'ध' : 'D',
    'न' : 'n',
    'प' : 'p',
    'फ' : 'P',
    'ब' : 'b',
    'भ' : 'B',
    'म' : 'm',
    'य' : 'y',
    'र' : 'r',
    'ल' : 'l',
    'व' : 'v',
    'श' : 'S',
    'ष' : 'z',
    'स' : 's',
    'ह' : 'h'})

# Vowel marks
_mar = dict({
    'ा' : 'A',
    'ि' : 'i',
    'ी' : 'I',
    'ु' : 'u',
    'ू' : 'U',
    'ृ' : 'f',
    'ॄ' : 'F',
    'ॢ' : 'x',
    'ॣ' : 'X',
    'े' : 'e',
    'ै' : 'E',
    'ो' : 'o',
    'ौ' : 'O'})

# Others
_oth = dict({
    'अ' : 'a',
    'आ' : 'A',
    'इ' : 'i',
    'ई' : 'I',
    'उ' : 'u',
    'ऊ' : 'U',
    'ऋ' : 'f',
    'ॠ' : 'F',
    'ऌ' : 'x',
    'ॡ' : 'X',
    'ए' : 'e',
    'ऐ' : 'E',
    'ओ' : 'o',
    'औ' : 'O',
    'ं' : 'M',
    'ः' : 'H',
    'ँ' : '~',
    '।' : '.',
    '॥' : '..',
    'ऽ' : '\'',
    '०' : '0',
    '१' : '1',
    '२' : '2',
    '३' : '3',
    '४' : '4',
    '५' : '5',
    '६' : '6',
    '७' : '7',
    '८' : '8',
    '९' : '9'})

def convert(src):
    '''
    Converts Devanagari characters into
    Sanskrit Library Phonetic Basic notation
    '''
    tgt = ''
    inc = 0
    while inc < len(src):
        now = src[inc]
        nxt = src[inc+1] if inc < len(src) - 1 else None
        if now in _con:
            tgt += _con[now]
            if nxt == '्':
                inc += 1
            elif nxt not in _mar:
                tgt += 'a'
        elif now in _mar:
            tgt += _mar[now]
        elif now in _oth:
            tgt += _oth[now]
        else:
            tgt += now
        inc += 1
    return tgt
