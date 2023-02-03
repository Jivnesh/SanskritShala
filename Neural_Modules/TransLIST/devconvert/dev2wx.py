#!/usr/bin/env python3

# Consonants
_con = dict({
    'क' : 'k',
    'ख' : 'K',
    'ग' : 'g',
    'घ' : 'G',
    'ङ' : 'f',
    'च' : 'c',
    'छ' : 'C',
    'ज' : 'j',
    'झ' : 'J',
    'ञ' : 'F',
    'ट' : 't',
    'ठ' : 'T',
    'ड' : 'd',
    'ढ' : 'D',
    'ण' : 'N',
    'त' : 'w',
    'थ' : 'W',
    'द' : 'x',
    'ध' : 'X',
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
    'ष' : 'R',
    'स' : 's',
    'ह' : 'h'})

# Vowel marks
_mar = dict({
    'ा' : 'A',
    'ि' : 'i',
    'ी' : 'I',
    'ु' : 'u',
    'ू' : 'U',
    'ृ' : 'q',
    'ॄ' : 'Q',
    'ॢ' : 'L',
    'ॣ' : 'V',
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
    'ऋ' : 'q',
    'ॠ' : 'Q',
    'ऌ' : 'L',
    'ॡ' : 'V',
    'ए' : 'e',
    'ऐ' : 'E',
    'ओ' : 'o',
    'औ' : 'O',
    'ं' : 'M',
    'ः' : 'H',
    'ँ' : 'z',
    '।' : '.',
    '॥' : '..',
    'ऽ' : 'Z',
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
