#!/usr/bin/env python3

_ws_map = str.maketrans('qQLVzZfFtTdDNwWxXR', 'fFxX~\'NYwWqQRtTdDz')
def convert (src):
    '''
    Converts WX scheme to Sanskrit Library Phonetic Basic notation
    '''
    return src.translate(_ws_map)
