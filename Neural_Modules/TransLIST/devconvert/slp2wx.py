#!/usr/bin/env python3

_sw_map = str.maketrans('fFxX~\'NYwWqQRtTdDz', 'qQLVzZfFtTdDNwWxXR')
def convert (src):
    '''
    Converts Sanskrit Library Phonetic Basic notation to WX scheme
    '''
    return src.translate(_sw_map)
