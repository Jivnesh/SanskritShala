
from logging import exception


def get_path(exp_type):
    if exp_type=='saCTI-large coarse':
        train = 'data/sacti/coarse/train.conll'
        dev = 'data/sacti/coarse/dev.conll'
        test = 'data/sacti/coarse/test.conll'
    elif exp_type=='saCTI-large fine':
        train = 'data/sacti/fine/train.conll'
        dev = 'data/sacti/fine/dev.conll'
        test = 'data/sacti/fine/test.conll'
    elif exp_type=='saCTI-base coarse':
        train = 'data/coling/coarse/train.conll'
        dev = 'data/coling/coarse/dev.conll'
        test = 'data/coling/coarse/test.conll'
    elif exp_type=='saCTI-base fine':
        train = 'data/coling/fine/train.conll'
        dev = 'data/coling/fine/dev.conll'
        test = 'data/coling/fine/test.conll'
    elif exp_type=='marathi':
        train = 'data/marathi/coarse/train.conll'
        dev = 'data/marathi/coarse/dev.conll'
        test = 'data/marathi/coarse/test.conll'
    elif exp_type=='english':
        train = 'data/english/coarse/train.conll'
        dev = 'data/english/coarse/dev.conll'
        test = 'data/english/coarse/test.conll'
    else:
        raise Exception("Please select a proper experimnet from the list")

    return train,dev,test

choices = ['saCTI-large coarse','saCTI-large fine','saCTI-base coarse','saCTI-base fine','marathi','english']