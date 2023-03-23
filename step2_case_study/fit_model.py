import pickle
import numpy as np


if __name__=='__main__':
    outcome = 'Dementia'
    with open(f'dataset_{outcome}.pickle', 'rb') as ff:
        res = pickle.load(ff)
    sids = res['sids']
    X = res['X']; Xnames = res['Xnames']
    S = res['S']
    Y = res['Y']
    #L = res['L']; Lnames = res['Lnames']
    import pdb;pdb.set_trace()