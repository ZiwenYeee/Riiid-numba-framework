from tqdm import tqdm
from numba import jit,njit
from numba import types
from numba.typed import Dict
import functools, time
from multiprocessing import Process, Manager,Pool
from functools import partial
from numba import prange
import numpy as np
import pandas as pd




from numba import prange
# @jit(nopython = False, fastmath = True, parallel = True)
def test_iter_item_func(vals, item_dict):
    cnt = 0
    feats = np.zeros((vals.shape[0], 2), dtype=np.float64)
    for cnt in range(vals.shape[0]):
        feats[cnt, 0] = item_dict[vals[cnt]]['item_count']
        feats[cnt, 1] = item_dict[vals[cnt]]['item_sum']
    return feats

def test_item(df, item_dict):        
    feats = test_iter_item_func(df['content_id'].values, item_dict)
    feats = pd.DataFrame(feats, columns = ['item_count', 'item_sum'])
    feats['item_mean'] = feats['item_sum']/(1e-7 + feats['item_count'])
    return feats 
