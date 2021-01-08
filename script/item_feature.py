from tqdm import tqdm
from numba import jit
from numba import types
from numba.typed import Dict
import functools, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
import functools, time
def timeit(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, np.round(time2-time1, 2)))

        return ret
    return wrap

from numba import prange
@jit(nopython = True, fastmath = False, parallel = True)
def iter_item_func(vals, count_dict, sum_dict):
    cnt = 0
    feats = np.zeros((vals.shape[0], 2), dtype=np.float64)
    for cnt in range(vals.shape[0]):
        feats[cnt, 0] = count_dict[vals[cnt]]
        feats[cnt, 1] = sum_dict[vals[cnt]]
#         cnt += 1
    return feats

@timeit
def initial_item(df, item_dict):
    count_dict = Dict.empty(key_type = types.int64, value_type = types.float64)
    sum_dict = Dict.empty(key_type = types.int64, value_type = types.float64)
    for key, val in item_dict.items():
        count_dict[key] = val['item_count']
        sum_dict[key] = val['item_sum']
        
    feats = iter_item_func(df['content_id'].values, count_dict, sum_dict)
    feats = pd.DataFrame(feats, columns = ['item_count', 'item_sum'])
    feats['item_mean'] = feats['item_sum']/(1e-7 + feats['item_count'])
    return feats 
