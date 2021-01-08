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

import functools, time


def timeit(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, np.round(time2-time1, 2)))

        return ret
    return wrap

@jit(nopython = True, fastmath = False)
def iter_user_func(vals, count_dict, sum_dict):
    cnt = 0
    feats = np.zeros((vals.shape[0], 2), dtype=np.int64)
    for row in vals:
        feats[cnt, 0] = count_dict[row[0]]
        feats[cnt, 1] = sum_dict[row[0]]
        count_dict[row[0]] += 1
        sum_dict[row[0]] += row[1]
        cnt += 1
    return feats

@timeit
def initial_user(df):
    count_dict = Dict.empty(key_type = types.int64, value_type = types.int64)
    sum_dict = Dict.empty(key_type = types.int64, value_type = types.int64)
    for user_ in df['user_id'].unique():
        count_dict[user_] = 0
        sum_dict[user_] = 0
    feats = iter_user_func(df[['user_id','answered_correctly']].values, count_dict, sum_dict)
    feats = pd.DataFrame(feats, columns = ['user_count', 'user_sum'])
    feats['user_mean'] = feats['user_sum']/(1e-7 + feats['user_count'])
    return feats


@njit
def shift(arr, num, fill_value=np.nan):
    if num >= 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))
    

@jit(nopython = True, fastmath = False)
def moving_sum(a, n=3) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret

@timeit
def valid_user(df, user_dict):
    count_dict = Dict.empty(key_type = types.int64, value_type = types.int64)
    sum_dict = Dict.empty(key_type = types.int64, value_type = types.float64)
    
    for user_ in df['user_id'].unique():
        count_dict[user_] = 0
        sum_dict[user_] = 0
        
    for user_, vals in user_dict.items():
        count_dict[user_] = vals['user_count']
        sum_dict[user_] = vals['user_sum']
    
        
    feats = iter_user_func(df[['user_id','answered_correctly']].values, count_dict, sum_dict)
    feats = pd.DataFrame(feats, columns = ['user_count', 'user_sum'])
    feats['user_mean'] = feats['user_sum']/(1e-7 + feats['user_count'])
    return feats