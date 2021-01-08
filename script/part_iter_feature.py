from tqdm import tqdm
from numba import jit,njit
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
from multiprocessing import Process, Manager,Pool
from functools import partial
from numba import prange
import numpy as np
import pandas as pd
from numba import types
from numba.typed import Dict
import functools, time
from numba.typed import List



def timeit(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, np.round(time2-time1, 2)))

        return ret
    return wrap


@jit(nopython = True, parallel = False)
def iter_part_func(vals, count_dict, sum_dict):
    part_ds = np.zeros((vals.shape[0], 3))
    for i in prange(vals.shape[0]):
        idx = np.int8(vals[i, 2]) - 1
        user_ = np.int32(vals[i, 0])
        part_ds[i, 0] = sum_dict[user_][idx]
        part_ds[i, 1] = count_dict[user_][idx]
        count_dict[user_][idx] += 1
        sum_dict[user_][idx] += np.int16(vals[i, 1])
    
    part_ds[:, 2] = part_ds[:, 0]/(1e-6 + part_ds[:, 1])
    return part_ds

@timeit
def user_part_feature(train):
    float_array = types.int16[:]
    count_dict = Dict.empty(key_type = types.int64, value_type = float_array)
    sum_dict = Dict.empty(key_type = types.int64, value_type = float_array)
    for idx in train['user_id'].unique():
        count_dict[idx] = np.zeros(7, dtype=np.int16)
        sum_dict[idx] = np.zeros(7, dtype=np.int16)
    vals = train[['user_id', 'answered_correctly', 'part']].values
    part_ds = iter_part_func(vals, count_dict, sum_dict)
    return part_ds

@timeit
def part_dict_init(train):
    
    @jit(nopython = True)
    def wrap_cal(part_dict, vals):
        for idx in prange(vals.shape[0]):
            val = vals[idx]
            part_dict[np.int64(val[0])][np.int(val[1]) - 1] = val[2:]
        return part_dict
    
    float_array = types.Array(types.uint16, 2, 'A')
    part_dict = Dict.empty(key_type = types.int64, value_type = float_array)
    for idx in train['user_id'].unique():
        part_dict[idx] = np.zeros((7,2), dtype=np.uint16)
    tmp = train.groupby(['user_id', 'part'])['answered_correctly'].agg(['sum', 'count']).reset_index()
    part_dict = wrap_cal(part_dict,tmp.values)
    return part_dict

@jit(nopython = True)
def valid_part_cal(vals, part_dict):
    valid_ds = np.zeros((vals.shape[0], 2), dtype = np.uint16)
    for i in range(vals.shape[0]):
        user_id = vals[i][0]
        part = vals[i][1] - 1
        if user_id in part_dict:
            part_array = part_dict[user_id]
            valid_ds[i,:] = part_array[part]
        else:
            part_dict[user_id] = np.zeros((7,2), dtype=np.uint16)
            part_array = part_dict[user_id]
            valid_ds[i,:] = part_array[part]
        part_array[part, 0] += vals[i][2]
        part_array[part, 1] += 1
        part_dict[user_id] = part_array
    return valid_ds

@timeit
def valid_part_feature(valid, part_dict):
    valid_ds = np.zeros((valid.shape[0], 3))
    valid_ds[:, [0, 1]] = valid_part_cal(valid[['user_id', 'part', 'answered_correctly']].values, part_dict)
    valid_ds[:,2] = valid_ds[:,0]/(1e-6 + valid_ds[:,1])
    return valid_ds