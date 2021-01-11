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

def timeit(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, np.round(time2-time1, 2)))

        return ret
    return wrap


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
def test_part_cal(vals, part_dict):
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
        part_dict[user_id] = part_array
    return valid_ds

# @timeit
def test_part_feature(valid, part_dict):
    valid_ds = np.zeros((valid.shape[0], 3))
    vals = valid[['user_id', 'part', 'answered_correctly']].values.astype(np.uint32)
    valid_ds[:, [0, 1]] = test_part_cal(vals, part_dict)
    valid_ds[:,2] = valid_ds[:,0]/(1e-6 + valid_ds[:,1])
    return valid_ds

def update_part_dict(part_dict, previous_test_df):
    for vals in previous_test_df[['user_id', 'part', 'answered_correctly']].values:
        user_id = vals[0]
        part = vals[1] - 1
        answer_record = vals[2]
        part_dict[user_id][part, 0] += answer_record
        part_dict[user_id][part, 1] += 1
        
        # if user_id in part_dict:
        #     part_dict[user_id][part, 0] += answer_record
        #     part_dict[user_id][part, 1] += 1
        # else:
        #     part_dict[user_id] = np.zeros((7,2), dtype=np.uint16)
        #     part_dict[user_id][part, 0] += answer_record
        #     part_dict[user_id][part, 1] += 1            
    return part_dict

def dict_trans(state_dict):
    float_array = types.Array(types.uint16, 2, 'A')
    count_dict = Dict.empty(key_type = types.uint32, value_type = float_array)
    for key, vals in state_dict.items():
        count_dict[key] = np.asarray(vals, dtype=np.uint16)
    return count_dict
