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
def bundle_dict_init(train):
    bundle_tmp = train.groupby(['user_id', 'bundle'])[['answered_correctly']].agg(['sum', 'count']).astype(np.uint16)
    bundle_tmp.columns = ['sum', 'count']
    bundle_tmp = bundle_tmp.reset_index().set_index('user_id')
    bundle_dict = bundle_tmp.groupby('user_id').apply(lambda x: np.array(x, dtype=np.uint16)).to_dict()
    bundle_dict = dict_trans(bundle_dict)
    return bundle_dict

@jit(nopython = True, fastmath = False)
def test_bundle_cal_wrap(vals, bundle_dict):
    attempt = np.zeros((vals.shape[0], 2))
    for i in range(vals.shape[0]):
        user_id = vals[i][0]
        bundle = vals[i][1]
        if user_id in bundle_dict:
            user_array = bundle_dict[user_id]
            idx = np.where(user_array[:,0] == bundle)[0]
            if  idx.shape[0] == 0:
                add_array = np.array([[bundle, 0, 0]], dtype=np.uint16)
                bundle_dict[user_id] =  np.concatenate((user_array, add_array))
        else:
            bundle_dict[user_id] = np.array([[bundle, 0, 0]], dtype=np.uint16)
        idx = np.where(bundle_dict[user_id][:,0] == bundle)[0]
        count = bundle_dict[user_id][idx, 1:]
        attempt[i,:] = count
    return attempt

# @timeit
def test_bundle_feature(df, bundle_dict):
    bundle_ds = np.zeros((df.shape[0], 3))
    vals = df[['user_id', 'bundle', 'answered_correctly']].values.astype(np.uint32)
    valid_bundle_ds = test_bundle_cal_wrap(vals, bundle_dict)
    bundle_ds[:,:2] = valid_bundle_ds
    bundle_ds[:, 2] = bundle_ds[:, 0]/(1e-6 + bundle_ds[:, 1])
    return bundle_ds


def update_bundle_dict(bundle_dict, previous_test_df):
    for vals in previous_test_df[['user_id', 'bundle', 'answered_correctly']].values:
        user_id = vals[0]
        bundle = vals[1]
        answer_record = vals[2]
        # if user_id in bundle_dict:
        #     user_array = bundle_dict[user_id]
        #     idx = np.where(user_array[:,0] == bundle)[0]
        #     if  idx.shape[0] == 0:
        #         add_array = np.array([[bundle, 0, 0]], dtype=np.uint16)
        #         bundle_dict[user_id] =  np.concatenate((user_array, add_array))
        # else:
        #     bundle_dict[user_id] = np.array([[bundle, 0, 0]], dtype=np.uint16)
            
        idx = np.where(bundle_dict[user_id][:,0] == bundle)[0]
        count = bundle_dict[user_id][idx, 1:]
        count[0,0] += answer_record
        count[0,1] += np.uint16(1)
        bundle_dict[user_id][idx,1:] = count
    return bundle_dict


def dict_trans(state_dict):
    float_array = types.Array(types.uint16, 2, 'A')
    count_dict = Dict.empty(key_type = types.uint32, value_type = float_array)
    for key, vals in state_dict.items():
        count_dict[key] = np.asarray(vals, dtype=np.uint16)
    return count_dict

