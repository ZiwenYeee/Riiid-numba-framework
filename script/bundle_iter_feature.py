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

def timeit(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, np.round(time2-time1, 2)))

        return ret
    return wrap

@timeit
def pre_group(train):
    a = train[['user_id', 'timestamp', 'bundle', 'answered_correctly']].values
    ind = np.lexsort((a[:,1],a[:,0]))
    a = a[ind]
    g = np.split(a, np.unique(a[:, 0], return_index=True)[1][1:])
    return g, ind

@jit(nopython = True, parallel = True, fastmath = True)
def resort_array(test_ds, idx):
    new_test = np.zeros(test_ds.shape)
    for i in range(idx.shape[0]):
        new_test[idx[i]] = test_ds[i]
    return new_test

@jit(nopython = True, fastmath = True)
def cum_feat_cal(tmp_g):
    sum_dict = Dict.empty(key_type = types.uint16, value_type = types.uint16)
    count_dict = Dict.empty(key_type = types.uint16, value_type = types.uint16)
    ds = np.zeros((tmp_g.shape[0], 2))
    for i in range(tmp_g.shape[0]):
        bundle = tmp_g[i][2]
        answer = tmp_g[i][3]
        if bundle in sum_dict:
            ds[i, 0] = sum_dict[bundle]
            ds[i, 1] = count_dict[bundle]
            sum_dict[bundle] += answer
            count_dict[bundle] += 1
        else:
            ds[i, 0] = 0
            ds[i, 1] = 0
            
            sum_dict[bundle] = answer
            count_dict[bundle] = 1
    return ds

@timeit
def cum_feat(Group, G_idx):
    res = []
    for tmp_g in Group:
        tmp_ds = cum_feat_cal(tmp_g)
        res.append(tmp_ds)
    ans = np.concatenate(res)
    ans = resort_array(ans, G_idx)
    return ans

def dict_trans(state_dict):
    float_array = types.Array(types.uint16, 2, 'A')
    count_dict = Dict.empty(key_type = types.uint32, value_type = float_array)
    for key, vals in state_dict.items():
        count_dict[key] = np.asarray(vals, dtype=np.uint16)
    return count_dict

@timeit
def bundle_feature(train):
    bundle_ds = np.zeros((train.shape[0], 3))
    Group, bundle_idx = pre_group(train)
    bundle_data = cum_feat(Group, bundle_idx)
    bundle_ds[:,:2] = bundle_data
    bundle_ds[:, 2] = bundle_ds[:, 0]/(1e-6 + bundle_ds[:, 1])
    bundle_tmp = train.groupby(['user_id', 'bundle'])[['answered_correctly']].agg(['sum', 'count']).astype(np.uint16)
    bundle_tmp.columns = ['sum', 'count']
    bundle_tmp = bundle_tmp.reset_index().set_index('user_id')
    bundle_dict = bundle_tmp.groupby('user_id').apply(lambda x: np.array(x, dtype=np.uint16)).to_dict()
    bundle_dict = dict_trans(bundle_dict)
    return bundle_ds, bundle_dict

@jit(nopython = True, fastmath = False)
def bundle_cal_wrap(vals, bundle_dict):
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
        count[0,0] += np.uint16(vals[i][2])
        count[0,1] += np.uint16(1)
        bundle_dict[user_id][idx, 1:] = count
    return attempt

@timeit
def bundle_valid_feature(df, bundle_dict):
    bundle_ds = np.zeros((df.shape[0], 3))
    vals = df[['user_id', 'bundle', 'answered_correctly']].values
    valid_bundle_ds = bundle_cal_wrap(vals, bundle_dict)
    bundle_ds[:,:2] = valid_bundle_ds
    bundle_ds[:, 2] = bundle_ds[:, 0]/(1e-6 + bundle_ds[:, 1])
    return bundle_ds
