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

@njit
def shift(arr, num, fill_value=np.nan):
    if num >= 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))
    
@timeit
def feat5_group(train):
    a = train[['user_id', 'timestamp', 'content_id', 'item_mean', 'prior_question_elapsed_time']].values
    ind = np.lexsort((a[:,2],a[:,1],a[:,0]))
    a = a[ind]
    g = np.split(a, np.unique(a[:, 0], return_index=True)[1][1:])
    return g, ind

@jit(nopython = True)
def step_shift(arr, step):
    new_arr = np.zeros(arr.shape)
    new_arr[0] = np.nan
    beg = 0
    for i in range(step.shape[0] - 1):
        beg = beg + step[i]
        new_arr[beg:beg + step[i+1]] = arr[beg - step[i]]
    return new_arr

@jit(nopython = True, fastmath = True)
def step_cal(arr, step):
    new_arr = np.zeros(arr.shape[0])
    beg = 0
    for i in range(step.shape[0] - 1):
        beg = beg + step[i]
        new_arr[beg:beg + step[i+1]] = step[i]
    return new_arr


@jit(nopython = True, fastmath = False)
def feat5_step_agg(arr, step):
    m = 4
    ret = np.zeros((arr.shape[0], m))
    beg = 0
    for i in step:
        tmp = arr[beg:beg+i]
        ret[beg:beg + i, 0] = np.nanmean(tmp)
        ret[beg:beg + i, 1] = np.nanstd(tmp)
        ret[beg:beg + i, 2] = np.nanmin(tmp)
        ret[beg:beg + i, 3] = np.nanmax(tmp)
        beg += i
    return ret

# @timeit
def feat5_cal(tmp_g):
    step = np.unique(tmp_g[:, 1], return_counts=True)[1]
    if tmp_g.shape[0] > 1:
        col1 = tmp_g[:,1] - shift(tmp_g[:,1], 1)
    else:
        col1 = np.full((tmp_g.shape[0], ), np.nan)
        
    if tmp_g.shape[0] > 2:
        col2 = tmp_g[:,1] - shift(tmp_g[:,1], 2)
    else:
        col2 = np.full((tmp_g.shape[0], ), np.nan)
        
    col3 = tmp_g[:,1] - step_shift(tmp_g[:, 1], step)
    time_col = np.array([col1, col2, col3]).T
    
    col_agg1 = feat5_step_agg(col1, step)
    col_agg2 = feat5_step_agg(tmp_g[:,3], step)
    
    item_col1 = shift(tmp_g[:,2], 1)
    item_col2 = shift(tmp_g[:,3], 1)
    item_col3 = tmp_g[:,3] - shift(tmp_g[:,3], 1)
    item_col = np.array([item_col1, item_col2, item_col3]).T
    new_col1 = tmp_g[:,1] - step_shift(step_shift(tmp_g[:,1],step),step)
    new_col2 = step_shift(col3, step)
    new_col3 = col3 - step_shift(col3, step)
    new_col4 = step_shift(new_col3, step)
    new_col = np.array([new_col1, new_col2, new_col3, new_col4]).T
    feat5_ds = np.concatenate((time_col, item_col, col_agg1, col_agg2, new_col), axis = 1)
    return feat5_ds

def feat5(group):
    res = []
    for i in tqdm(range(len(group))):
        tmp_g = group[i]
        tmp_feat = feat5_cal(tmp_g)
        res.append(tmp_feat)
    ans = np.concatenate(res)
    return ans

def roll_init(gp, record = 30):
    dic = {}
    for tmp in gp:
        if tmp.shape[0] > record:
            dic[tmp[0][0]] = tmp[-record:]
        else:
            dic[tmp[0][0]] = tmp
    return dic

def valid_feats5(valid_gp, rolling_gp):
    res = []
    for i in range(len(valid_gp)):
        tmp_g = valid_gp[i]
        valid_shape = tmp_g.shape[0]
        if tmp_g[0,0] in rolling_gp:
            tmp_rolling = rolling_gp[tmp_g[0, 0]]
            tmp_g = np.concatenate([tmp_rolling, tmp_g])
        
        tmp_res = feat5_cal(tmp_g)
        tmp_res = tmp_res[-valid_shape:]
        res.append(tmp_res)
    ans = np.concatenate(res)
    return ans


def feats5_wrap(train):
    group5, g5_idx = feat5_group(train)
    ds5 = feat5(group5)
    ds5 = resort_array(ds5, g5_idx)
    
    col_name = []
    col_name += ['user_d1', 'user_d2', 'task_set_distance']
    col_name += ['content_shift1', 'item_mean_shift1', 'item_mean_diff1']
    col_name += ['user_diff_mean', 'user_diff_std', 'user_diff_min', 'user_diff_max']
    col_name += ['task_set_item_mean', 'task_set_item_std', 'task_set_item_min', 'task_set_item_max']
    col_name += ['task_set_distance2', 'task_distance_shift', 'task_set_distance_diff', 'task_distance_diff_shift']

    return group5, ds5, col_name

def valid_feats5_wrap(valid, rolling_gp):
    valid_gp, valid_idx = feat5_group(valid)
    valid_ds5 = valid_feats5(valid_gp, rolling_gp)
    valid_ds5 = resort_array(valid_ds5, valid_idx)
    
    return valid_ds5

@jit(nopython = True, parallel = True, fastmath = False)
def resort_array(test_ds, idx):
    new_test = np.zeros(test_ds.shape)
    for i in range(idx.shape[0]):
        new_test[idx[i]] = test_ds[i]
    return new_test