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


def rolling_feat_group(train, col_used):
    a = train[col_used].values
    ind = np.lexsort((a[:,2],a[:,1],a[:,0]))
    a = a[ind]
    g = np.split(a, np.unique(a[:, 0], return_index=True)[1][1:])
    return g, ind, col_used

@jit(nopython = True, fastmath = True)
def rolling_cal(arr, step, window = 5, shift_ = 1):
    m = 2
    arr_ = np.concatenate((np.full((window, ), np.nan), arr))
    ret = np.zeros((arr.shape[0], m))
    beg = window
    for i in step: 
        tmp = arr_[beg-window:beg]
        ret[beg - window:(beg - window + i), 0] = np.nanmean(tmp)
        ret[beg - window:(beg - window + i), 1] = np.nansum(tmp)
        beg += i
    return ret


@jit(nopython = True, fastmath = True)
def rolling_time_cal(arr, window = 5, shift_ = 1):
    m = 1
    arr_ = np.concatenate((np.full((window, ), np.nan), arr))
    ret = np.zeros((arr.shape[0], m))
    for i in range(0,arr.shape[0], 1): 
        tmp = arr_[i:i+window+1]
        ret[i, 0] = np.nanmean(tmp)
    return ret

def rolling_cal_wrap(tmp_g, shift_period):
    m = 2
    tmp_res = []
    step = np.unique(tmp_g[:, 1], return_counts=True)[1]
    for window_size in shift_period:
        tmp = rolling_cal(tmp_g[:, 2], step, window_size)
        tmp_res.append(tmp)
    tmp_res = np.concatenate(tmp_res, axis = 1)
    return tmp_res

def rolling_time_cal_wrap(tmp_g, shift_period):
    m = 2
    tmp_res = []
    for window_size in shift_period:
        tmp = rolling_time_cal(tmp_g[:, 2], window_size)
        tmp_res.append(tmp)
    tmp_res = np.concatenate(tmp_res, axis = 1)
    return tmp_res


def rolling_feat_cal(tmp_g, name_dict, global_period):
    answer_idx = name_dict.index('answered_correctly')
    prior_idx = name_dict.index('prior_question_elapsed_time')
    item_mean_idx = name_dict.index('item_mean')
    task_set_idx = name_dict.index('task_set_distance')
    tmp_res1 = rolling_cal_wrap(tmp_g[:,[0,1, answer_idx]], global_period)
    tmp_res2 = rolling_time_cal_wrap(tmp_g[:,[0,1, prior_idx]], global_period)
    tmp_res3 = rolling_time_cal_wrap(tmp_g[:,[0,1, item_mean_idx]], global_period)
    tmp_res4 = rolling_time_cal_wrap(tmp_g[:,[0,1, task_set_idx]], global_period)
    tmp_res = np.concatenate([tmp_res1, tmp_res2, tmp_res3, tmp_res4], axis = 1)
    return tmp_res

# @timeit
def rolling_feature(g, name_dict, global_period):
    res = []
    for i in tqdm(range(len(g))):
        tmp_g = g[i]
        tmp_res = rolling_feat_cal(tmp_g, name_dict, global_period)
        res.append(tmp_res)
    ans = np.concatenate(res)
    return ans


from multiprocessing import Process, Manager,Pool
from functools import partial
from joblib import Parallel, delayed


def parallel_wrap(group, name_dict, shift_period_1, method = 'joblib'):
    if method == 'joblib':
        res = Parallel(n_jobs = 12, backend = 'loky')\
              (delayed(rolling_feat_cal)(group[i], name_dict, shift_period_1)
              for i in tqdm(range(len(group)))
              )
    else:
        manager = Manager()
        Gp = manager.list(group)
        p = Pool(processes = 12)
        res = p.map(partial(rolling_feat_cal, name_dict, shift_period_1 = shift_period_1), group)
        p.close()
    ans = np.concatenate(res)
    return ans


def valid_rolling_feature(valid_gp, rolling_gp, name_dict, global_period):
    res = []
    for i in range(len(valid_gp)):
        tmp_g = valid_gp[i]
        valid_shape = tmp_g.shape[0]
        if tmp_g[0,0] in rolling_gp:
            tmp_rolling = rolling_gp[tmp_g[0, 0]]
            if tmp_rolling.shape[0] > 50:
                tmp_rolling = tmp_rolling[-50:,:]
            tmp_g = np.concatenate([tmp_rolling, tmp_g])
        
        tmp_res = rolling_feat_cal(tmp_g, name_dict, global_period)
        tmp_res = tmp_res[-valid_shape:]
        res.append(tmp_res)
    ans = np.concatenate(res)
    return ans


def test_rolling_feature_wrapper(current_test, group, roll_keep):
    shift_period_1 = [1, 5, 10, 20, 30, 40]
    valid_gp, valid_idx, name_dict = rolling_feat_group(current_test, roll_keep)
    valid_roll_ds = valid_rolling_feature(valid_gp, group, name_dict, shift_period_1)
    valid_roll_ds = resort_array(valid_roll_ds, valid_idx)
    
    rolling_name = []
    func_list = ['mean']
    rolling_name += [f'container_{func}_{p}' for p in shift_period_1 for func in ['mean', 'std']]
    rolling_name += [f'prior_question_elapsed_time_{func}_{p}' for p in shift_period_1 for func in func_list]
    rolling_name += [f'item_mean_{func}_{p}' for p in shift_period_1 for func in func_list]
    rolling_name += [f'task_set_distance_{func}_{p}' for p in shift_period_1 for func in func_list]
    return valid_roll_ds,rolling_name


def roll_init(gp, record = 30):
    dic = {}
    for tmp in gp:
        if tmp.shape[0] > record:
            dic[tmp[0][0]] = tmp[-record:]
        else:
            dic[tmp[0][0]] = tmp
    return dic


def resort_array(test_ds, idx):
    new_test = np.zeros(test_ds.shape)
    for i in range(idx.shape[0]):
        new_test[idx[i]] = test_ds[i]
    return new_test
