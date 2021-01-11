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

from multiprocessing import Process, Manager,Pool
from functools import partial
from joblib import Parallel, delayed

def timeit(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, np.round(time2-time1, 2)))

        return ret
    return wrap

# @timeit
def rolling_feat_group(train, col_used):
    a = train[col_used].values
    ind = np.lexsort((a[:,2],a[:,1],a[:,0]))
    a = a[ind]
    g = np.split(a, np.unique(a[:, 0], return_index=True)[1][1:])
    return g, ind, col_used

def resort_array(test_ds, idx):
    new_test = np.zeros(test_ds.shape)
    for i in range(idx.shape[0]):
        new_test[idx[i]] = test_ds[i]
    return new_test

@jit(nopython = True)
def divide_agg(tmp_g, step, name_dict, beg = 0):
    time_idx = name_dict.index('timestamp')
    answer_idx = name_dict.index('answered_correctly')
    item_mean_idx = name_dict.index('item_mean')
    distance_idx = name_dict.index('task_set_distance')
    content_idx = name_dict.index('content_id')
    prior_idx = name_dict.index('prior_question_elapsed_time')
    bundle_idx = name_dict.index('bundle_id')
    
    
    m = 2
    ret1 = np.zeros((tmp_g.shape[0], m))
    ret2 = np.zeros((tmp_g.shape[0], m))
    ret3 = np.zeros((tmp_g.shape[0], m))
    ret4 = np.zeros((tmp_g.shape[0], m))
    ret5 = np.zeros((tmp_g.shape[0], 2))
    ret6 = np.zeros((tmp_g.shape[0], m))
    ret7 = np.zeros((tmp_g.shape[0], m))
    ret8 = np.zeros((tmp_g.shape[0], 4))
    
    ret9 = np.zeros((tmp_g.shape[0], 2))
    ret10 = np.zeros((tmp_g.shape[0], 5))
    
    ret11 = np.zeros((tmp_g.shape[0], 4))
    
    bundle_ret1 = np.zeros((tmp_g.shape[0], m))
    bundle_ret2 = np.zeros((tmp_g.shape[0], m))
    bundle_ret3 = np.zeros((tmp_g.shape[0], m))
    
    bundle_ret4 = np.zeros((tmp_g.shape[0], 3))
    
    for i in step:
        tmp = tmp_g[:beg]
        idx1 = np.where(tmp[:,  answer_idx] == 0)[0]
        if idx1.shape[0] > 0:
#             wrong_time_diff = tmp_g[:beg, time_idx] - tmp[idx1, time_idx].reshape(-1, 1)
            wrong_time_diff = tmp[idx1, time_idx] - np.concatenate((np.full(1, np.nan), tmp[idx1, time_idx][:-1]))
            
            ret1[beg:beg+i, 0] = np.nanmean(tmp[idx1, item_mean_idx])
            ret1[beg:beg+i, 1] = np.nanmedian(tmp[idx1, item_mean_idx])
        
            ret2[beg:beg+i, 0] = np.nanmean(tmp[idx1, distance_idx])
            ret2[beg:beg+i, 1] = np.nanmedian(tmp[idx1, distance_idx])
            
            ret5[beg:beg+i, 0] = tmp_g[beg][time_idx] - tmp[idx1][-1, time_idx]
            ret6[beg:beg+i, 0] = np.nanmean(wrong_time_diff)
            ret6[beg:beg+i, 1] = np.nanmedian(wrong_time_diff)
            
#             ret11[beg:beg+i, 0] = ret5[beg, 0]/(1e-6 + ret7[beg, 0])
#             ret11[beg:beg+i, 1] = np.ptp(tmp[idx1, item_mean_idx])
            
            
            hard_item = np.where(tmp[idx1, item_mean_idx] < 0.6)[0]
            if hard_item.shape[0] > 0:
                hard_time = tmp[idx1,:][hard_item, time_idx]
#                 hard_time = hard_time - np.concatenate((np.full(1, np.nan), hard_time[:-1]))
                ret9[beg:beg+i, 0] = tmp_g[beg, time_idx] - hard_time[-1]
            
        idx2 = np.where(tmp[:, answer_idx] == 1)[0]
        if idx2.shape[0] > 0:
#             right_time_diff = tmp_g[:beg, time_idx] - tmp[idx2, time_idx].reshape(-1, 1)
            right_time_diff = tmp[idx2, time_idx] - np.concatenate((np.full(1, np.nan), tmp[idx2, time_idx][:-1]))
            ret3[beg:beg+i, 0] = np.nanmean(tmp[idx2, item_mean_idx])
            ret3[beg:beg+i, 1] = np.nanmedian(tmp[idx2, item_mean_idx])
#             ret3[beg:beg+i, 2] = np.nanstd(tmp[idx2, item_mean_idx])
            
            ret4[beg:beg+i, 0] = np.nanmean(tmp[idx2, distance_idx])
            ret4[beg:beg+i, 1] = np.nanmedian(tmp[idx2, distance_idx])
#             ret4[beg:beg+i, 2] = np.nanstd(tmp[idx2, distance_idx])
            
            ret5[beg:beg+i, 1] = tmp_g[beg][time_idx] - tmp[idx2][-1, time_idx]
        
            ret7[beg:beg+i, 0] = np.nanmean(right_time_diff)
            ret7[beg:beg+i, 1] = np.nanmedian(right_time_diff)
            
#             ret11[beg:beg+i, 2] = ret5[beg, 1]/(1e-6 + ret7[beg, 1])
#             ret11[beg:beg+i, 3] = np.ptp(tmp[idx2, item_mean_idx])
            
            hard_item = np.where(tmp[idx2, item_mean_idx] < 0.6)[0]
            if hard_item.shape[0] > 0:
                hard_time = tmp[idx2, :][hard_item, time_idx]
                ret9[beg:beg+i, 1] = tmp_g[beg, time_idx] - hard_time[-1]
               
        for j in range(i):
            last_idx = np.where(tmp[:, content_idx] == tmp_g[beg+j, content_idx])[0]
            if last_idx.shape[0] > 0:
                last_content_time = tmp[last_idx[-1], time_idx]
                ret8[beg+j, 1] = np.nanmean(tmp[last_idx, answer_idx])
                ret8[beg+j, 2] = np.nansum(tmp[last_idx, answer_idx])
                ret8[beg+j, 3] = tmp[last_idx, answer_idx].shape[0]
            else:
                last_content_time = np.nan
            ret8[beg+j, 0] = tmp_g[beg+j, time_idx] - last_content_time

            
        for shift in range(1, 6):
            if tmp.shape[0] > shift:
                ret10[beg:beg+i, shift - 1] = tmp[-shift, distance_idx]
        
        selected_bundle = tmp_g[beg, bundle_idx]
        group_idx = np.where(tmp[:, bundle_idx] == selected_bundle)[0]
        if group_idx.shape[0] > 0:
            part_diff_time = tmp[group_idx, time_idx] - np.concatenate((np.full(1, np.nan), tmp[group_idx, time_idx][:-1]))
            bundle_ret1[beg:beg+i, 0] = np.nanmean(tmp[group_idx, item_mean_idx])
            bundle_ret1[beg:beg+i, 1] = np.nanmedian(tmp[group_idx,  item_mean_idx])
            bundle_ret2[beg:beg+i, 0] = np.nanmean(tmp[group_idx, distance_idx])
            bundle_ret2[beg:beg+i, 1] = np.nanmedian(tmp[group_idx, distance_idx])
            bundle_ret3[beg:beg+i, 0] = np.nanmean(part_diff_time)
            bundle_ret3[beg:beg+i, 1] = np.nanmedian(part_diff_time)
        
            bundle_ret4[beg:beg+i, 0] = np.nansum(tmp[group_idx, answer_idx])
            bundle_ret4[beg:beg+i, 1] = tmp[group_idx, answer_idx].shape[0]
            bundle_ret4[beg:beg+i, 2] = np.nanmean(tmp[group_idx, answer_idx])
        
        beg += i
    
    bundle_ret = np.concatenate((bundle_ret1, bundle_ret2, bundle_ret3, bundle_ret4), axis = 1)
    ret = np.concatenate((ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8, ret9, ret10, bundle_ret), axis = 1)
    return ret

@jit(nopython = True)
def last_group_cal(tmp_g, step, name_dict, beg = 0):
    part_group = np.zeros((tmp_g.shape[0], 9))
    part_idx = name_dict.index('part')
    time_idx = name_dict.index('timestamp')
    answer_idx = name_dict.index('answered_correctly')
    item_mean_idx = name_dict.index('item_mean')
    for i in step:
        tmp = tmp_g[:beg]
        selected_part = tmp_g[beg, part_idx]
        idx_list = []
        data = np.where(tmp[:, part_idx] == selected_part)[0]
        idx = np.where(np.diff(data) != 1)[0] + 1
        if idx.shape[0] > 0:
            for cnt in range(idx.shape[0] + 1):
                if cnt == 0:
                    idx_list.append(data[np.arange(idx[cnt])])
                elif cnt == idx.shape[0]:
                    idx_list.append(data[idx[cnt - 1]:])
                else:
                    idx_list.append(data[np.arange(idx[cnt - 1], idx[cnt])])
        else:
            if data.shape[0] > 0:
                idx_list.append(data)
                
        if len(idx_list) > 0:
            last_group = idx_list[-1]
        else:
            last_group = np.full((0, ), np.nan, dtype=np.int64)
            
        if last_group.shape[0] > 0:
            time_begin = tmp[last_group[0], time_idx]
            time_end = tmp[last_group[-1], time_idx]
            answer_ratio = np.mean(tmp[last_group, answer_idx])
            answer_sum = np.sum(tmp[last_group, answer_idx])
            answer_count = tmp[last_group, answer_idx].shape[0]
            part_time_diff = tmp[last_group, time_idx] - np.concatenate((np.full(1, np.nan), tmp[last_group, time_idx][:-1]))
            item_mean_group = tmp[last_group, item_mean_idx]
            
        else:
            time_begin = np.nan
            time_end = np.nan
            answer_ratio = np.nan
            answer_sum = np.nan
            answer_count = np.nan
            part_time_diff = np.full((1, ), np.nan)
            item_mean_group = np.full((1, ), np.nan)
            
        begin_diff = tmp_g[beg, time_idx] - time_begin
        end_diff = tmp_g[beg, time_idx] - time_end
        time_last = time_end - time_begin
        time_diff_mean = np.nanmean(part_time_diff)
    
        part_group[beg:beg + i, 0] = begin_diff
        part_group[beg:beg + i, 1] = end_diff
        part_group[beg:beg + i, 2] = time_last
        part_group[beg:beg + i, 3] = time_diff_mean
        part_group[beg:beg + i, 4] = answer_ratio
        part_group[beg:beg + i, 5] = answer_sum
        part_group[beg:beg + i, 6] = answer_count
        part_group[beg:beg + i, 7] = np.mean(item_mean_group)
        part_group[beg:beg + i, 8] = np.median(item_mean_group)                 
        beg += i
    return part_group

@jit(nopython = True)
def full_group_cal(tmp_g, step, name_dict, beg = 0):
    m = 2
    ret1 = np.zeros((tmp_g.shape[0], m))
    ret2 = np.zeros((tmp_g.shape[0], m))
    ret3 = np.zeros((tmp_g.shape[0], m))
    
    ret4 = np.zeros((tmp_g.shape[0], m))
    ret6 = np.zeros((tmp_g.shape[0], m))
    ret7 = np.zeros((tmp_g.shape[0], m))
    
    ret8 = np.zeros((tmp_g.shape[0], 3 + 2))
    ret9 = np.zeros((tmp_g.shape[0], 2 * 3))
    
    part_idx = name_dict.index('part')
    time_idx = name_dict.index('timestamp')
    item_mean_idx = name_dict.index('item_mean')
    distance_idx = name_dict.index('task_set_distance')
    answer_idx = name_dict.index('answered_correctly')
    for i in step:
        tmp = tmp_g[:beg]
        selected_part = tmp_g[beg, part_idx]
        group_idx = np.where(tmp[:, part_idx] == selected_part)[0]
        if group_idx.shape[0] > 0:
            tmp_group = tmp[group_idx]
            idx1 = np.where(tmp_group[:,  answer_idx] == 0)[0]
            if idx1.shape[0] > 0:
                wrong_time_diff = tmp_group[idx1, time_idx] - np.concatenate((np.full(1, np.nan), 
                                                                              tmp_group[idx1, time_idx][:-1]))
                ret1[beg:beg+i, 0] = np.nanmean(tmp_group[idx1, item_mean_idx])
                ret1[beg:beg+i, 1] = np.nanmedian(tmp_group[idx1, item_mean_idx])
        
                ret2[beg:beg+i, 0] = np.nanmean(tmp_group[idx1, distance_idx])
                ret2[beg:beg+i, 1] = np.nanmedian(tmp_group[idx1, distance_idx])
            
                ret6[beg:beg+i, 0] = np.nanmean(wrong_time_diff)
                ret6[beg:beg+i, 1] = np.nanmedian(wrong_time_diff)
                
#                 ret10[beg:beg+i, 0] = tmp_g[beg, time_idx] - tmp_group[idx1, time_idx][-1]
                
            idx2 = np.where(tmp_group[:, answer_idx] == 1)[0]
            if idx2.shape[0] > 0:
                right_time_diff = tmp_group[idx2, time_idx] - np.concatenate((np.full(1, np.nan),
                                                                              tmp_group[idx2, time_idx][:-1]))
                ret3[beg:beg+i, 0] = np.nanmean(tmp_group[idx2, item_mean_idx])
                ret3[beg:beg+i, 1] = np.nanmedian(tmp_group[idx2, item_mean_idx])
            
                ret4[beg:beg+i, 0] = np.nanmean(tmp_group[idx2, distance_idx])
                ret4[beg:beg+i, 1] = np.nanmedian(tmp_group[idx2, distance_idx])
            
                ret7[beg:beg+i, 0] = np.nanmean(right_time_diff)
                ret7[beg:beg+i, 1] = np.nanmedian(right_time_diff)
                
#                 ret10[beg:beg+i, 1] = tmp_g[beg, time_idx] - tmp_group[idx2, time_idx][-1]
         
            ret8[beg:beg+i, 0] = np.nansum(tmp_group[:, answer_idx])
            ret8[beg:beg+i, 1] = tmp_group[:, answer_idx].shape[0]
            ret8[beg:beg+i, 2] = np.nanmean(tmp_group[:, answer_idx])
            ret8[beg:beg+i, 3] = ret8[beg:beg+i, 1]/tmp.shape[0]
            ret8[beg:beg+i, 4] = ret8[beg:beg+i, 0]/tmp.shape[0]

            ret9[beg:beg+i, 0] = np.nansum(tmp_group[-1:, answer_idx])
            ret9[beg:beg+i, 1] = np.nanmean(tmp_group[-1:, answer_idx])
            
            ret9[beg:beg+i, 2] = np.nansum(tmp_group[-5:, answer_idx])
            ret9[beg:beg+i, 3] = np.nanmean(tmp_group[-5:, answer_idx])
            
            ret9[beg:beg+i, 4] = np.nansum(tmp_group[-10:, answer_idx])
            ret9[beg:beg+i, 5] = np.nanmean(tmp_group[-10:, answer_idx])
        
        beg += i
    ret = np.concatenate((ret1, ret2, ret6, ret3, ret4, ret7, ret8, ret9), axis = 1)
    return ret

@jit(nopython = True)
def user_answer_cal(tmp_g, step, name_dict, item_answer, beg = 0):
    content_idx = name_dict.index('content_id')
    time_idx = name_dict.index('timestamp')
    item_mean_idx = name_dict.index('item_mean')
    answer_idx = name_dict.index('answered_correctly')
    user_answer_idx = name_dict.index('user_answer')
    distance_idx = name_dict.index('task_set_distance')
    item_answer_idx = name_dict.index('correct_answer')
    
    ans = np.zeros((tmp_g.shape[0], 1),)
    tmp_ans = item_answer[tmp_g[:,content_idx].astype(np.uint16), :]
    for k in range(tmp_ans.shape[0]):
        ans[k] = tmp_ans[k, np.uint8(tmp_g[k, user_answer_idx])]
        
    ret1 =  np.zeros((tmp_g.shape[0], 2))
    ret2 = np.zeros((tmp_g.shape[0], 3 * 2))
    tmp_ = np.concatenate((tmp_g, ans), axis = 1)
    for i in step:
        tmp = tmp_[:beg]
        if tmp.shape[0] > 0:
            ret1[beg:beg+i, 0] = np.mean(tmp[:, -1])
            ret1[beg:beg+i, 1] = np.median(tmp[:, -1])
            for j in range(i):
                item_ans = tmp[beg + j, item_answer_idx]
                ans_index = np.where(tmp[:, user_answer_idx] == item_ans)[0]
                if ans_index.shape[0] > 0:
                    ret2[beg+j, 0] = np.sum(tmp[ans_index, answer_idx])
                    ret2[beg+j, 1] = np.mean(tmp[ans_index, answer_idx])
                    ret2[beg+j, 2] = tmp[ans_index, answer_idx].shape[0]
                    
                ans_index = np.where(tmp[:, item_answer_idx] == item_ans)[0]
                if ans_index.shape[0] > 0:
                    ret2[beg+j, 3] = np.sum(tmp[ans_index, answer_idx])
                    ret2[beg+j, 4] = np.mean(tmp[ans_index, answer_idx])
                    ret2[beg+j, 5] = tmp[ans_index, answer_idx].shape[0]
        beg += i
    ret = np.concatenate((ret1, ret2), axis = 1)
    return ret


def last_group_cal_wrap(tmp_g, name_dict, item_answer_dict, beg = 0):
    if beg == 0:
        step = np.unique(tmp_g[:, 1], return_counts=True)[1]
        ds1 = last_group_cal(tmp_g, step, name_dict, beg)
        ds2 = full_group_cal(tmp_g, step, name_dict, beg)
        ds3 = divide_agg(tmp_g, step, name_dict, beg)
        ds4 = user_answer_cal(tmp_g, step, name_dict, item_answer_dict, beg)
    else:
        step = np.array([beg])
        ds1 = last_group_cal(tmp_g, step, name_dict, tmp_g.shape[0] - beg)
        ds2 = full_group_cal(tmp_g, step, name_dict, tmp_g.shape[0] - beg)
        ds3 = divide_agg(tmp_g, step, name_dict, tmp_g.shape[0] - beg)
        ds4 = user_answer_cal(tmp_g, step, name_dict, item_answer_dict, tmp_g.shape[0] - beg)
        
    ds = np.concatenate((ds1, ds2, ds3, ds4), axis = 1)
    # ds = np.concatenate((ds1, ds2, ds3), axis = 1)
    return ds


def test_last_group_feature(valid_gp, rolling_gp, name_dict, valid_idx, item_answer_dict):
    res = []
    for i in range(len(valid_gp)):
        tmp_g = valid_gp[i]
        valid_shape = tmp_g.shape[0]
        if tmp_g[0,0] in rolling_gp:
            tmp_rolling = rolling_gp[tmp_g[0, 0]]
            tmp_g = np.concatenate([tmp_rolling, tmp_g])
        
        tmp_res = last_group_cal_wrap(tmp_g, name_dict, item_answer_dict, valid_shape)
        tmp_res = tmp_res[-valid_shape:]
        res.append(tmp_res)
    ans = np.concatenate(res)
    ans = resort_array(ans, valid_idx)
    return ans


def get_last_name():


    last_group_name = []
    last_group_name += ['begin_time_diff', 'end_time_diff', 'part_duration_time', 'part_time_diff_mean']
    last_group_name += ['part_session_mean', 'part_session_sum', 'part_session_count']
    last_group_name += ['last_part_item_mean', 'last_part_item_median']

    last_group_name += [f'full_group{g}_{var}_{func}'  for g in [0, 1]
                        for var in ['item_mean', 'task_set_distance', 'timestamp'] for func in ['mean', 'median']]
    last_group_name += ['part_sum', 'part_count', 'part_mean']
    last_group_name += ['part_count_global_ratio', 'part_sum_global_ratio']

    last_group_name += [f'part_{func}_{i}' for i in [1, 5, 10] for func in ['sum', 'mean']]
    new_func_list = ['mean', 'median']
    last_group_name += [f"cum_answer0_{func}_{val}" 
                 for val in ['item_mean', 'task_set_distance'] 
                 for func in new_func_list]
    last_group_name += [f"cum_answer1_{func}_{val}" 
                 for val in ['item_mean', 'task_set_distance'] 
                 for func in new_func_list]
    last_group_name += ["cum_answer0_time_diff", "cum_answer1_time_diff"]
    last_group_name += [f"global_task_set_shift{i}" for i in range(1, 6)]
# last_group_name += [f"global_item_mean_shift{i}" for i in range(1, 6)]

    last_group_name += [f"cum_answer0_{func}_{val}" 
                 for val in ['wrong_time_diff'] 
                 for func in new_func_list]
    last_group_name += [f"cum_answer1_{func}_{val}" 
                 for val in ['right_time_diff'] 
                 for func in new_func_list]

    last_group_name += ['last_content_id_time_diff']
    # last_group_name += ['content_correct_mean', 'content_correct_sum']
    last_group_name += ['content_correct_mean', 'content_correct_sum', 'content_correct_count']
    last_group_name += ['hard_answer0_time', 'hard_answer1_time']

    last_group_name += [f'full_bundle_{var}_{func}' 
         for var in ['item_mean', 'task_set_distance', 'timestamp'] for func in ['mean', 'median']]
    last_group_name += ['bundle_sum', 'bundle_count', 'bundle_mean']

    last_group_name += [f'user_trend_{func}' for func in ['mean', 'median']]
    last_group_name += [f'user_trend_roll_{p}_{func}' 
                    for p in ['user_ans', 'item_ans'] 
                    for func in ['sum', 'mean', 'count']]
    return last_group_name


def pre_merge_group(valid_gp, rolling_gp):
    valid_shape_list = []
    for i in range(len(valid_gp)):
        tmp_g = valid_gp[i]
        valid_shape = tmp_g.shape[0]
        if tmp_g[0,0] in rolling_gp:
            tmp_rolling = rolling_gp[tmp_g[0, 0]]
            tmp_g = np.concatenate([tmp_rolling, tmp_g])
        valid_gp[i] = tmp_g
        valid_shape_list.append(valid_shape)
    return valid_gp, valid_shape_list


def test_last_group_wrap(tmp_g, valid_shape, name_dict, item_answer_dict):
    tmp_res = last_group_cal_wrap(tmp_g, name_dict, item_answer_dict, valid_shape)
    tmp_res = tmp_res[-valid_shape:]
    return tmp_res

def test_parallel_last_group_wrap(group, valid_shape_list, name_dict, valid_idx, item_answer_dict):
    res = Parallel(n_jobs = 4, backend = 'loky')\
              (delayed(test_last_group_wrap)(group[i], valid_shape_list[i], name_dict, item_answer_dict)
              for i in range(len(group)))
    ans = np.concatenate(res)
    ans = resort_array(ans, valid_idx)
    return ans


def test_last_group_wrapper(test, last_part_group, roll_keep, item_answer_dict):
    test_gp, valid_idx, name_dict = rolling_feat_group(test, roll_keep)
    test_gp, valid_shape_list = pre_merge_group(test_gp, last_part_group)
    test_last_group_ds = test_parallel_last_group_wrap(test_gp, valid_shape_list, 
                                                       name_dict, valid_idx, item_answer_dict)
    return test_last_group_ds
