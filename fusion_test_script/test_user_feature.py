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

def test_iter_user_func(vals, user_dict):
    cnt = 0
    feats = np.zeros((vals.shape[0], 2), dtype=np.int64)
    for row in prange(vals.shape[0]):
        if vals[row] in user_dict:
            feats[cnt, 0] = user_dict[vals[row]]['user_count']
            feats[cnt, 1] = user_dict[vals[row]]['user_sum']
        else:
            feats[cnt, 0] = 0
            feats[cnt, 1] = 0
        cnt += 1
    return feats

def test_user(df, user_dict):
    feats = test_iter_user_func(df['user_id'].values, user_dict)
    feats = pd.DataFrame(feats, columns = ['user_count', 'user_sum'], index = df.index)
    feats['user_mean'] = feats['user_sum']/(1e-7 + feats['user_count'])
    return feats

@timeit
def initial_dict(train):
    user_dict = train.groupby(['user_id'])['answered_correctly'].agg(["sum", 'count'])
    user_dict.columns = ['user_sum', 'user_count']
    user_dict = user_dict.to_dict('index')
    
    item_dict = train.groupby(['content_id'])['answered_correctly'].agg(["sum", 'count'])
    item_dict.columns = ['item_sum', 'item_count']
    item_dict = item_dict.to_dict('index')
    return user_dict, item_dict


def update_user_dict(user_dict, previous_test_df):
    for vals in previous_test_df[['user_id', 'content_id', 'answered_correctly']].values:
        user_id = vals[0]
        content_id = vals[1]
        answer_record = vals[2]
        if user_id in user_dict:
            user_dict[user_id]['user_count'] += 1
            user_dict[user_id]['user_sum'] += answer_record
        else:
            user_dict[user_id] = {'user_count':1, 'user_sum':answer_record}
    return user_dict
