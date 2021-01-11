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

@timeit   
def initial_attempt(train):
        
    # add user content attempts
    user_attempts = \
    train.groupby(['user_id', 'content_id'])[['content_id']].count().astype(np.uint8)
    user_attempts -= 1
    user_attempts.columns = ['count']
    user_attempts = user_attempts.reset_index().set_index('user_id')
    state = user_attempts.groupby('user_id').apply(lambda x: np.array(x, dtype=np.uint16)).to_dict()
    
#     tmp = train.groupby(['user_id', 'content_id'])[['content_id']].transform('cumcount')
#     tmp = pd.DataFrame(tmp, columns = ['attempt'])
    return state

def dict_trans(state_dict):
    float_array = types.Array(types.uint16, 2, 'A')
    count_dict = Dict.empty(key_type = types.uint32, value_type = float_array)
    for key, vals in state_dict.items():
        count_dict[key] = np.asarray(vals, dtype=np.uint16)
    return count_dict


@jit(nopython = True)
def state_cal_wrap(vals, state_dict):
    attempt = np.zeros(vals.shape[0])
    for i in range(vals.shape[0]):
        user_id = vals[i][0]
        content_id = vals[i][1]
        if user_id in state_dict:
            user_array = state_dict[user_id]
            idx = np.where(user_array[:,0] == content_id)[0]
            if  idx.shape[0] > 0:
                state_dict[user_id][idx, 1] = user_array[idx, 1] + 1
            else:
                add_array = np.array([[content_id, 0]], dtype=np.uint16)
                state_dict[user_id] =  np.concatenate((state_dict[user_id], add_array))
        else:
            state_dict[user_id] = np.array([[content_id, 0]], dtype=np.uint16)
        idx = np.where(state_dict[user_id][:,0] == content_id)[0]
        count = state_dict[user_id][idx, 1][0]
        attempt[i] = count
    return attempt, state_dict

from numba import jit
# @timeit
def state_feature(df, state):
    vals = df[['user_id', 'content_id']].values
    attempt, state = state_cal_wrap(vals, state)
    attempt = pd.DataFrame(attempt, columns = ['attempt'], index = df.index)
    return attempt, state