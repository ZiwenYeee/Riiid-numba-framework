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


def timeit(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, np.round(time2-time1, 2)))

        return ret
    return wrap

@timeit
def initial_dict(train):
    user_dict = train.groupby(['user_id'])['answered_correctly'].agg(["sum", 'count'])
    user_dict.columns = ['user_sum', 'user_count']
    user_dict = user_dict.to_dict('index')
    
    item_dict = train.groupby(['content_id'])['answered_correctly'].agg(["sum", 'count'])
    item_dict.columns = ['item_sum', 'item_count']
    item_dict = item_dict.to_dict('index')
    return user_dict, item_dict
