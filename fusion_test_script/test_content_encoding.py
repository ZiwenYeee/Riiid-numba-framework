from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from numba import jit
from numba import types
from numba.typed import Dict
import functools, time
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

@jit(nopython = True, parallel = True)
def map_wrap(vals, q_dict, m = 4):
    ds = np.zeros((vals.shape[0], m))
    for i in prange(vals.shape[0]):
        ds[i] = q_dict[vals[i]]
    return ds

@timeit
def answer_dict_init_(train, path = './questions.csv'):
    item_answer = train.groupby(['content_id', 'user_answer'])['user_answer'].agg('count').unstack()
    item_answer = item_answer.to_numpy()

    total_sum = np.sum(item_answer, axis = 1)
    for i in range(4):
        item_answer[:, i] = item_answer[:, i]/total_sum

    le = LabelEncoder()
    questions = pd.read_csv(path)
    questions['le_tag'] = le.fit_transform(questions['tags'].astype(str))
    q_dict = questions.set_index('question_id')[['bundle_id', 'correct_answer', 'part', 'le_tag']].values
    cols = [f"answer_ratio_{func}" for func in range(4)] + ['bundle_id', 'correct_answer', 'part', 'le_tag']
    return item_answer, q_dict, cols

@timeit
def content_encoder1(train):
    pre_ds = train[['user_id', 'content_id', 'answered_correctly']].copy()
    pre_ds['user_mean'] = train.groupby(['user_id'])['answered_correctly'].transform('mean')
    enc_0 = pre_ds[pre_ds['answered_correctly'] == 0].groupby(['content_id']).agg(
    {"user_mean":['mean', 'median']})
    enc_1 = pre_ds[pre_ds['answered_correctly'] == 1].groupby(['content_id']).agg(
    {"user_mean":['mean', 'median']})
    enc_0.columns = [f"question_correct_user_ablility_{col[1]}" for col in enc_0.columns.ravel()]
    enc_1.columns = [f"question_wrong_user_ablility_{col[1]}" for col in enc_1.columns.ravel()]
    enc_df = pd.DataFrame(np.arange(13523), columns = ['content_id'])
    enc_df = enc_df.merge(enc_0, on = ['content_id'], how = 'left')
    enc_df = enc_df.merge(enc_1, on = ['content_id'], how = 'left')
    enc_df.fillna(0, inplace = True)
    col_name = enc_df.iloc[:,1:].columns.to_list()
    col_array = enc_df.iloc[:,1:].to_numpy()
    return col_array, col_name

@timeit
def content_encoder2(train):
    pre_ds = train[['user_id', 'content_id', 'answered_correctly']].copy()
    pre_ds['user_mean'] = train.groupby(['user_id'])['answered_correctly'].transform('mean').round(1)
    enc_df = pd.DataFrame(np.arange(13523), columns = ['content_id'])
    yy = pre_ds.pivot_table(index='content_id', columns='user_mean', 
                                 values='answered_correctly', aggfunc='mean').reset_index()
    yy.columns = ['content_id'] + [f'mean_{i}_ratio' for i in range(11)]
    enc_df = pd.merge(enc_df, yy, on = ['content_id'], how = 'left')
    enc_df.fillna(0, inplace = True)
    col_name = enc_df.iloc[:,1:].columns.to_list()
    col_array = enc_df.iloc[:,1:].to_numpy()
    return col_array, col_name


@timeit
def read_enc_data(path = './'):
    df_w2v = pd.read_csv(path + "df_w2v.csv")
    df_svd = pd.read_csv(path + "df_svd.csv")
    df_tag = pd.read_csv(path + "tag_w2v_features.csv")

    df_enc1 = pd.read_csv(path + "content_id_real_time.csv")
    df_enc1.fillna(0, inplace = True)

    df_enc2 = pd.read_csv(path + "content_id_task_set_distance.csv")
    df_enc2.fillna(0, inplace = True)

    tag_df = df_tag.sort_values(['content_id']).iloc[:,1:].to_numpy()
    w2v_df = df_w2v.sort_values(['content_id']).iloc[:,1:].to_numpy()
    svd_df = df_svd.sort_values(['content_id']).iloc[:,1:].to_numpy()
    enc_df1 = df_enc1.sort_values(['content_id']).iloc[:,1:].to_numpy()
    enc_df2 = df_enc2.sort_values(['content_id']).iloc[:,1:].to_numpy()
    
    
    emb_df = np.concatenate((w2v_df, svd_df, tag_df, enc_df1, enc_df2), axis = 1)
    
    enc_name1 = df_enc1.iloc[:,1:].columns.to_list()
    enc_name2 = df_enc2.iloc[:,1:].columns.to_list()
    tag_name = df_tag.columns[1:].to_list()
    w2v_name = [f'word2vec_{i}' for i in range(5)] + [f'svd_{i}' for i in range(5)]
    w2v_name += tag_name
    w2v_name += enc_name1
    w2v_name += enc_name2
    return emb_df, w2v_name

@timeit
def content_emb_dict_init_(train, path = './', question_path = './questions.csv'):
    answer_dict, q_dict, question_file_name = answer_dict_init_(train, question_path)
    enc_dict, enc_name = content_encoder1(train)
    enc_dict1, enc_name1 = content_encoder2(train)
    emb_df, w2v_name = read_enc_data(path)
    
    content_emb_dict = np.concatenate((answer_dict, q_dict, enc_dict, emb_df, enc_dict1), axis = 1)
    emb_cols = question_file_name + enc_name + w2v_name + enc_name1
    return content_emb_dict, emb_cols

# @timeit
def initial_embedding_feature(ds, content_emb_dict):
    emb_size = content_emb_dict.shape[1]
    feats = map_wrap(ds['content_id'].values, content_emb_dict, emb_size)
    return feats

