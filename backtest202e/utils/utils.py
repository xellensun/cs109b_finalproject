import pandas as pd
import pathlib as pl 
import string
import random
from pandas.tseries.offsets import MonthEnd

# check file path before writing a file
def write_check_path(path, overwrite=False):
    path = pl.Path(path)
    if path.exists() and not overwrite:
        raise FileExistError(f'{path} exists and overwrite is set to False!')
    return path

# check file path before reading a file
def read_check_path(path):
    path = pl.Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} does not exist.')
    return path

# get random unique column name for pd.DataFrame
def get_random_col_name(df, length=10):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df has to be pd.DataFrame.')
    col_name = df.columns[0]
    while col_name in df:
        col_name = ''.join(random.sample(string.ascii_letters,length))
    return col_name

# lag or forward data
def shift_data(df, group, col, shift_num):
    res = df[group+[col]].set_index(group)
    res = res.unstack().shift(shift_num)
    res = res.stack(dropna=False).reset_index()
    if shift_num >=0:
        label = '_lag'
    else:
        label = '_fwd'
    df = df.merge(res, on=group, how='left', suffixes=('',label))
    return df

# generate month end date
def month_end(df, num,name='date'):
    df['day']=1
    df['date']=pd.to_datetime(df[['year','month','day']])+MonthEnd(num+1)
    df.drop(['day'],axis=1,inplace=True)
    return df