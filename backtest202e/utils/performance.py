import pandas as pd
import numpy as np

# annualized return
def ann_mean(x):
    return x.mean()*52

# annualized volatility
def ann_vol(x):
    return x.std()*52**0.5

"""
# compound annual growth rate
def cagr(x):
    return x.add(1).prod()**(12/x.count())-1"""

# information ratio
def ann_ir(x):
    if ann_vol(x)==0:
        return np.nan
    else:
        return ann_mean(x)/ann_vol(x)
    
#t_stats - is monthly return significantly different from 0
def tstat(x):
    if x.std==0:
        return np.nan
    else:
        return x.mean()/x.std()*x.count()**0.5

# percentage of positive returns
def consistency(x):
    return (x>=0).sum()/x.count()

# time series of monthly drawdown using last monthly high watermark
def drawdown(x):
    cum_x=x.cumsum()
    return (cum_x.cummax()-cum_x)

# maximum drawdown 
def max_drawdown(x):
    return drawdown(x).max()
