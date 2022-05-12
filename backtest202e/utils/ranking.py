#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
from pandas.api.types import is_list_like


# In[55]:


# z_score 
def z_score(x, higher_better=True):
    x=check_series(x)
    if not higher_better:
        x=-x
    return (x-x.mean())/x.std()


# In[53]:


# return percentile rank as n/N+1
def pct_rank(x, higher_better=True):
    x=check_series(x)
    return x.rank(method='average',ascending=not higher_better).transform(lambda a: a/(a.count()+1))


# In[54]:


# quantile cut 
"""q: int or list like float
   Return: series of basket labels in numeric format
   cut command creates equispaced bins but frequency of samples is unequal in each bin
   qcut command creates unequal size bins but frequency of samples is equal in each bin."""
def quantile_cut(x, q, higher_better=True):
    x=check_series(x)
    baskets = pd.qcut(x,q,duplicates='drop',labels=False)+1
    if higher_better:
        baskets=baskets.transform(lambda a: (a-a.max()).abs().add(1))
    baskets = format_basket_name(baskets)
    return baskets


# In[56]:


# check if x is pd.series
def check_series(x):
    if not isinstance(x,pd.Series):
        x=pd.Series(x)
    return x


# In[57]:


# format basket name as 'Qi'string
def format_basket_name(baskets):
    baskets = check_series(baskets)
    if not baskets.isna().all():
        lead_zero_num=int(np.log10(baskets.max()))+1
        format_str='Q{0:0'+f'{lead_zero_num:d}'+'d}'
        baskets[pd.notnull(baskets)]=baskets[pd.notnull(baskets)].astype(int).map(format_str.format)
    return baskets

