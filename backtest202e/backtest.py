#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from pandas.api.types import is_list_like
import warnings
import pathlib as pl
import json
from backtest202e.utils.utils import*
from backtest202e.utils.performance import*
from backtest202e.utils.formatting import*
from .strategy import Strategy
import pdb
from datetime import datetime, timedelta


# 1. add turnover, hit rate and persistency
# 2. implement transaction cost
# 3. finish decorator 
# 4. implement different return frequency

class Backtest:
    def __init__(self, strategy, df_ret, return_col, trans_cost=0):
        self._strategy = strategy
        self._df_ret = df_ret
        self._trans_cost = trans_cost
        self._return_col = return_col
    
    @property
    def strategy(self):
        return self._strategy
    
    @strategy.setter
    def strategy(self,strategy):
        if strategy is not None:
            if hasattr(strategy, '_portfolios'):
                self._strategy = strategy
            else:
                raise TypeError('startegy has to be of strategy class type')
        else:
            raise TypeError('strategy has to be specified')
            
    @property
    def df_ret(self):
        return self._df_ret
    @df_ret.setter
    def df_ret(self,df_ret):
        pass
    
    @property
    def return_col(self):
        return self._return_col
    @return_col.setter
    def return_col(self, return_col):
        pass
    
    #@property
    
    #@tran_cost.setter
    
    @property
    def portfolios(self):
        return self._portfolios
    
    @property
    def _top_basket(self):
        return self._baskets.factor_basket.min()
    
    @property
    def _bot_basket(self):
        return self._baskets.factor_basket.max()
    
    def run(self):
        # prepare data
        self._attach_testing_data()
        # compute backtest returns
        self._compute_universe_returns()
        self._compute_basket_returns()
        self._compute_long_short_returns()
    
    # put return data and factor portfolio data together
    def _attach_testing_data(self):
        self._portfolios = self._strategy.portfolios.copy()
        self._df_ret[self._strategy.date_col] = pd.to_datetime(self._df_ret[self._strategy.date_col])
        self._portfolios = self._portfolios.merge(self._df_ret[[self._strategy.id_col, self._strategy.date_col,self._return_col]], on =[self._strategy.id_col, self._strategy.date_col],how='left').fillna({self._return_col:0}) 
    
    # get universe/benchmark return
    def _compute_universe_returns(self):
        self._universe = self._portfolios.assign(fwd_ret_weighted = lambda df: df['universe_weight']*df[self._return_col]).groupby([self._strategy.date_col,'factor'])['fwd_ret_weighted'].sum().to_frame('fwd_univ_ret').reset_index()
    
    # compute return of each quantile basket portfolio
    def _compute_basket_returns(self):
        self._baskets = self._portfolios.assign(fwd_ret_weighted = lambda df: df['basket_weight']*df[self._return_col]).groupby([self._strategy.date_col,'factor','factor_basket'])['fwd_ret_weighted'].sum().to_frame('fwd_basket_ret').reset_index()
        self._baskets = pd.merge(self._baskets,self._universe[[self._strategy.date_col,'factor','fwd_univ_ret']], on = [self._strategy.date_col,'factor']).assign(fwd_act_ret=lambda df: df['fwd_basket_ret']-df['fwd_univ_ret']).drop(columns='fwd_univ_ret')
    
    # compute return of long short portfolio
    def _compute_long_short_returns(self):
        self._long_short = self._baskets[self._baskets.factor_basket.isin([self._top_basket,self._bot_basket])].copy()
        #self._long_short['count'] = self._long_short.groupby(['date','factor'])['factor_basket'].transform('count')
        #self._long_short = self._long_short.loc[self._long_short['count'] == 2]
        long_short_ret_n = self._long_short.groupby([self._strategy.date_col,'factor'])['fwd_basket_ret'].count()
        if any(long_short_ret_n !=2):
            raise ValueError('one or more bottom or top basket return is missing')
        self._long_short['fwd_basket_ret_signed'] = np.where(self._long_short['factor_basket']==self._bot_basket,self._long_short['fwd_basket_ret']*-1,self._long_short['fwd_basket_ret'])
        self._long_short = self._long_short.groupby([self._strategy.date_col,'factor'])['fwd_basket_ret_signed'].agg(fwd_ls_ret='sum').reset_index()

    def _check_run(self):
        if not hasattr(self,'_portfolios'):
            raise AttributeError('factor basket return has to be computed first')
    
    # organize returns
    def returns(self, baskets=['Q1'], long_short=True, universe=False, active=True, longdata=False):
        self._check_run()
        if not baskets and not long_short and not universe:
            raise ValueError('at least one return type has to be selected')
        rets=[]
        
        # basket return
        if len(baskets)>0:
            baskets_rets = self._baskets[self._baskets.factor_basket.isin(baskets)].copy()
            if active:
                baskets_rets = baskets_rets.drop(columns='fwd_basket_ret').rename(columns={'fwd_act_ret':'ret'})
            else:
                baskets_rets = baskets_rets.drop(columns='fwd_act_ret').rename(columns={'fwd_basket_ret':'ret'})

            rets.append(baskets_rets)

        # long short return
        if long_short:
            long_short_rets=self._long_short.rename(columns={'fwd_ls_ret':'ret'}).assign(factor_basket=self._top_basket+'-'+self._bot_basket)
            rets.append(long_short_rets)
        
        # universe return
        if universe:
            universe_rets=self._universe.rename(columns={'fwd_univ_ret':'ret'}).assign(factor_basket='Universe')
            rets.append(universe_rets)
        
        # convert forward returns to backward returns
        rets = pd.concat(rets,sort=False).rename(columns={self._strategy.date_col:'date'}).sort_values(['factor','date','factor_basket']).reset_index(drop=True)
        rets.date = rets.date + pd.offsets.Week()
        ### IMPLEMENT TRANSACTION COST HERE ###
        
        # convert to wide format
        if not longdata:
            rets = rets.set_index(['date','factor','factor_basket']).unstack(['factor','factor_basket']).rename_axis(None).droplevel(0,axis=1)
            rets = rets.reindex(columns=sorted(rets.columns))
        
        return rets
    
    # compute rank IC 
    def ic(self, longdata=False):
        self._check_run()
        if self._strategy.higher_better:
            correl_sign=1
        else:
            correl_sign=-1
        self._portfolios.rename(columns={self._return_col:'mret_fwd'},inplace=True)
        rank_corr = self._portfolios.groupby([self._strategy.date_col,'factor'])[['factor_score','mret_fwd']].corr(method='spearman')
        rank_corr.index.names=[self._strategy.date_col,'factor','factor_2']
        ic = rank_corr.reset_index().query('factor_2== "mret_fwd"').drop(columns=['mret_fwd','factor_2']).rename(columns={self._strategy.date_col:'date','factor_score':'ic'}).assign(ic = lambda df: df.ic*correl_sign)
        ic.date = ic.date + pd.offsets.Week()
        
        if not longdata:
            ic = ic.set_index(['date','factor']).unstack().rename_axis(None).droplevel(0,axis=1)
        
        return ic
    
    # summary table
    def summary(self, baskets=['Q1'], long_short=True, universe=False, window=None,extend=False):
        rets = self.returns(baskets=baskets,long_short=long_short,universe=universe, active=False,longdata=True)
        act_rets = self.returns(baskets=baskets,long_short=long_short,universe=universe, active=True,longdata=True).rename(columns={'ret':'act_ret'})
        if universe:
            act_rets.loc[act_rets.factor_basket=='Universe','act_ret']=0
        
        summary_data=pd.merge(rets, act_rets, on=['factor','factor_basket','date'],how='outer')
        summary_data = pd.merge(summary_data, self.ic(longdata=True),on=['factor','date'],how='left')
        
        if window:
            window_dates=summary_data.date.drop_duplicates().nlargest(window)
            summary_data=summary_data[summary_data.date.isin(window_dates)]
        
        summary_table = summary_data.groupby(['factor','factor_basket']).agg(Alpha = ('act_ret',ann_mean),
                                                                            Tracking_Error = ('act_ret',ann_vol),
                                                                            IR = ('act_ret',ann_ir),
                                                                            T_stat = ('act_ret',tstat),
                                                                            IC = ('ic','mean')).T
        if extend:
            extend_summary_table = summary_data.groupby(['factor','factor_basket']).agg(Avg_Return=('ret',ann_mean),
                                                                    Avg_Vol = ('ret',ann_vol),
                                                                    Consistency=('act_ret',consistency),
                                                                    Periods=('ret','count'),
                                                                    Max_Drawdown=('ret',max_drawdown)).T
            summary_table = pd.concat([summary_table, extend_summary_table],sort=False)
            
        summary_table =summary_table.reindex(columns=sorted(summary_table.columns))
        return format_summary(summary_table)
    
    def cumulative_returns(self,baskets=['Q1'],long_short=True,universe=False, active=True,longdata=False):
        rets = self.returns(baskets=baskets,long_short=long_short, universe=universe, active=active,longdata=False)
        rets = self.returns(baskets=baskets,long_short=long_short, universe=universe, active=active, longdata=False)
        #beom = pd.offsets.BusinessMonthEnd()
        bw = pd.offsets.Week()
        # assign zero to first month
        #zero_rets = rets[['factor','factor_basket']].drop_duplicates().assign(ret=0,date=rets.date.min()-bw)
        #rets = rets.append(zero_rets,sort=False).sort_values(['factor','factor_basket','date']).reset_index(drop=True)
        #rets.ret = rets.groupby(['factor','factor_basket']).ret.transform(lambda x: x.cumsum())
        
        cum_rets = rets.agg(lambda x : x.cumsum())
        #if not longdata:
         #   rets = rets.set_index(['date','factor','factor_basket']).unstack(['factor','factor_basket']).rename_axis(None).droplevel(0,axis=1)
          #  rets = rets.reindex(columns=sorted(rets.columns))
        return cum_rets
    
    def plot_performance(self,baskets=['Q1'],long_short=True,universe=False, active=True):
        rets = self.cumulative_returns(baskets=baskets,long_short=long_short, universe=universe, active=active,longdata=False)
        rets.columns = rets.columns.map('{0[0]} - {0[1]}'.format)
        line_plot(rets)
        return rets
    
    def plot_rolling_performance(self,window=12,baskets=['Q1'],long_short=True,universe=False, active=False):
        rets = self.returns(baskets=baskets,long_short=long_short, universe=universe, active=active, longdata=False)
        roll_rets = rets.rolling(window).agg(lambda x : x.sum())
        roll_rets = roll_rets.iloc[(window-1):]
        roll_rets.columns = roll_rets.columns.map('{0[0]} - {0[1]}'.format)
        line_plot(roll_rets)
        return rets
        
    def plot_drawdown(self,baskets=['Q1'],long_short=True,universe=False, active=False):
        rets = self.returns(baskets=baskets,long_short=long_short, universe=universe, active=active, longdata=False)
        drawdowns = rets.transform(drawdown)
        drawdowns.columns=drawdowns.columns.map('{0[0]} - {0[1]}'.format)
        line_plot(drawdowns)
        return rets
    
    def to_excel(self,filepath):
        xl_writer = pd.ExcelWriter(filepath,engine='xlsxwriter')
        self.summary().to_excel(xl_writer,sheet_name='Summary',na_rep="#N/A")
        datetime_index_to_date_index(self.returns(active=False)).to_excel(xl_writer,sheet_name='Returns',na_rep="#N/A",freeze_panes=[2,1])
        datetime_index_to_date_index(self.returns(active=True)).to_excel(xl_writer,sheet_name='Active Returns',na_rep="#N/A",freeze_panes=[2,1])
        datetime_index_to_date_index(self.cumulative_returns(active=False)).to_excel(xl_writer,sheet_name='Cumulative Returns',na_rep="#N/A",freeze_panes=[2,1])
        datetime_index_to_date_index(self.cumulative_returns(active=True)).to_excel(xl_writer,sheet_name='Cumulative Active Returns',na_rep="#N/A",freeze_panes=[2,1])
        datetime_index_to_date_index(self.ic()).to_excel(xl_writer,sheet_name='IC',na_rep="#N/A",freeze_panes=[1,1])
        xl_writer.save()
        
        



