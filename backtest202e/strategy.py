import pandas as pd
import numpy as np
from backtest202e.utils.ranking import*
from backtest202e.utils.utils import*
import warnings
from pandas.api.types import is_list_like


# 1. need to think about where to implement incl universe dummy (on agenda)
# 2. holding period? - moving average of signals, equal weight past n months (implement in main)
# 3. should we replace na in quantile analysis? (on agenda)
# 4. coverage function
# 5. output portfolio name
# 6. portfolio transition matrix

class Strategy:
    def __init__(self, id_col, date_col, bmk_col=None, weight_col=None, factors=None, grouping=None, quantiles=5, higher_better=True, standardize=True):
        self._id_col = id_col
        self._date_col = date_col
        self._bmk_col = bmk_col
        self._weight_col= weight_col
        self._factors = factors
        self._grouping= grouping
        self._quantiles=quantiles
        self._higher_better = higher_better
        self._standardize = standardize
    
    @property
    def id_col(self):
        return self._id_col
    
    @id_col.setter
    def id_col(self, id_col):
        if id_col:
            if isinstance(id_col, str):
                self._id_col = id_col
            else:
                raise TypeError('id_col has to be a string with the stock id column name')
        else:
            raise TypeError('id_col has to be specified')
    
    @property
    def date_col(self):
        return self._date_col
    
    @date_col.setter
    def date_col(self, date_col):
        if date_col:
            if isinstance(date_col, str):
                self._date_col = date_col
            else:
                raise TypeError('date_col has to be a string with the date column name')
        else:
            raise TypeError('date_col has to be specified')
            
    @property
    def bmk_col(self):
        return self._bmk_col
    
    @bmk_col.setter
    def bmk_col(self, bmk_col):
        if bmk_col:
            if isinstance(bmk_col, str):
                self._bmk_col = bmk_col
            else:
                raise TypeError('bmk_col has to be a string with bmk weight column name')
        else:
            raise TypeError('bmk_col has to be specified')
    
    @property
    def weight_col(self):
        return self._weight_col
    
    @weight_col.setter
    def weight_col(self, weight_col):
        if weight_col is not None:
            if isinstance(weight_col, str):
                self._weight_col = weight_col
            else:
                raise TypeError('weight_col has to be a string with the basket portfolio weight column name')
        else:
            self._weight_col=None
    
    @property
    def factors(self):
        return self._factors
    
    @factors.setter
    def factors(self, factors):
        if factors is not None:
            if isinstance(factors, str):
                self._factors=[factors]
            elif is_list_like(factors):
                if all([isinstance(i, str) for i in factors]):
                    self._factors = list(factors)
                else:
                    raise TypeError('At least one element of factors is not a string')
            else:
                raise TypeError('factors has to be a string or list like object with column names')
        else:
            raise TypeError('At least one factor has to be specified')
            
    @property
    def grouping(self):
        return self._grouping
    
    @grouping.setter
    def grouping(self, grouping):
        if grouping is not None:
            if isinstance(grouping, str):
                self._grouping=[grouping]
            elif is_list_like(grouping):
                if all([isinstance(i, str) for i in grouping]):
                    self._grouping = list(grouping)
                else:
                    raise TypeError('At least one element of grouping is not a string')
            else:
                raise TypeError('grouping has to be a string or list like object with column names')
        else:
            self._grouping=None
    
    @property
    def quantiles(self):
        return self._quantiles
    
    @quantiles.setter
    def quantiles(self,quantiles):
        if quantiles is not None:
            if isinstance(quantiles,(int, tuple, list)):
                self._quantiles = quantiles
            else:
                raise TypeError('quantiles has to been defined as either an integer, a tuple or a list')
        else:
            self._quantiles=None
 
    @property 
    def higher_better(self):
        return self._higher_better
    
    @higher_better.setter
    def higher_better(self, higher_better):
        if isinstance(higher_better, bool):
            self._higher_better=higher_better
        else:
            raise TypeError('higher_better has to be in boolean format')
    
    @property
    def standardize(self):
        return self._standardize
    
    @standardize.setter
    def standardize(self, standardize):
        if isinstance(standardize, bool):
            self._standardize=standardize
        else:
            raise TypeError('standardize has to be in boolean format')

    
    @property
    def _keep_cols(self):
        keep_cols=[self._id_col,self._date_col,self._bmk_col]
        if self._weight_col is not None:
            keep_cols.append(self._weight_col)
        if self.grouping is not None:
            keep_cols.extend(self._grouping)
        return keep_cols
        
    @property
    def _groupby_cols(self):
        if self.grouping is not None:
            return [self._date_col, 'factor', *self._grouping]
        else:
            return [self._date_col, 'factor']
        
    @property
    def _has_data(self):
        return hasattr(self, '_portfolios')
    
    @property 
    def portfolios(self):
        if hasattr(self, '_portfolios'):
            return self._portfolios
    
    def run(self, data):
        self._attach_data(data)
        self._process_data()
    
    # check if data format is correct for processing
    def _check_data(self, data):
        # check if dataframe
        if not isinstance(data, pd.DataFrame):
            raise TypeError('data has to be pd.DataFrame')
        # check factors
        for factor in self._factors:
            if factor not in data:
                raise KeyError(f'Factor "{factor}" is not among data columns.')
        # check attr columns
        for col in self._keep_cols:
            if col not in data:
                raise KeyError(f'Column "{col}" is not among data columns.')
    
    # add portfolio property        
    def _attach_data(self,data):
        if self._has_data:
            raise warnings.warn('Existing portfolios data will be overwritten!', Warning)
        # run checks on data
        self._check_data(data)
        # filter only columns of interest
        self._portfolios = data.copy(deep=True).filter(items=self._keep_cols+self._factors).dropna(subset=self._keep_cols).melt(id_vars=self._keep_cols,value_vars=self._factors,var_name='factor', value_name='factor_raw_score')
        # convert to datetime column
        self._portfolios[self.date_col]=pd.to_datetime(self._portfolios[self._date_col])
   
    # form stock portfolios 
    def _process_data(self):
        if self._has_data:
            # preprocess factor scores
            self._preprocess_factors()
            # split factors into baskets
            self._basket_factors()
            # compute universe and basket weights
            self._compute_weights()
            # sort columns
            self._sort_portfolios()
    
    def _preprocess_factors(self):
        if 'factor_raw_score' not in self._portfolios:
            raise KeyError ('factor_raw_score is not in data columns.')
        
        self._portfolios['factor_score'] = self._portfolios.factor_raw_score
        
        # standardize
        self._portfolios['factor_score'] = self._portfolios.groupby(self._groupby_cols)['factor_score'].transform(lambda x: z_score(x, higher_better=self._higher_better))
    
    def _basket_factors(self):
        if 'factor_score' not in self._portfolios:
            raise KeyError ('factor_score is not in data columns.')
        # split into quantile baskets
        self._portfolios['factor_basket'] = self._portfolios.groupby(self._groupby_cols)['factor_score'].transform(lambda x: quantile_cut(x, q=self._quantiles, higher_better=self._higher_better))
    
    def _compute_weights(self):
        if 'factor_basket' not in self._portfolios:
            raise KeyError ('factor_basket is not in data columns.')
        # compute universe weight - default to be cap weighted nsn bmk return 
        self._portfolios['universe_weight']=self._portfolios.groupby([self._date_col, 'factor'])[self._bmk_col].transform(lambda x: x/sum(x))
        # compute basket weight - default equal weighted within each basket, nsn
        not_na_mask = self._portfolios.factor_basket.notnull()
        if self._weight_col is not None:
            self._portfolios.loc[not_na_mask, 'basket_weight'] = self._portfolios[not_na_mask].groupby(self._groupby_cols)[self._bmk_col].transform('sum')
            self._portfolios.loc[not_na_mask, 'basket_weight'] = self._portfolios.loc[not_na_mask, 'basket_weight']*self._portfolios[not_na_mask].groupby(self._groupby_cols+['factor_basket'])[self._weight_col].transform(lambda x: x/sum(x))
            self._portfolios.loc[not_na_mask, 'basket_weight'] = self._portfolios[not_na_mask].groupby([self._date_col,'factor','factor_basket'])['basket_weight'].transform(lambda x: x/sum(x))
            
        else:
            self._portfolios.loc[not_na_mask, 'basket_weight'] = self._portfolios[not_na_mask].groupby(self._groupby_cols)[self._bmk_col].transform('sum')
            self._portfolios.loc[not_na_mask, 'basket_weight'] = self._portfolios.loc[not_na_mask, 'basket_weight']*self._portfolios[not_na_mask].groupby(self._groupby_cols+['factor_basket'])[self._id_col].transform(lambda x: 1/len(x))
            self._portfolios.loc[not_na_mask, 'basket_weight'] = self._portfolios[not_na_mask].groupby([self._date_col,'factor','factor_basket'])['basket_weight'].transform(lambda x: x/sum(x))
        
    def _sort_portfolios(self):
        sorting_columns = self._groupby_cols+['factor_raw_score']
        self._portfolios = self._portfolios.sort_values(sorting_columns, ascending=[True]*(len(sorting_columns)-1)+[not self._higher_better]).reset_index(drop=True)
        
    def coverage(self, longdata=False):
        coverage=self._portfolios.groupby([self._date_col,'factor']).factor_score.apply(lambda x: x.count()/len(x)).to_frame('coverage')
        coverage = coverage.fillna(0).reset_index()
        if not longdata:
            coverage = coverage.set_index([self._date_col,'factor']).unstack().rename_axis(None)
        return coverage

