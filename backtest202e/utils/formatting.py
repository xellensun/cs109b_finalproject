import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

def format_summary(data):
    row_formatters = {'Avg_Return':lambda x: f'{x/100:.2%}',
                      'CAGR':lambda x: f'{x:.2%}',
                      'Avg_Vol':lambda x: f'{x/100:.2%}',
                      'Alpha':lambda x: f'{x/100:.2%}',
                      'Tracking_Error':lambda x: f'{x/100:.2%}',
                      'IR':lambda x: f'{x:0.2f}',
                      'T_stat':lambda x: f'{x:.2f}',
                      'Consistency':lambda x: f'{x:.2%}',
                      'Max_Drawdown':lambda x: f'{x/100:.2%}',
                      'IC':lambda x: f'{x:.2%}',
                      'Persistency':lambda x: f'{x:.2%}',
                      'Avg_Turnover':lambda x: f'{x:.2%}',
                      'Avg_Hit_Rate':lambda x: f'{x:.2%}',
                      'Periods':lambda x: f'{x:.0f}'}
    
    for row, formatter in row_formatters.items():
        if row in data.index:
            data.loc[row] = data.loc[row].apply(formatter)
    
    return data


def line_plot(data):
    register_matplotlib_converters()
    #data *=100
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots()
        fig.set_size_inches([14,6])
        ax=sns.lineplot(data=data,palette='Set1',dashes=False,linewidth=2,ax=ax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.set_minor_formatter(mtick.PercentFormatter())
        ax.set_xlim(data.index.min(), data.index.max())
        lgnd = ax.legend(loc=0, edgecolor='white')
    plt.show()
    
    
def datetime_index_to_date_index(data):
    if isinstance(data.index, pd.core.indexes.datetimes.DatetimeIndex):
        data.index=data.index.date
    return data