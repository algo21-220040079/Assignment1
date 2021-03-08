import numpy as np
import pandas as pd
import tushare as ts
import talib as ta

def get_data(start, end):
    df = ts.get_k_data('sh', start=start,end=end)
    df.index = pd.to_datetime(df.date)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df

def add_features(data):
    data['MA13']=ta.MA(data.close,timeperiod=13)
    data['MA34']=ta.MA(data.close,timeperiod=34)
    data['MA89']=ta.MA(data.close,timeperiod=89)
    data['EMA10']=ta.EMA(data.close,timeperiod=10)
    data['EMA30']=ta.EMA(data.close,timeperiod=30)
    data['EMA200']=ta.EMA(data.close,timeperiod=200)
    data['MOM10']=ta.MOM(data.close,timeperiod=10)
    data['MOM30']=ta.MOM(data.close,timeperiod=30)
    data['RSI10']=ta.RSI(data.close,timeperiod=10)
    data['RSI30']=ta.RSI(data.close,timeperiod=30)
    data['RS200']=ta.RSI(data.close,timeperiod=200)
    data['K10'],data['D10']=ta.STOCH(data.high,data.low,data.close, fastk_period=10)
    data['K30'],data['D30']=ta.STOCH(data.high,data.low,data.close, fastk_period=30)
    data['K20'],data['D200']=ta.STOCH(data.high,data.low,data.close, fastk_period=200)