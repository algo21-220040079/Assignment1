from MyData import MyData
from Signal import Signal
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = MyData.get_data('2005-01-01','2021-03-01')
    df_train, df_test = df.loc[:'2017'], df.loc['2018':]
    df_tr1 = df_train.copy(deep=True)
    df_te1 = df_test.copy(deep=True)
    Signal.trade_signal(df_tr1)

    df_tr2 = df_tr1.copy(deep=True)
    MyData.add_features(df_tr2)
    Signal.plot_corr_map(df_tr2,'signal',figsize=(15,0.5))



