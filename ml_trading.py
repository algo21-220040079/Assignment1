from MyData import MyData
from Signal import Signal
from Model import Model
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = MyData.get_data('2005-01-01','2021-03-01')
    df_train, df_test = df.loc[:'2017'], df.loc['2018':]
    df_tr1 = df_train.copy(deep=True)
    df_te1 = df_test.copy(deep=True)
    Signal.trade_signal(df_tr1)

    df_tr2 = df_tr1.copy(deep=True)
    df_te2 = df_te1.copy(deep=True)
    MyData.add_features(df_tr2)
    Signal.plot_corr_map(df_tr2,'signal',figsize=(15,0.5))

    df_tr1 = df_tr1.dropna()
    df_tr2 = df_tr2.dropna()
    df_te2 = df_te2.dropna()
    Model.modelEval(df_tr1)
    Model.modelEval(df_tr2, cv_yrange=(0.8, 1.0), hm_vvals=[0.8, 1.0, 0.9])



