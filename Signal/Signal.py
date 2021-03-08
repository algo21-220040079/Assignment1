import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False


def trade_signal(data,short=10,long=60,tr_id=False):
    data['SMA1'] = data.close.rolling(short).mean()
    data['SMA2'] = data.close.rolling(long).mean() 
    data['signal'] = np.where(data['SMA1'] >data['SMA2'], 1.0, 0.0) 
    if(tr_id is not True):
        print(data['signal'].value_counts())
    # signal_plot(data)
    data = data.drop(['SMA1', 'SMA2'], axis=1)
    plot_corr_map(data,'signal',figsize=(7,1))

def signal_plot(data):
    plt.figure(figsize=(14, 12), dpi=80)
    ax1 = plt.subplot(211)
    plt.plot(data.close, color='b')
    plt.title('上证指数走势', size=15)
    plt.xlabel('')
    ax2 = plt.subplot(212)
    plt.plot(data.signal, color='r')
    plt.title('交易信号', size=15)
    plt.xlabel('')
    plt.show()
    data[['SMA1', 'SMA2', 'signal']].iloc[-250:].plot(figsize=(14, 6), secondary_y=['signal'])
    plt.show()


def plot_corr_map(df,target='demand',figsize=(9,0.5)):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    corr_mat = df.corr().round(2)
    corr_mat = corr_mat.transpose()
    corr = corr_mat.loc[:, df.columns == target].transpose().copy()
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, vmin=-0.5, vmax=0.5, center=0,
                cmap=cmap, square=False, lw=2, annot=True, cbar=False)
    plt.title(f'Feature Correlation to {target}')
    plt.show()
