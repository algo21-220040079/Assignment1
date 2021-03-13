from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
# import shap
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False





def modelEval(ldf, feature='signal', split_id=0.2, eval_id=[True, True, True, True],
              n_fold=5, scoring='accuracy', cv_yrange=None, hm_vvals=[0.5, 1.0, 0.75]):
    ''' Split Train/Evaluation <DataFrame> Set Split '''

    # split_id : Train/Test split [%,timestamp], whichever is not None
    # test_id : Evaluate trained model on test set only

    models = []
    # 轻量级模型
    # 线性监督模型
    models.append(('LR', LogisticRegression(n_jobs=-1)))
    models.append(('TREE', DecisionTreeClassifier()))
    # 非监督模型
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('NB', GaussianNB()))
    # 高级模型
    models.append(('GBM', GradientBoostingClassifier(n_estimators=25)))
    models.append(('XGB', XGBClassifier(n_estimators=25, use_label_encoder=False)))
    models.append(('RF', RandomForestClassifier(n_estimators=25)))

    train_df, eval_df = train_test_split(ldf, test_size=split_id, shuffle=False)


    ''' Train/Test Feature Matrices + Target Variables Split'''
    y_train = train_df[feature]
    X_train = train_df.loc[:, train_df.columns != feature]
    y_eval = eval_df[feature]
    X_eval = eval_df.loc[:, eval_df.columns != feature]
    X_one = pd.concat([X_train, X_eval], axis=0)
    y_one = pd.concat([y_train, y_eval], axis=0)

    ''' Cross Validation, Training/Evaluation, one evaluation'''
    lst_res = []
    names = []
    lst_train = []
    lst_eval = []
    lst_one = []
    lst_res_mean = []
    if (any(eval_id)):
        for name, model in models:
            names.append(name)

            # Cross Validation Model on Training Se
            if (eval_id[0]):
                kfold = KFold(n_splits=n_fold, shuffle=True)
                cv_res = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
                lst_res.append(cv_res)

            # Evaluate Fit Model on Training Data
            if (eval_id[1]):
                res = model.fit(X_train, y_train)
                train_res = accuracy_score(res.predict(X_train), y_train)
                lst_train.append(train_res)
            if (eval_id[2]):
                if (eval_id[1] is False):  # If training hasn't been called yet
                    res = model.fit(X_train, y_train)
                eval_res = accuracy_score(res.predict(X_eval), y_eval)
                lst_eval.append(eval_res)

            # Evaluate model on entire dataset
            if (eval_id[3]):
                res = model.fit(X_one, y_one)
                one_res = accuracy_score(res.predict(X_one), y_one)
                lst_one.append(one_res)

            ''' [out] Verbal Outputs '''
            lst_res_mean.append(cv_res.mean())
            fn1 = cv_res.mean()
            fn2 = cv_res.std()
            fn3 = train_res
            fn4 = eval_res
            fn5 = one_res

    s0 = pd.Series(np.array(lst_res_mean), index=names)
    s1 = pd.Series(np.array(lst_train), index=names)
    s2 = pd.Series(np.array(lst_eval), index=names)
    s3 = pd.Series(np.array(lst_one), index=names)
    pdf = pd.concat([s0, s1, s2, s3], axis=1)
    pdf.columns = ['cv_average', 'train', 'test', 'all']

    ''' Visual Ouputs '''
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].set_title(f'{n_fold} Cross Validation Results')
    sns.boxplot(data=lst_res, ax=ax[0], orient="v", width=0.3)
    ax[0].set_xticklabels(names)
    sns.stripplot(data=lst_res, ax=ax[0], orient='v', color=".3", linewidth=1)
    ax[0].set_xticklabels(names)
    ax[0].xaxis.grid(True)
    ax[0].set(xlabel="")
    if (cv_yrange is not None):
        ax[0].set_ylim(cv_yrange)
    sns.despine(trim=True, left=True)
    sns.heatmap(pdf, vmin=hm_vvals[0], vmax=hm_vvals[1], center=hm_vvals[2],
                ax=ax[1], square=False, lw=2, annot=True, fmt='.3f', cmap='Blues')
    ax[1].set_title('Accuracy Scores')
    plt.show()