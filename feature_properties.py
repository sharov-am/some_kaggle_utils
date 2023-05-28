# from scipy.stats import f_oneway
#
# def anova(f,y):
#     return f_oneway(*[y[f==x] for x in f.unique()])
#
# int_cols = [F'f_{i:02d}' for i in np.arange(7)+7]
# float_cols = [F'f_{i:02d}' for i in list(range(0,7))+list(range(14,29))]
# pvalues = []
# for row in int_cols:
#     row_arr = []
#     for col in float_cols:
#         _,p = anova(data[row],data[col])
#         row_arr.append(p)
#     pvalues.append(row_arr)
#
# pd.DataFrame(pvalues,index=int_cols,columns=float_cols).style
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_selection import mutual_info_classif


def plot_feature_importance(importance,names,model_type):
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

#Usage
#plot the xgboost result
#plot_feature_importance(xgb_model.feature_importances_,train.columns,'XG BOOST')

#plot the catboost result
#plot_feature_importance(cb_model.get_feature_importance(),train.columns,'CATBOOST')


def calc_mi(df:DataFrame, target:str):
    num_iter = 10
    results = []
    for i in range(num_iter):
        results.append(mutual_info_classif(df.drop([target],axis=1),df[target]))
    mi = zip(np.mean(results,axis=0), np.std(results,axis=0))
    data = [mu for mu,sigma in mi]
    columns=[col for col in df.columns if col != target]
    temp = pd.DataFrame( data = np.array([data, columns]).T, columns=['value', 'feature'])
    temp['value'] = temp['value'].astype(float)
    return  temp