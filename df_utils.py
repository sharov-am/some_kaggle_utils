import pandas as pd
from pandas import DataFrame
from scipy.stats import stats
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt

def base_data_info(df):
    '''Shows missing, unique and so on data...'''
    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(
            stats.entropy(df[name].value_counts(normalize=True), base=2), 2)

    return summary



# https://stackoverflow.com/a/46740476/241446
def remove_outlier(df, col_name, q=0.25):
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(1 - q)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
    return df_out


def remove_outlier_percentile(df, col_name, high_perc=99.9):
    h = np.percentile(df[col_name], high_perc)
    l = np.percentile(df[col_name], 1 - high_perc)
    df_out = df.loc[(df[col_name] > l) & (df[col_name] < h)]
    return df_out


def apply_fn(df, col_target, fn, col1, col2):
    df[col_target] = fn(df[col1], df[col2])  # df.apply(lambda x: fn(df[col1], df[col2]), axis=1)
    return df


def generate_features(df, cols, fn):
    col_len = len(cols)
    for i in range(0, col_len):  # exclude id column
        for j in range(i + 1, col_len):
            target = str(cols[i]) + "+" + str(cols[j])
            df = apply_fn(df, target, fn, cols[i], cols[j])
    return df


def get_feature_cols(train):
    return [col for col in train.columns.tolist() if col not in ['id', 'target']]



def create_folds(df, n_s=5, n_grp=None):
    df['Fold'] = -1

    if n_grp is None:
        skf = KFold(n_splits=n_s)
        target = df.target
    else:
        skf = StratifiedKFold(n_splits=n_s)
        df['grp'] = pd.cut(df.target, n_grp, labels=False)
        target = df.grp

    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        df.loc[v, 'Fold'] = fold_no
    return df


def check_feature_for_normality(df: DataFrame, col_name: str, alpha: float):
    k2, p = stats.normaltest(df.loc[:, col_name])
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        return False, p
    else:
        return True, p


def describe_with_background_gradient(df: DataFrame):
    return  df.describe().T.style.background_gradient(cmap='Blues')


def plot_train_vs_test_hist(train: DataFrame, test: DataFrame, cols=None):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-deep')
    if cols is None:
        cols = test.columns
    for col in cols:
        col_in_train = col in train.columns
        col_in_test = col in test.columns
        if not (col_in_test and col_in_train):
            continue
        x = train[col]
        y = test[col]

        plt.hist([x, y], label=['train', 'test'], density = True)
        plt.title(col)
        plt.legend(loc='upper right')
        plt.show()

def plot_train_vs_test_kde(train: DataFrame, test: DataFrame, cols=None):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-deep')
    if cols is None:
        cols = test.columns
    for col in cols:
        col_in_train = col in train.columns
        col_in_test = col in test.columns
        if not (col_in_test and col_in_train):
            continue
        x = train[col]
        y = test[col]
        sns.kdeplot(data=x, color='crimson', label='train', fill=True)
        sns.kdeplot(data=y, color='limegreen', label='test', fill=True)

        plt.legend(loc='upper right')
        plt.show()

def plot_train_vs_test_origin_kde(train: DataFrame, test: DataFrame, origin:DataFrame, cols=None):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-deep')
    if cols is None:
        cols = test.columns
    for col in cols:
        col_in_train = col in train.columns
        col_in_test = col in test.columns
        if not (col_in_test and col_in_train):
            continue
        x = train[col]
        y = test[col]
        o = origin[col]
        sns.kdeplot(data=x, color='crimson', label='train', fill=True)
        sns.kdeplot(data=y, color='limegreen', label='test', fill=True)
        sns.kdeplot(data=o, color='blue', label='orgin', fill=True)

        plt.legend(loc='upper right')
        plt.show()


#https://gist.github.com/wjptak/88575bbc5dde446e1186ffd41475c0f1
def print_highly_correlated(df:DataFrame, features=None, threshold=0.5):
    """Prints highly correlated features pairs in the data frame (helpful for feature engineering)"""
    if features is None:
        features = list(df.columns)
    corr_df = df[features].corr() # get correlations
    correlated_features = np.where(np.abs(corr_df) > threshold) # select ones above the abs threshold
    correlated_features = [(corr_df.iloc[x,y], x, y) for x, y in zip(*correlated_features) if x != y and x < y] # avoid duplication
    s_corr_list = sorted(correlated_features, key=lambda x: -abs(x[0])) # sort by correlation value

    if s_corr_list == []:
        print("There are no highly correlated features with correlation above", threshold)
    else:
        for v, i, j in s_corr_list:
            cols = df[features].columns
            print ("%s and %s = %.3f" % (corr_df.index[i], corr_df.columns[j], v))



def get_common_rows(df1,df2):
    return df1.merge(df2, how='inner', indicator=False)


def plot_advanced(df:DataFrame):
  def plot_col(df:DataFrame,variable):
     if df[variable].dtype != object:
        # define figure size
        fig, ax = plt.subplots(1, 5, figsize=(24, 4))

        # histogram
        sns.histplot(df[variable], bins=30, kde=True, ax=ax[0])
        ax[0].set_title('Histogram')

        # KDE plot
        sns.kdeplot(df[variable], ax=ax[1])
        ax[1].set_title('KDE Plot')

        # boxplot
        sns.boxplot(y=df[variable], ax=ax[3])
        ax[3].set_title('Boxplot')

        # scatterplot
        sns.scatterplot(x=df.index, y=df[variable], ax=ax[4])
        ax[4].set_title('Scatterplot')

        plt.tight_layout()
        plt.show()
  for col in df.columns:
      plot_col(df ,col)

def plot_numerical_data_hue_target(df:DataFrame,X,hue):
    fig, axes =plt.subplots(1, 2, figsize = (15,4))
    sns.histplot(ax = axes[0], x=X, hue=hue, data = df, element="step",kde=True)
    sns.boxplot(ax = axes[1], x=hue, y=X, hue=hue,data=df)

def get_pseudo_combinations(df:DataFrame,columns, tr=None, threshhold:int = 5):
    """Returns all combinations of columns
        Args:
            columns: array of column names
            tr: (int) num of columns. If None, default all columns.

        Returns:
            all_combs: (list of lists) all possible column combinations.
    """
    from itertools import combinations
    n_comb = len(columns)
    if tr:
        n_comb = len(columns[:tr])
    all_combs = []
    for i in range(13, n_comb+1):
        all_combs += list(map(list, combinations(columns, r=i)))

    for cols in all_combs:
        s = df[cols].duplicated().sum()
        if s > threshhold:
             print(f'{s}{blk}, {cols}')