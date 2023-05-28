import numpy as np
from pandas import DataFrame as pd, DataFrame
from sklearn.preprocessing import LabelEncoder


def label_encoder(train, X: DataFrame):
    label = LabelEncoder()
    train = pd.DataFrame()
    for c in X.columns:
        if X[c].dtype == 'object':
            train[c] = label.fit_transform(X[c])
        else:
            train[c] = X[c]
    return train


def cyclic_encoder(X: DataFrame, columns):
    X_train_cyclic = X.copy()
    columns = ['day', 'month']
    for col in columns:
        X_train_cyclic[col + '_sin'] = np.sin((2 * np.pi * X_train_cyclic[col]) / max(X_train_cyclic[col]))
        X_train_cyclic[col + '_cos'] = np.cos((2 * np.pi * X_train_cyclic[col]) / max(X_train_cyclic[col]))
    X_train_cyclic = X_train_cyclic.drop(columns, axis=1)
    return X_train_cyclic


def target_encoding(df_train):
    X_target = df_train.copy()
    for col in X_target.columns:
        if X_target[col].dtype == 'object':
            target = dict(X_target.groupby(col)['target'].agg('sum') / X_target.groupby(col)['target'].agg('count'))
            X_target[col] = X_target[col].replace(target).values
    return X_target
