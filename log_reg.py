from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def logistic(X, y, rand_state, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rand_state, test_size=test_size)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pre = lr.predict(X_test)
    return accuracy_score(y_test, y_pre)


def temp():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

    p = ['l1', 'l2']
    c = [0.0001, 0.001, 0.01, 0.1, 1]
    solver_map = {'l1': 'liblinear', 'l2': 'liblinear', 'elasticnet': 'saga'}
    for p_ in p:
        for c_ in c:
            acc_arr = []
            for train_index, test_index in skf.split(X_transformed, y):
                X_train, X_test = X_transformed[train_index], X_transformed[test_index]
                y_train, y_test = y[train_index], y[test_index]

                clf = LogisticRegression(penalty=p_, C=c_, max_iter=5000, solver='liblinear', multi_class="ovr",
                                         n_jobs=-1,
                                         random_state=21)
                clf.fit(X_train, y_train)

                train_predictions = clf.predict(X_test)
                acc = accuracy_score(y_test, train_predictions)
                acc_arr.append(acc)
            print("for penalty = {0}, C = {1}, max_acc = {2}".format(p_, c_, max(acc_arr)))
