import xgboost as xgb
def objective(trial):

    """
    gamma [default=0, alias: min_split_loss]

    A node is split only when the resulting split gives a positive reduction in the loss function.
    Gamma specifies the minimum loss reduction required to make a split.
    It makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
    The larger gamma is, the more conservative the algorithm will be.
    Range: [0,∞]
    """

    param = {
        #'tree_method':'gpu_hist',
        "objective":trial.suggest_categorical("objective",['reg:squarederror']),
        'eval_metric':trial.suggest_categorical('eval_metric',['rmsle']),

        #L1\L2 regularization

        # L2 regularization term on weights (analogous to Ridge regression).
        # This is used to handle the regularization part of XGBoost.
        # Increasing this value will make model more conservative.
        #STATQUEST NOTE: intended to reduce the prediction sensitivity to individual
        #observations. Amount of similiarity score inversely propotional to number of
        #preds in the node.
        'lambda': trial.suggest_float('lambda', 1, 10),

        # L1 regularization term on weights (analogous to Lasso regression).
        # It can be used in case of very high dimensionality so that the algorithm runs faster when implemented.
        # Increasing this value will make model more conservative.
        'alpha': trial.suggest_float('alpha', 1, 10),


        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6,0.7,0.8,0.9, 1.0]),

        #subsample [default=1]

        #It denotes the fraction of observations to be randomly samples for each tree.
        #Subsample ratio of the training instances.
        #Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. - This will prevent overfitting.
        #Subsampling will occur once in every boosting iteration.
        #Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
        #Typical values: 0.5-1
        #range: (0,1]

        'subsample': trial.suggest_categorical('subsample', [0.5,0.6,0.7,0.8,1.0]),
        'n_estimators': trial.suggest_categorical('n_estimators', [100,500,1000,5000]),
        'max_depth': trial.suggest_int('max_depth', 3,7),

        #min_child_weight [default=1]

        #It defines the minimum sum of weights of all observations required in a child.
        #This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
        #It is used to control over-fitting.
        #Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
        #Too high values can lead to under-fitting.
        #Hence, it should be tuned using CV.
        #The larger min_child_weight is, the more conservative the algorithm will be.
        #range: [0,∞]
        'min_child_weight': trial.suggest_int('min_child_weight', 30, 300),

        #It is analogous to learning rate in GBM.
        #It is the step size shrinkage used in update to prevent overfitting.
        #After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
        #It makes the model more robust by shrinking the weights on each step.
        #ange : [0,1]
        #Typical final values : 0.01-0.2.

        'eta' : trial.suggest_float('eta' , 0.01,.2),

        'verbosity':trial.suggest_categorical('verbosity',[0]),
        'random_state':trial.suggest_categorical('random_state', [37]),
        #'num_class':trial.suggest_categorical('num_class', [6]),
        'use_label_encoder':False,
        'early_stopping_rounds':trial.suggest_categorical('early_stopping_rounds',[100]),
    }

    kf = KFold(n_splits = 5,random_state = 37,shuffle=True)
    mse_log_avg_val = []
    mse_log_avg_train = []
    for train_index, test_index in kf.split(X, y):
        train_x, test_x = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y[train_index], y[test_index]

        model = xgb.XGBRegressor(**param)
        model.fit(train_x,train_y,eval_set=[(test_x,test_y)],verbose=False)

        y_pred = model.predict(test_x)
        log_loss_score = mean_squared_log_error(test_y, y_pred, squared=False)
        mse_log_avg_val.append(log_loss_score)

        y_pred = model.predict(train_x)
        log_loss_score = mean_squared_log_error(train_y, y_pred, squared=False)
        mse_log_avg_train.append(log_loss_score)

    print(np.mean(mse_log_avg_train))
    return np.mean(mse_log_avg_val)



optuna.logging.set_verbosity(optuna.logging.INFO) # i do not want to see trail information
xgb_study = optuna.create_study(direction ='minimize', study_name ='xgb')
xgb_study.optimize(objective, n_trials = 5)
print('numbers of the finished trials:', len(xgb_study.trials))
print('the best params:', xgb_study.best_trial.params)
print('the best value:', xgb_study.best_value)