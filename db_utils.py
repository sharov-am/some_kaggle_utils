#закинуть данные в базу (должны быть соотв. провайдеры) установлены
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://microsrv_usr:asddsa1@localhost:5432/microsrv')
df = pd.read_csv("d:\\kaggle\\alpha_battle\\alfabattle2_abattle_train_target.csv")
df.to_sql('train_target', engine, if_exists="append", chunksize=2000000)


def fit_xgb(trial, xtr, ytr, xval, yval):
    # params = {
    #     "n_estimators": trial.suggest_categorical("n_estimators", [150, 200, 250, 300]),
    #     "subsample": trial.suggest_discrete_uniform("subsample", 0.6,1,0.1),
    #     "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.6,1,0.1),
    #     "eta": trial.suggest_loguniform("eta",1e-2,0.1),
    #     # "gamma": trial.suggest_loguniform("gamma",0.05,1),
    #     "max_depth": trial.suggest_categorical("max_depth",[5,7,8,9]),
    #     "min_child_weight": trial.suggest_int("min_child_weight",5,11),
    #     "random_state": 21
    # }

    model = xgb.XGBRegressor(**params)
    model.fit(xtr, ytr.reshape(-1,))

    y_val_pred = model.predict(xval)

    log = {
        "train rmse": np.sqrt(mean_squared_error(ytr, model.predict(xtr))), # setting squared=False returns root_mean_squared_error
        "valid rmse": np.sqrt(mean_squared_error(yval, y_val_pred))  # setting squared=False returns root_mean_squared_error
    }

    return model, log