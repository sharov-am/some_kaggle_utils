import numpy as np
import shap
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance


class ModelExplainer:
    """
    Methods to create feature importance and validation metric plots
    """
    @staticmethod
    def features_plot(model, features, target):
        """
        Feature importance plots
        :param model: (catboost.core.CatBoostRegressor, dict) model
        :param features: features data (X)
        :param target: target data (y)
        """
        if hasattr(model,"feature_importances_"):
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            fig = plt.figure(figsize=(18, 14))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), np.array(features.columns)[sorted_idx])
            plt.title('Feature Importance')

        perm_importance = permutation_importance(model, features, target, n_repeats=10, random_state=37)
        sorted_idx = perm_importance.importances_mean.argsort()
        fig = plt.figure(figsize=(18, 14))
        plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(features.columns)[sorted_idx])
        plt.title('Permutation Importance');

        return #буду отдельно рассматривать shap



#shap.summary_plot(shap_values, X.values,
#                  plot_type="bar",
#                  class_names= list(y.unique()),
#                  feature_names = X.columns)

#%%
