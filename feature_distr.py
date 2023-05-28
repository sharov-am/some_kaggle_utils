import numpy as np
from matplotlib import pyplot as plt


def plot_df_hist(train=None, FEATURES=None):
    i = 0
    if FEATURES is None:
        FEATURES = train.columns
    fig, axs = plt.subplots(int(len(FEATURES)/4) +1, 4,figsize=(30,30))
    for f in FEATURES:
        current_ax = axs.flat[i]
        current_ax.hist(train[f], bins=100)
        current_ax.set_title(f)
        current_ax.grid()
        i = i + 1

def Q_Q_plot(values):
   from scipy import stats
   stats.probplot(values, dist='poisson', sparams=(np.mean(values),), plot=plt)
   plt.show()