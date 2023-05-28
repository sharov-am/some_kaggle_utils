import  matplotlib as plt
import pandas as pd
import seaborn as sns
from matplotlib import gridspec


def ploting_cat_fet(df, cols, vis_row=5, vis_col=2):
    grid = gridspec.GridSpec(vis_row, vis_col)  # The grid of chart
    plt.figure(figsize=(17, 35))  # size of figure

    # loop to get column and the count of plots
    for n, col in enumerate(df[cols]):
        tmp = pd.crosstab(df[col], df['target'], normalize='index') * 100
        tmp = tmp.reset_index()
        tmp.rename(columns={0: 'No', 1: 'Yes'}, inplace=True)

        ax = plt.subplot(grid[n])  # feeding the figure of grid
        sns.countplot(x=col, data=df, order=list(tmp[col].values), color='green')
        ax.set_ylabel('Count', fontsize=15)  # y axis label
        ax.set_title(f'{col} Distribution by Target', fontsize=18)  # title label
        ax.set_xlabel(f'{col} values', fontsize=15)  # x axis label

        # twinX - to build a second yaxis
        gt = ax.twinx()
        gt = sns.pointplot(x=col, y='Yes', data=tmp,
                           order=list(tmp[col].values),
                           color='black', legend=False)
        gt.set_ylim(tmp['Yes'].min() - 5, tmp['Yes'].max() * 1.1)
        gt.set_ylabel("Target %True(1)", fontsize=16)
        sizes = []  # Get highest values in y
        for p in ax.patches:  # loop to all objects
            height = p.get_height()
            sizes.append(height)
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 3,
                    '{:1.2f}%'.format(height / len(df) * 100),
                    ha="center", fontsize=14)
        ax.set_ylim(0, max(sizes) * 1.15)  # set y limit based on highest heights

    plt.subplots_adjust(hspace=0.5, wspace=.3)
    plt.show()


# fig, ax = plt.subplots(figsize=(17,8))
# sns.histplot(data=oof, x="prediction",bins = 100, ax=ax, kde = True)
# ax2 = ax.twinx()
# sns.boxplot(data=oof, x="prediction", ax=ax2,boxprops=dict(alpha=.7))
# ax2.set(ylim=(-.5, 10))
# plt.suptitle('Cost countplot and Boxplot for my train predictions', fontsize=20)
# ax.grid(True)
# plt.show()