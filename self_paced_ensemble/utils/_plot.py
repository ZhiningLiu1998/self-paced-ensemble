"""Utilities for data visualization."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

import pandas as pd
import numpy as np

from copy import copy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import KernelPCA

DEFAULT_VIS_KWARGS = {
    'cmap': plt.cm.rainbow,
    'edgecolor': 'black',
    'alpha': 0.6,
}

def set_ax_border(ax, border_color='black', border_width=2):
    for _, spine in ax.spines.items():
        spine.set_color(border_color)
        spine.set_linewidth(border_width)

    return ax

def plot_scatter(X, y, ax=None, weights=None, title='',
                 projection=None, vis_params=None):
    if ax is None:
        ax = plt.axes()
    X_vis = projection.transform(X) if X.shape[1] > 2 else X
    title += ' (2D projection by {})'.format(
        str(projection.__class__).split('.')[-1][:-2]
    )
    size = 50 if weights is None else weights
    if np.unique(y).shape[0] > 2:
        vis_params['palette'] = plt.cm.rainbow
    sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1],
        hue=y, style=y, s=size, **vis_params, legend='full', ax=ax)

    ax.set_title(title)
    ax = set_ax_border(ax, border_color='black', border_width=2)
    ax.grid(color='black', linestyle='-.', alpha=0.5)

    return ax

def plot_class_distribution(y, ax=None, title='',
                            sort_values=False, plot_average=True):
    count = pd.DataFrame(list(Counter(y).items()),
                         columns=['Class', 'Frequency'])
    if sort_values:
        count = count.sort_values(by='Frequency', ascending=False)
    if ax is None:
        ax = plt.axes()
    count.plot.bar(x='Class', y='Frequency', title=title, ax=ax)

    ax.set_title(title)
    ax = set_ax_border(ax, border_color='black', border_width=2)
    ax.grid(color='black', linestyle='-.', alpha=0.5, axis='y')

    if plot_average:
        ax.axhline(y=count['Frequency'].mean(),ls="dashdot",c="red")
        xlim_min, xlim_max, ylim_min, ylim_max = ax.axis()
        ax.text(
            x=xlim_min+(xlim_max-xlim_min)*0.82,
            y=count['Frequency'].mean()+(ylim_max-ylim_min)*0.03,
            c="red",s='Average')

    return ax

def plot_2Dprojection_and_cardinality(X, y, figsize=(10, 4), vis_params=None,
                                     projection=None, weights=None, plot_average=True,
                                     title1='Dataset', title2='Class Distribution'):

    if vis_params is None:
        vis_params = copy(DEFAULT_VIS_KWARGS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    if projection == None:
        projection = KernelPCA(n_components=2).fit(X, y)
    ax1 = plot_scatter(X, y, ax=ax1, weights=weights, title=title1,
                    projection=projection, vis_params=vis_params)
    ax2 = plot_class_distribution(y, ax=ax2, title=title2,
                    sort_values=True, plot_average=plot_average)
    plt.tight_layout()
    return fig