from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def diag_corr(df, title):
    ''' Diagonal correlation plot from 
    '''

    sns.set(style="white")
    # Compute the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title(title)


    sns.set(style="white", context="talk")


def bars(x, y, ylabel, title):
    ''' Plot a niice barplot
    '''
    f, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(x=x, y=y, palette="rocket", ax=ax)
    ax.axhline(0, color="k", clip_on=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Finalize the plot
    sns.despine(bottom=True)

    # plt.setp(f.axes, yticks=[])
    plt.tight_layout(h_pad=2)

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    plt.show()

def lines(x, y, hue, title):
    ''' Line plot with custom line widths
    '''
    sns.set(style="white", context="talk")
    f, ax = plt.subplots(figsize=(10, 6))
    ax = sns.pointplot(x=x,
                       y=y,
                       hue=hue,
                       scale=.6,
                       errwidth=0.6)

    sns.despine(bottom=True)
    plt.title(title)
    ax.legend(bbox_to_anchor=(1.15, 1.05))
    plt.show()


def joint(x, y, title):
    sns.set(style="white", context="talk")
    f, ax = plt.subplots(figsize=(10, 6))

    ax = sns.scatterplot(x=x,
                   y=y)
    sns.despine(bottom=True)
    plt.title(title)

