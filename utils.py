import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def show(features):
    """
    Show all columns of features
    :param features:
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', len(features.columns), 'display.width', 1000):
        print(features)


def errors(errors):
    """
    Show error rate
    :param errors: array of error rate
    """
    plt.plot([np.mean(errors[i]) for i in range(len(errors))])
    plt.show()


def scatter(features):
    """ Show scatter relations of matrix
    :param features: DataFrame
    :return:
    """
    scatter_matrix = pd.plotting.scatter_matrix(features)
    for ax in scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=0)
        ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=0)

    plt.show()


def histograms(df, n_cols):
    _row = len(df.columns) / n_cols
    row = _row if len(df.columns) % n_cols == 0 else _row + 1
    fig = plt.figure()
    for i, var_name in enumerate(df.columns):
        ax = fig.add_subplot(row, n_cols, i+1)
        df[var_name].hist(bins=10, ax=ax)
        ax.set_title(var_name)
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()
