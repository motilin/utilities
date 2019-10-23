import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
matplotlib.use('TkAgg')


def get_pcs(table, cols=None, n_components=None):
    '''
    plots the accumulated explained variance ratio in a PCA
    :param table: a count table in which columns are samples and genes are in the index
    :param cols: a list containing the columns of interest
    :param n_components: the total number of principal components to be calculated
    :return: a components table, as well as a combined weight given for each of the genes
    '''
    cols = cols or list(table.columns)
    n_components = n_components or len(cols)
    table = table[cols]
    gene_names = list(table.index)
    sc = StandardScaler()
    normed = sc.fit_transform(table.transpose())
    pca = PCA(n_components=n_components)
    pca.fit_transform(normed)
    components_df = pd.DataFrame(np.array(pca.components_).transpose(), index=gene_names)
    var = pca.explained_variance_ratio_
    cum_var = np.cumsum(var)
    for i, v in enumerate(cum_var):
        print(i + 1, v)
    plt.plot(cum_var, marker='o')
    ax = plt.gca()
    ax.plot(var, marker='o')
    ax.set_xticks(list(range(n_components)))
    ax.set_xticklabels(list(range(1, n_components + 1)))
    ax.set_xlabel('PC')
    ax.set_ylabel('Explained variance ratio')
    plt.show(block=True)
    gene_scores = components_df.mul(var).abs().sum(1).sort_values(ascending=False)
    return components_df, gene_scores


def plot_PCA(table, comps=None, n_components=None, cols=None):
    '''
    :param table: a count table in which the column names are samples and the index is genes
    :param comps: a list of int. represents the two components that should be plotted
    :param n_components: the total number of components that should be calculated
    :param cols: the columns that should be included in the PCA plot
    :return: plots a PCA plot for two principal components of choice
    '''
    comps = comps or [1, 2]
    compA = comps[0] - 1
    compB = comps[1] - 1
    cols = cols or list(table.columns)
    n_components = n_components or len(cols)
    sc = StandardScaler()
    table = table[cols]
    normed = sc.fit_transform(table)
    pca = PCA(n_components=n_components)
    pca.fit(normed)
    x = pca.components_[compA]
    y = pca.components_[compB]
    plt.scatter(x, y, cmap='seismic')
    ax = plt.gca()
    for i, point in enumerate(zip(x, y)):
        plt.text(point[0], point[1], cols[i], ha='center', va='center', size=10, rotation=30)
    ax.set_xlabel(f'PC{compA} ({pca.explained_variance_ratio_[compA]:.2%})')
    ax.set_ylabel(f'PC{compB} ({pca.explained_variance_ratio_[compB]:.2%})')
    plt.show(block=True)
