# Python 3.5.2 |Anaconda 4.2.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited: 2017-09-13
Author: Luke Shirley (L.Shirley@warwick.ac.uk)

Description: Methods and helper methods for generating synthetic time-series
with a change-point.
"""

import numpy as np


def covariance_matrix(a, b, dim):M
    """heuristically:
    float 0 < a < 1 (low->high correlation)
    float 0 < b < inf (low->high variance)
    int dim : number of dimensions
    """
    var = np.zeros(shape=(dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            var[i, j] = b*pow(a, np.abs(i - j))
    return var


def halfway_cp(T, dim, mean1, mean2, a1, a2, b1, b2, seed=100):
    """generates normally distributed data with a change-point halfway"""

    """generate covariance matrices"""
    var1 = covariance_matrix(a1, b1, dim)
    var2 = covariance_matrix(a2, b2, dim)

    """generate data"""
    np.random.seed(seed)
    data = np.random.normal(loc=0, scale=1, size=(T, dim))
    chol1 = np.linalg.cholesky(var1)
    chol2 = np.linalg.cholesky(var2)
    for i in range(0, T // 2):
        data[i, :] = np.matmul(chol1, data[i, :]) + mean1
        data[i + (T // 2), :] = np.matmul(chol2, data[i + (T // 2), :]) + mean2

    return data


def two_models(T):
    """generates data with a change-point in specific correlations"""

    var1 = covariance_matrix(0.99, 1, 2)
    chol1 = np.linalg.cholesky(var1)
    var2 = covariance_matrix(0.99, 1, 4)
    chol2 = np.linalg.cholesky(var2)

    np.random.seed(100)
    data = np.random.normal(loc=0, scale=1, size=(T, 4))

    for i in range(0, T//2):
        data[i, [0, 1]] = np.matmul(chol1, data[i, [0, 1]])
        data[i, [2, 3]] = np.matmul(chol1, data[i, [2, 3]])

        data[i + (T // 2), :] = np.matmul(chol2, data[i + (T // 2), :])

    return data


def line_tree(n):
    """default tree for HIW is a line"""

    tree = []
    for i in range(n-1):
        tree.append((i, i+1))
    return tree

# testing
if __name__ == "__main__":
    from r_distr_plot import r_data_plot

    mean1, mean2 = 0.0, 0.0
    a1, a2 = 0.01, 0.01
    b1, b2 = 0.25, 0.25

    """choose data structure parameters"""
    T = 200
    s1, s2 = 1, 1
    # tree = line_tree(s1)f

    """generate data"""
    data = halfway_cp(T, s1 * s2, mean1, mean2, a1, a2, b1, b2)
    for i in range(0, T // 2):
        data[i, :] = data[i, :] + i/50
        data[i + (T // 2), :] = data[i + (T // 2), :] + 2 - i/50
    r_data_plot(data, T, s1*s2, [T//2])
