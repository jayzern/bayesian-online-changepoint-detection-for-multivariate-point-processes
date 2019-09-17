# Python 3.5.2 |Anaconda 4.2.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited: 2017-09-13
Author: Luke Shirley (L.Shirley@warwick.ac.uk)

Description: Methods for plotting time-series data with run-length
distributions from a detector object.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from matplotlib.colors import LogNorm
#from matplotlib import rc
#rc('text', usetex=True)



EPS = pow(10, -5)

def visual_helper(distr):
    print(distr.shape)
    T1, T2 = distr.shape
    for i in range(T1):
        for j in range(T2):
            #jmax = j + min(T2//100, T2-j+1)
            # Set min to 50
            jmax = j + min(T2//50, T2-j+1)
            distr[i, j] = max(distr[i, j:jmax])
    return distr


def r_data_plot(
        data,
        T,
        dim,
        known_changepoints=(),
        dateindex=0,
        dateincr=1,
        title="Generated Data"
):
    """ plot generated data"""
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.xlim(dateindex-dateincr, dateindex+(T+1)*dateincr)
    plt.ticklabel_format(style="plain", axis="both")
    for x in known_changepoints:
        plt.axvline(x=x, color='b', linestyle='--', label="change-point")
    for i in range(dim):
        ax1.plot(np.linspace(start=dateindex, stop=dateindex+T*dateincr, num=T), data[:T, i])
    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()


def r_distr_plot(
        data,
        T,
        detector,
        dim,
        known_changepoints=(),
        dateindex=0,
        dateincr=1,
        title="",
        ylabel="Counts",
        xlabel="Time"
):
    """ plot generated data and r-distr"""

    model_universe_size = detector.Q

    if model_universe_size == 1:
        """without model selection"""
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(dateindex - dateincr, dateindex + (T + 1) * dateincr)
        plt.ticklabel_format(style="plain", axis="both")
        for x in known_changepoints:
            plt.axvline(x=x, color='b', linestyle='--', label="CPs")
        ax2 = fig.add_subplot(212)
        for i in range(dim):
            ax1.plot(np.linspace(start=dateindex, stop=dateindex + T * dateincr, num=T), data[:T, i], linestyle='-', marker='.' ,label="count", linewidth=0.5) #linestyle='-', marker='.'

    elif model_universe_size > 1:
        """WITH model selection"""
        gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 3])
        fig = plt.figure()
        ax1 = plt.subplot(gs[0])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(dateindex - dateincr, dateindex + (T + 1) * dateincr)
        plt.ticklabel_format(style="plain", axis="both")
        for x in known_changepoints:
            plt.axvline(x=x, color='b', linestyle='--', label="CPs")
        # Model selection
        ax3 = plt.subplot(gs[1])
        model_posterior_lgcp = []
        model_posterior_pg = []
        for i in range(0, len(detector.storage_model_and_run_length_log_distr)):
            model_posterior_lgcp.append(np.sum(np.exp(detector.storage_model_and_run_length_log_distr[i][0])))
            model_posterior_pg.append(np.sum(np.exp(detector.storage_model_and_run_length_log_distr[i][1])))
        ax3.plot(model_posterior_lgcp, 'g-', label="model 1")
        ax3.plot(model_posterior_pg, 'c-', label="model 2")
        plt.ylabel("p(m|y)")
        ax3.legend(loc='upper left')
        ax3.xaxis.set_visible(False)

        ax2 = plt.subplot(gs[2])
        for i in range(dim):
            ax1.plot(np.linspace(start=dateindex, stop=dateindex + T * dateincr, num=T), data[:T, i], linestyle='-', marker='.' ,label="count")

    """For good visualisation """
    storage_run_length_distr = np.zeros(shape=(detector.T+1, detector.T+1))
    for i in range(0,detector.T):
        x = i + 1
        y = range(0,detector.storage_run_length_log_distr[i].shape[0])
        storage_run_length_distr[x][y] = np.exp(detector.storage_run_length_log_distr[i])

    r = visual_helper(storage_run_length_distr)

    """Plot run-length distr"""
    plt.title("Run-Length Distributions")
    plt.ylabel("Run-length")
    plt.xlim(-1, T + 1)
    plt.ylim(-T // 100, T)
    im = ax2.imshow(
        r.transpose(),
        cmap='gray_r',
        interpolation='nearest',
        norm=LogNorm(vmin=EPS, vmax=1),
        aspect='auto'
    )
    cb = plt.colorbar(im, use_gridspec=True, orientation='horizontal')#, shrink=0.5, aspect=40)
    cb.set_label("Key: gradient representing probability density of run-length distributions")

    """Plot the MAP change-points"""
    i = 0
    j = 0
    CP_array = []
    cps = np.zeros(shape=len(detector.CPs[-1]), dtype=int)
    for cp in cps:
        CP_array.append(detector.CPs[-1][j][0])
        j += 1

    """Always remove MAP at time 2 due to detector bug"""
    if 2 in CP_array:
        CP_array.remove(2)

    for cp in CP_array:
        cps[i] = cp
        if i == 0:
            plt.axvline(x=cp, color='r', linestyle='-', label="MAP CPs")
        else:
            plt.axvline(x=cp, color='r', linestyle='-')
        i = i + 1

    """Remove extra element at the now after we removed one MAP CP"""
    cps = np.delete(cps,-1)


    for x in known_changepoints:
        plt.axvline(x=T // 2, color='b', linestyle='--', label="CPs")
    plt.legend(loc=2)
    print(cps)
    print(len(cps))

    plt.tight_layout()
    plt.show()
