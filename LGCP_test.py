# Python 3.5.2 |Anaconda 4.2.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited:
Author: Jay Zern Ng (J.Ng.3@warwick.ac.uk)
Description: Script to run change-point detection on generated or imported
data.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import gpflow
import GPy
import pylab as pb

from lgcp_model import LGCPModel
from mlgcp_model import mLGCPModel
from poisson_gamma_model import PGModel

from cp_probability_model import CpModel
from detector import Detector
from r_distr_plot import r_data_plot, r_distr_plot


def test():

    filepath = './example_data/'
    filepath += 'toy_data.csv'

    with open(filepath) as f:
        colnames = f.readline().split(",")
        firstline = f.readline().split(",")

    firstdateindex = float(firstline[0])
    datacols = list(range(1, (len(colnames))))

    data = np.loadtxt(
        filepath,
        dtype=np.float64,
        delimiter=",",
        skiprows=1,
        usecols=datacols
    )
    if len(data.shape) == 1:
        data = data.reshape((data.shape[0], 1))
    T, s1 = data.shape
    s2 = 1

    """Set prior hazard and pruning threshold"""
    prior_hazard = 30
    pruning_threshold = T + 1

    """create hazard model object"""
    cp_model = CpModel(prior_hazard)

    """create model object(s)"""
    lgcp_model = LGCPModel(
        prior_signal_variance=1,
        prior_lengthscale=1,
        custom_kernel=None,
        inference_method='laplace', #'laplace', 'variational_inference', 'sparse_variational_inference',
        refresh_rate=8,
        M_pseudo_input_size=10,
        S1=s1,
        S2=s2,
        auto_prior_update=True, #put false for now True
    )

    mlgcp_model = mLGCPModel(
        prior_signal_variance=1,
        prior_lengthscale=1,
        custom_kernel=None,
        inference_method='laplace',
        refresh_rate=5,
        M_pseudo_input_size=10,
        S1=s1,
        S2=s2,
        auto_prior_update=True,
    )

    """Single model"""
    detector = Detector(
        data,
        np.array([lgcp_model]),
        np.array([1]),
        cp_model,
        s1,
        s2,
        T,
        threshold = pruning_threshold, # None for now, pruning_threshold
        trim_type = "keep_K",
        store_mrl = True,
        store_rl = True
    )

    """Multiple models"""
    # detector = Detector(
    #     data,
    #     np.array([lgcp_model, pg_model]),
    #     np.array([1/2,1/2]),
    #     cp_model,
    #     s1,
    #     s2,
    #     T,
    #     threshold = pruning_threshold,
    #     trim_type = "keep_K",
    #     store_mrl = True,
    #     store_rl = True
    # )

    """Measure computation time"""
    start = time.time()

    """run MVBOCPDwMS"""
    print("Running CP detection...")
    j = 0
    for t in range(0, T):
        detector.next_run(data[t, :], t+1)
        if t >= j*T//100:
            print(j, "% Complete")
            j += 1

    print("Done. Plotting...")

    end = time.time()
    print("Execution time (seconds): ", end - start)

    """Plot CPs and run lengths"""
    r_distr_plot(
        data,
        T,
        detector,
        s1*s2,
        dateindex=float(0),
        dateincr=float(1),
        #title="",
        #ylabel="",
        #xlabel=""
    )

if __name__ == "__main__":
    test()
