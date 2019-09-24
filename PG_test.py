# Python 3.5.2 |Anaconda 4.2.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited: 2017-09-13
Author: Yannis Zachos (i.zachos@warwick.ac.uk)

Description: Script to run change-point detection on generated or imported
data.
"""

import numpy as np
from cp_probability_model import CpModel
from detector import Detector
from r_distr_plot import r_data_plot, r_distr_plot
from poisson_gamma_model import PGModel
from Evaluation_tool import EvaluationTool

def test():

    #filepath = '../data/'
    filepath = './combined_df.csv'

    """Artificial datasets"""
    #filepath += 'univariate_linear.csv'
    #filepath += 'PS_len10_stretch1.csv'
    # filepath += 'PS_len20_stretch1.csv'
    # filepath += 'PS_len30_stretch1.csv'
    # filepath += 'PS_len10_stretch2.csv'
    # filepath += 'PS_len10_stretch3.csv'
    # filepath += 'PS_len10_stretch0p5.csv'
    # filepath += 'PS_len10_stretch0p33.csv'
    #filepath += 'PS_linear.csv'
    #filepath += 'PS_cyclic.csv'
    #filepath += 'PS_combined.csv'

    """Property Transaction"""
    #filepath += 'property_transactions_monthly_england_scale1000.csv'

    """Dow Jones Volume"""
    #filepath += 'MSFT_dowjones_volume_scale1000000.csv'

    """Cervest"""
    #filepath += 'ecmwf_weekly_tp_percentile065_count_grid0.csv'
    #filepath += 'ecmwf_weekly_tcc_geq_thresh70_count_grid0.csv'
    # filepath += 'ecmwf_weekly_st_geq_percentile065_count_grid0.csv'
    # filepath += 'ecmwf_weekly_sm_geq_percentile065_count_grid1.csv'
    # filepath += 'ecmwf_weekly_t2m_geq_percentile065_count_grid0.csv'
    # filepath += 'ecmwf_monthly_tp_percentile065_count_grid0.csv'
    # filepath += 'ecmwf_monthly_tcc_geq_thresh70_count_grid0.csv'
    # filepath += 'ecmwf_monthly_st_geq_percentile065_count_grid0.csv'
    #filepath += 'ecmwf_monthly_sm_geq_percentile065_count_grid1.csv'
    # filepath += 'ecmwf_monthly_t2m_geq_percentile065_count_grid0.csv'

    #filepath += 'mPS.csv'
    #filepath += 'mPS3.csv'
    #filepath += 'test.csv'

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

    """OPTIONAL: Plot the actual intensity"""
    try:
        """check if there is a real_intensity"""
        real_intesity = np.loadtxt(filepath.replace('.csv', '_real.csv'), dtype=np.float64, delimiter=",", skiprows=1, usecols=datacols)
    except:
        """otherwise don't plot anything"""
        real_intesity = None

    """choose prior parameters"""
    # Parameters common across models
    prior_hazard = 30
    pruning_threshold = T + 1

    # Parameter for PG model
    prior_alpha = 1 #1 #150 #8000
    prior_beta = 1 #1 #6 #20
    prior_sums = np.zeros(s1*s2)

    """create hazard model object"""
    cp_model = CpModel(prior_hazard)

    """create model object(s)"""
    pg_model = PGModel(
         prior_alpha,
         prior_beta,
         prior_sums,
         s1,
         s2,
         auto_prior_update=False,
    )

    """create Detector object"""
    detector = Detector(
        data,
        np.array([pg_model]),
        np.array([1]),
        cp_model,
        s1,
        s2,
        T,
        threshold=pruning_threshold,
        trim_type = "keep_K",
        store_mrl = True,
        store_rl = True
    )

    j = 0
    for t in range(0, T):
        detector.next_run(data[t, :], t+1)
        # print('t:',t)
        if t >= j*T//100:
            print(j, "% Complete")
            j += 1

    print("Done. Plotting...")

    """Plot CPs and run lengths"""
    r_distr_plot(
        data,
        T,
        detector,
        s1*s2,
        dateindex=float(0),
        dateincr=float(1),
        #title="Monthly 2 Meter Temperature ($\geq$65%), Location (50.5, -5.0)", #(50.5, -5.0) (50.5, -4.5)
        #ylabel="No. of counts",
        #xlabel="Time in months"
    )

if __name__ == "__main__":
    test()
