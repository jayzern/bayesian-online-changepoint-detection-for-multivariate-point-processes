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
#from OLDdetector import OLDdetector
from r_distr_plot import r_data_plot, r_distr_plot
from poisson_gamma_model import PGModel
#from OLDmultinomial_dirichlet_model import OLDMDGPModel
from multinomial_dirichlet_model import MDGPModel
#from Evaluation_tool import EvaluationTool

def test():

    filepath = '../data/'

    #filepath += 'chicago.txt'
    #filepath += 'bit_1year.txt'
    #filepath += 'bit_all.txt'
    #filepath += 'crypto.txt'
    #filepath += 'crypto_volumes.txt'
    #filepath += 'market_history_2017.txt'
    #filepath += 'property_transactions_monthly.txt'
    #filepath += 'PS1'
    #filepath += 'univariate_CP0.csv'
    #filepath += 'univariate_CP1.csv'
    #filepath += 'univariate_CP2.csv'
    #filepath += 'univariate_CP3.csv'
    #filepath += 'PoissonSimulationTest2'
    #filepath += 'PoissonSimulationTest5'
    #filepath += 'PoissonSimulationTest5Extended'
    #filepath += 'PoissonSimulationTest5Extended2'
    #filepath += 'PoissonSimulationTest5Extended3'
    #filepath += 'PoissonSimulationTest5Stretched'
    #filepath += 'PoissonSimulationTest5Stretched2'
    #filepath += 'PoissonSimulationTest5Stretched3'
    #filepath += 'PoissonSimulationTest6'
    #filepath += 'MDGP_test1'
    #filepath += 'MDGP_test3'
    #filepath += 'MDGP_test4'
    #filepath += 'MDGP_test5'
    #filepath += 'MDGP_test5Extended50'
    #filepath += 'MDGP_test5Stretched1_2'
    #filepath += 'MDGP_test5Stretched1_4'
    #filepath += 'MDGP_test5Stretched0_5'
    #filepath += 'MDGP_test5Stretched0_2'
    #filepath += 'MDGP_test6'
    #filepath += 'MDGP_test7'
    #filepath += 'MDGP_test8'
    #filepath += 'MDGP_test8extended200'
    #filepath += 'MDGP_test8extended400'
    #filepath += 'MDGP_test8Stretched1_4'
    #filepath += 'MDGP_test8Stretched1_2'

    """Artificial datasets"""
    #filepath += 'univariate_linear.csv'
    #filepath += 'PS_len10_stretch1.csv'
    # filepath += 'PS_len20_stretch1.csv'
    # filepath += 'PS_len30_stretch1.csv'
    # filepath += 'PS_len10_stretch2.csv'
    # filepath += 'PS_len10_stretch3.csv'
    # filepath += 'PS_len10_stretch0p5.csv'
    # filepath += 'PS_len10_stretch0p33.csv'

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
    # filepath += 'ecmwf_monthly_sm_geq_percentile065_count_grid1.csv'
    # filepath += 'ecmwf_monthly_t2m_geq_percentile065_count_grid0.csv'

    #filepath += 'mPS.csv'
    filepath += 'test.csv'

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

    """check Data!"""
    #print("Generating plot before running CP detection...")
    #known_changepoints = [T // 2]
    #r_data_plot(
    #    data,
    #    T,
    #    s1 * s2,
    #    dateindex=firstdateindex,
    #    dateincr= 1,#/365.25
    #    title="Data"
    #)


    """choose prior parameters"""
    # Parameters common across models
    prior_hazard = 30
    pruning_threshold = T + 1

    # Parameter for PG model
    prior_alpha = 1 #4 * 1000000000# 1 #1150 # 4 * 1000000000
    prior_beta = 1 # 1
    prior_sums = np.ones(s1*s2)

    # Parameter for MDGP model
    prior_alphas = np.repeat(1, s1*s2) # 1

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

    # New MDGP model instantiation
    mdgp_model = MDGPModel(
         prior_alphas,
         s1,
         s2,
         auto_prior_update=False,
    )

    """create Detector object"""

    #Jay: Detector with both MDGP and PG
    # detector = Detector(
    #     data,
    #     np.array([mdgp_model,pg_model]),
    #     np.array([3/4,1/4]),
    #     cp_model,
    #     s1,
    #     s2,
    #     T,
    #     threshold = None, # Copy the old detector. No trimming for now
    #     # threshold=pruning_threshold,
    #     trim_type = "keep_K",
    #     store_mrl = True,
    #     store_rl = True
    # )

    # Jay: Just the MDGP Model
    detector = Detector(
        data,
        np.array([mdgp_model]),
        np.array([1]),
        cp_model,
        s1,
        s2,
        T,
        threshold = None, # Copy the old detector. No trimming for now
        #threshold=pruning_threshold,
        trim_type = "keep_K",
        store_mrl = True,
        store_rl = True
    )


    """run MVBOCPDwMS"""
    print("Running CP detection...")
    j = 0
    for t in range(0, T):
        detector.next_run(data[t, :], t+1)
        #old_detector.next_run(data[t, :], t+1)
        if t >= j*T//100:
            print(j, "% Complete")
            j += 1

    # j = 0
    # for t in range(0, T):
    #     detector.next_run(data[t, :], t+1)
    #     #print('t:',t)
    #     if t >= j*T//100:
    #         print(j, "% Complete")
    #         j += 1


    print("Done. Plotting...")

    # EvT = EvaluationTool()
    # EvT.build_EvaluationTool_via_run_detector(detector)
    #
    #
    # EvT.plot_run_length_distr(buffer=0, show_MAP_CPs = True,
    #                 mark_median = False,
    #                 mark_max = True, upper_limit = T-3, print_colorbar = True,#upper_limit = T-2
    #                 colorbar_location= 'bottom',log_format = True, aspect_ratio = 'auto',
    #                 #C1=1,C2=0,
    #                 time_range = np.linspace(1,T-3, T-3, dtype=int),
    #                 #start = 622 + 2, stop = 1284, #start=start, stop = stop,
    #                 #start = 0,
    #                 #stop = T,
    #                 all_dates = None, #all_dates,
    #                 space_to_colorbar = 0.52,
    #                 custom_colors = ["blue", "blue"], #["blue"]*len(event_time_list),
    #                 custom_linestyles = ["solid"]*3,
    #                 custom_linewidth = 3,
    #                 arrow_colors= ["black"],
    #                 number_fontsize = 14,
    #                 arrow_length = 135,
    #                 arrow_thickness = 3.0,
    #                 xlab_fontsize = 14,
    #                 ylab_fontsize = 14,
    #                 arrows_setleft_indices = [0],
    #                 arrows_setleft_by = [50],
    #                 zero_distance = 0.0,
    #                 ax = None, figure = None,
    #                 no_transform = True,
    #                 date_instructions_formatter = None, #yearsFmt,
    #                 date_instructions_locator = None,
    #                 #ylabel_coords = ylabel_coords,
    #                 xlab = "Time",
    #                 arrow_distance = 25)
    #
    # EvT.plot_raw_TS(data, indices = [0,1,2], print_plt = True,
    #                 show_MAP_CPs = False,
    #                 legend = False, legend_labels = None,
    #                 legend_position = None, time_range = None,
    #                 start_plot = None, stop_plot = None,
    #                 aspect_ratio = 'auto',
    #                 xlab = "Time",
    #                 ylab = "Value",
    #                 ax = None,
    #                 xlab_fontsize = 10,
    #                 ylab_fontsize = 10,
    #                 xticks_fontsize = 10,
    #                 yticks_fontsize = 10,
    #                 all_dates = None,
    #                 custom_linestyles = None,
    #                 custom_colors_series = None,
    #                 custom_colors_CPs = None,
    #                 custom_linewidth = 3.0,
    #                 ylabel_coords = None,
    #                 true_CPs = None)
    #
    # EvT.plot_predictions(indices = [0,1,2], print_plt = True, legend = False,
    #                      legend_labels = None,
    #                      legend_position = None, time_range = None,
    #                      show_var = True, show_CPs = False,
    #                      ax = None, aspect_ratio = 'auto')
    # EvT.plot_prediction_error(data,  indices=[0], time_range = None,
    #                           print_plt=True,
    #                           legend=False, show_MAP_CPs = False,
    #                           show_real_CPs = False, show_var = False,
    #                           custom_colors = None,
    #                           ax=None, xlab = "Time", ylab = "Value",
    #                           aspect_ratio = 'auto', xlab_fontsize = 10,
    #                           ylab_fontsize = 10,
    #                           xticks_fontsize = 10,
    #                           yticks_fontsize = 10,
    #                           ylabel_coords = None)
    # EvT.plot_model_posterior(indices = [0], plot_type = "trace", #plot types: trace, MAP
    #                          y_axis_labels = None, #only needed for MAP type
    #                          print_plt = True, time_range = None,
    #                          start_plot = None, stop_plot = None,
    #                          legend=False,
    #                          period_time_list = None,
    #                          label_list = None,
    #                          show_MAP_CPs = True, show_real_CPs = False,
    #                          log_format = True, smooth = False, window_len = 126,
    #                          aspect = 'auto', xlab = "Time", ylab = "P(m|y)",
    #                          custom_colors = None, ax = None,
    #                          start_axis = None, stop_axis = None,
    #                          xlab_fontsize= 10, ylab_fontsize = 10,
    #                          number_offset = 0.25,
    #                          number_fontsize=10,
    #                          period_line_thickness = 3.0,
    #                          xticks_fontsize = 12,
    #                          yticks_fontsize = 12,
    #                          ylabel_coords = None,
    #                          SGV = False,
    #                          log_det = False,
    #                          all_dates=None,
    #                          true_CPs = None)
    # EvT.plot_model_and_run_length_distr(print_plt = True, time_range = None,
    #                           show_MAP_CPs = True, show_real_CPs = False,
    #                           mark_median = True, log_format = True,
    #                           CP_legend = False, buffer = 50)

    r_distr_plot(
        data,
        T,
        detector,
        #old_detector,
        s1*s2,
        dateindex=firstdateindex,
        dateincr=1,
        title="Simulation Residual Data"
    )

if __name__ == "__main__":
    test()
