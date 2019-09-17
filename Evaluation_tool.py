# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:22:50 2017

@author: Jeremias Knoblauch (j.knoblauch@warwick.ac.uk)

Description: Convenient object that serves as a wrapper for experiments and
(i) creates model universe members and their detectors, (ii) runs the algo,
(iii) stores results (either to itself or the HD), (iv) can simply re-call old
results that were stored on HD (iv) allows for a variety of plotting functions
once the algo has run/once data has been read in.
"""

from detector import Detector
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.dates import drange
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import misc
import numpy as np
import pickle
import os
import datetime
import collections
from nearestPD import NPD
import matplotlib.lines as mlines


from BVAR_NIG import BVARNIG

class EvaluationTool:
    """Description: Convenient object that serves as a wrapper for experiments and
    (i) creates model universe members and their detectors, (ii) runs the algo,
    (iii) stores results (either to itself or the HD), (iv) can simply re-call old
    results that were stored on HD (iv) allows for a variety of plotting functions
    once the algo has run/once data has been read in.

    type:
            -> Gives you info on how the tool was initialized. Was it given
            a detector only? Or was it given specs for creating models and
            a detector? Or was it created from old data, and we only wanna plot?
            The higher the number, the less work the tool has left to do before
            plots can be generated
            ---------
            1 => We get a collection of model specifications that are not
                 in a detector yet
            2 => We get a collection of (ready) models that are not in a
                 detector yet
            3 => We get a detector
            4 => We have the results from running the detector (but not
                 necessarily any detector, model specifications, etc.)
    """

    def __init__(self):
        """the object is initialized as empty, and there are basically two ways
        of creating one: Either you input models + detectors (or their
        arguments), or you give it the path to data that has been stored from
        a previous experiment"""
        self.type = None
        self.has_true_CPs = False
        self.names = ["names", "execution time", "S1", "S2", "T",
                 "trimmer threshold", "MAP CPs", "model labels",
                     "run length log distribution",
                     "model and run length log distribution",
                     "one-step-ahead predicted mean",
                     "one-step-ahead predicted variance",
                     "all run length log distributions",
                     "all model and run length log distributions",
                     "all retained run lenghts",
                     "has true CPs",  "true CP locations",
                     "true CP model index", "true CP model labels"]

        """NOTE: Plotting will work with 7 different colors and 4 different
                 line styles!"""
        self.linestyle = ["-", "--", "-.", ":",]*20
        self.colors = ['b',  'c', 'm', 'y', 'k', 'g']*20
        self.CP_color = 'r'

        self.cushion = 10 #cushion when plotting RL distro
        self.median_color = 'g'
        self.max_color = 'r'

        """Initialize s.t. the baseline assumption is that we do not know the
        true CPs"""
        self.has_true_CPs = False
        self.true_CP_location = self.true_CP_model_index = self.true_CP_model_label = None



    """*********************************************************************

                TYPE I FUNCTIONS: Create Evaluation Tool via Models

    *********************************************************************"""

    def build_EvaluationTool_models_via_specs(self, model_class_list,
            specification_list, data=None, detector_specs=None):
        """ *model_class_list* gives you the list of models to be created, and
        *specification_list* will give you the list of lists of parameters for
        each model to be created. *data* will give you the data to run the
        models on, and *detector_specs* are additional specifications for the
        detector (like intensity of the CP occurence, ...)"""

        """STEP 1: Store the essentials and get your type"""
        self.model_class_list = model_class_list
        self.specification_list = specification_list
        self.type = 1

        """STEP 2: Build all your models, and advance the type by one"""
        self.model_universe = []
        for (model_class, specs) in zip(self.model_class_list,
                self.specification_list):
            self.model_universe.append(model_class(*specs))
        self.type = 2

        """STEP 3: If you have data, put all models into a detector that is
        created with default specs unless you supplied other specs. If you
        create this detector, advance type by one again."""
        if not( data is None ):
            if not (detector_specs is None):
                self.detector = Detector(data, *detector_specs)
            else:
                self.detector = Detector(data)
            self.type = 3


    def build_EvaluationTool_models_via_universe(self, model_universe, data=None,
                                        detector_specs=None):
        """ *model_list* stores a collection of models for the segments, and
        *data* gives you the data you wish to analyze using those models. As
        before, *detector_specs* is an optional list of arguments for the
        detector that will be created (if the data is passed)."""

        """STEP 1: Store the model universe that was passed to the function"""
        self.model_universe = model_universe
        self.type = 2

        """STEP 2: If you can/should, create the detector"""
        if not( data is None ):
            if not (detector_specs is None):
                self.detector = Detector(data, *detector_specs)
            else:
                self.detector = Detector(data)
            self.type = 3


    def build_EvaluationTool_models_via_csv(self, specification_csv,
                                     data=None, detector_specs=None):
        """ *specification_csv* stores the path and name of a csv file
        containing the model_class_list and the specification_list equivalent
        from before, but simply in a .csv file with a certain structure."""
        #structure: First row are the headers, i.e. gives you the names
        #           of the quantities defining the BVAR structure
        pass #DEBUG: Not clear how to read a spreadsheet, and how to structure it yet


    """*********************************************************************

            TYPE I FUNCTIONS: Create Evaluation Tool via results

    *********************************************************************"""

    def build_EvaluationTool_via_results(self, result_path):
        """Give it the path to a pickle-created list of results that you can
        work with to do all the plotting"""

        f_myfile = open(result_path, 'rb')
        self.results = pickle.load(f_myfile)
        f_myfile.close()
#        with open(result_path, 'rb') as fp:
#            self.results = pickle.load(fp)
        self.names = self.results[0]
        self.type=4

    def build_EvaluationTool_via_run_detector(self, detector,
            true_CP_location=None, true_CP_model_index = None,
            true_CP_model_label = None):

        if ((true_CP_location is not None) and  (true_CP_model_index is not None)):
            self.add_true_CPs(true_CP_location, true_CP_model_index,
                     true_CP_model_label )

        self.detector = detector
        self.type = 4
        self.store_detector_results_to_object()

    """*********************************************************************

                TYPE I FUNCTIONS: Add data/true Cps/... To Evaluation Tool

    *********************************************************************"""


    def add_data_and_detector_via_hand(self, data, detector_specs = None):
        """ Let's you create the detector if you don't wish to pass the data
        into the EvaluationTool object right away."""
        if self.type is None or self.type < 2:
            print("Error! Passing in data and detector specs before " +
                  "creating model_universe!")
        else:
            if not (detector_specs is None):
                self.detector = Detector(data, *detector_specs)
            else:
                self.detector = Detector(data)
            self.type = 3


    def create_data_add_detector_via_simulation_obj(self, sim_class,
                                sim_parameters,
                                detector_specs = None):
        """You run a simulation *sim_class* that takes parameters stored in
        *sim_parameters* and creates a detector afterwards. It is essential
        that the sim_class object stores the generated data as ".data" """
        if self.type is None or self.type < 2:
            print("Error! Passing in data and detector specs before " +
                  "creating model_universe!")
        else:
            data = (sim_class(*sim_parameters).generate_data()).data
            if not (detector_specs is None):
                self.detector = Detector(data, *detector_specs)
            else:
                self.detector = Detector(data)
            self.type = 3


    def create_data_add_detector_via_simulation_csv(self, sim_csv,
                                detector_specs = None):
        """You run a simulation *sim_class* that takes parameters stored in
        *sim_parameters* and creates a detector afterwards. It is essential
        that the sim_class object stores the generated data as ".data" """
        #DEBUG: also store true CPS!
        #DEBUG: Store a boolean indicating that data was created with true CPs
        pass

    def add_true_CPs(self, true_CP_location, true_CP_model_index,
                     true_CP_model_label = None):
        """Add true CPs and store them. *true_CP_location* gives you the time
        at which the CP occured. *true_CP_model* gives you the model index in
        the detector object corresponding to the model starting at the corr.
        CP location. *true_CP_model_label* gives you the string label of the
        true DGP starting at the CP, e.g.
        true_CP_model_label = ["BVAR(4,4,1)", "BVAR(1,1,1,1,1)"]."""
        self.true_CP_location = true_CP_location
        self.true_CP_model_index = true_CP_model_index
        #if self.model_labels is None:
        #    self.model_labels = true_CP_model_label
        self.true_CP_model_label = true_CP_model_label
        self.has_true_CPs = True


    """*********************************************************************

                TYPE II FUNCTIONS: Run the algorithm, store results

    *********************************************************************"""


    def run_algorithm(self, start=None, stop=None):
        """Just runs the detector and stores all the results that we want
        inside the Evaluation_tool object"""

        """STEP 1: Run algo"""
        if start is None or stop is None:
            self.detector.run()
        else:
            self.detector.run(start, stop)
        self.type = 4
        self.store_detector_results_to_object()


    def store_detector_results_to_object(self):
        if self.type < 4:
            print("CAREFUL! Your detector seems to not have been run yet, " +
                  "but you still store its content to your EvaluationTool object!")
        """STEP 2: Store all raw quantities inside the object"""
        self.S1, self.S2, self.T = self.detector.S1, self.detector.S2, self.detector.T
        self.execution_time = self.detector.execution_time
        self.CPs = self.detector.CPs
        self.threshold = self.detector.threshold
        self.run_length_log_distr = self.detector.run_length_log_distr
        self.model_and_run_length_log_distr = self.detector.model_and_run_length_log_distr
        if self.detector.save_performance_indicators:
            self.MSE = self.detector.MSE
            self.negative_log_likelihood = self.detector.negative_log_likelihood

        if self.detector.store_rl or self.detector.store_mrl:
            self.storage_all_retained_run_lengths = (
                    self.detector.storage_all_retained_run_lengths)
        if self.detector.store_rl:
            self.storage_run_length_log_distr = self.detector.storage_run_length_log_distr
        else:
            self.storage_run_length_log_distr = None
        if self.detector.store_mrl:
            self.storage_model_and_run_length_log_distr = (self.
                            detector.storage_model_and_run_length_log_distr)
        else:
            self.storage_model_and_run_length_log_distr = None

        #self.data = self.detector.data
        self.storage_mean = self.detector.storage_mean
        self.storage_var = self.detector.storage_var

        """STEP 2.1: Store strings that give you the model label"""
        if isinstance(self.detector.model_universe, list):
            M = int(len(self.detector.model_universe))
        else:
            M = self.detector.model_universe.shape[0]
        self.model_labels = [None]*M
        for i in range(0,M):
            class_label = str(
                type(self.detector.model_universe[i])).split(
                ".")[-1].split("'")[0]
            if self.detector.model_universe[i].has_lags:
                if isinstance(self.detector.model_universe[i], BVARNIG):
                    nbh_label = "[BAR]"
                else:
                    if self.detector.model_universe[i].nbh_sequence is None:
                        #general nbh
                        lag_length = self.detector.model_universe[i].lag_length
                        nbh_label = "[general nbh, " + str(lag_length) + "]"
                    else:
                        nbh_label = str(self.detector.model_universe[i].nbh_sequence)
                self.model_labels[i] = class_label + nbh_label
            else:
                self.model_labels[i] = class_label


        """STEP 3: Sum them all up in a results-object"""
        self.results = [self.execution_time, self.S1, self.S2, self.T,
                   self.threshold, self.CPs, self.model_labels,
                       self.run_length_log_distr,
                       self.model_and_run_length_log_distr,
                       self.storage_mean, self.storage_var,
                       self.storage_run_length_log_distr,
                       self.storage_model_and_run_length_log_distr, #,
                       self.storage_all_retained_run_lengths
                       ]
        if self.detector.save_performance_indicators:
            self.names.append("MSE")
            self.names.append("NLL")
            self.results.append(self.MSE)
            self.results.append(self.negative_log_likelihood)
        if self.has_true_CPs:
            self.names.append("has true CP")
            self.names.append("true CP location")
            self.names.append("true CP model index")
            self.names.append("true CP model label")
            self.results.append(self.has_true_CPs)
            self.results.append(self.true_CP_location)
            self.results.append(self.true_CP_model_index)
            self.results.append(self.true_CP_model_label)

        """append the names to the front of results"""
        self.results.insert(0, self.names)


    def store_results_to_HD(self, results_path):
        """For all objects stored inside the object, store them in a certain
        structure to the location at *path*, provided that the algorithm has
        run already."""

        """Check if the algorithm has already been run. If so create a list of
        names and results and store to HD!"""
        if self.type == 4:
            """store to HD"""
            f_myfile = open(results_path, 'wb')
            pickle.dump(self.results, f_myfile)
            f_myfile.close()
            #with open(results_path, 'rb') as fp:
            #    pickle.dump(self.results, fp)


    def run_algorithm_store_results_to_HD(self, results_path, start=None, stop=None):
        self.run_algorithm(start,stop)
        self.store_results_to_HD(results_path)


    """*********************************************************************

            TYPE III FUNCTIONS: create/store/read model configurations

    *********************************************************************"""

    #DEBUG: Static method to store configurations in pickle format
    def store_BVAR_configurations_to_HD(configs, path):
        """Store the configurations passed as *config* to *path* using the
        pickle module, i.e. s.t. you can retrieve them directly as a list of
        arguments"""
        i = 0
        for config in configs:
            config_path = path + "\\" + str(i)
            f_myfile = open(config_path, 'wb')
            pickle.dump(config, f_myfile)
            f_myfile.close()
#            with open(config_path, 'rb') as fp:
#                pickle.dump(config, fp)
            i = i+1

    def read_BVAR_configuration_from_HD(path):
        """Retrieve previously stored configs and return them in a list of
        lists of arguments"""
        list_of_file_paths = os.listdir(path)
        list_of_configs = []
        i = 0
        for config_path in list_of_file_paths:
            f_myfile = open(config_path, 'rb')
            config = pickle.load(f_myfile)
            f_myfile.close()
            list_of_configs.append(config)
#            config = pickle.load(open(config_path, 'rb'))
#            list_of_configs.append(config)
            i = i+1


    def create_BVAR_configurations_easy(num_configs,
                                        a, b, prior_mean_beta, prior_var_beta,
                                        S1,S2,nbh_sequence, restriction_sequence,
                                        intercept_grouping = None,
                                        general_nbh_sequence  = None,
                                        general_nbh_restriction_sequence = None,
                                        nbh_sequence_exo = None, exo_selection = None,
                                        padding = None, auto_prior = None):
        """Idea: You create a sequence of BVAR configs s.t. all parameters which
        are only put in once into the model are used for each individual spec-
        ification, but all parameters which are put in *num_configs* times are
        varied across the *num_configs* created specifications"""

        """STEP 1: Loop over all the arguments that are passed to this fct.
                  If they have only one entry, make that entry the entry of
                  each config. If they don't, do nothing"""
        a = EvaluationTool.create_args(num_configs, a)
        b = EvaluationTool.create_args(num_configs, b)
        prior_mean_beta = EvaluationTool.create_args(num_configs, prior_mean_beta)
        prior_var_beta = EvaluationTool.create_args(num_configs, prior_var_beta)
        S1, S2 = EvaluationTool.create_args(num_configs, S1), EvaluationTool.create_args(num_configs, S2)
        nbh_sequence = EvaluationTool.create_args(num_configs, nbh_sequence)
        restriction_sequence= EvaluationTool.create_args(num_configs, restriction_sequence)
        intercept_grouping = EvaluationTool.create_args(num_configs,
                                                        intercept_grouping)
        general_nbh_sequence= EvaluationTool.create_args(num_configs, general_nbh_sequence)
        general_nbh_restriction_sequence= EvaluationTool.create_args(
                num_configs, general_nbh_restriction_sequence)
        nbh_sequence_exo= EvaluationTool.create_args(num_configs, nbh_sequence_exo)
        exo_selection= EvaluationTool.create_args(num_configs, exo_selection)
        padding= EvaluationTool.create_args(num_configs, padding)
        auto_prior= EvaluationTool.create_args(num_configs, auto_prior)

        """STEP 2: Create all the configurations in a list of configs"""
        configs = [None] * num_configs
        for i in range(0, num_configs):
           #create the configs
           configs[i] = [a[i], b[i], prior_mean_beta[i], prior_var_beta[i],
                   S1[i], S2[i], nbh_sequence[i], restriction_sequence[i],
                   intercept_grouping[i],
                   general_nbh_sequence[i], general_nbh_restriction_sequence[i],
                   nbh_sequence_exo[i],exo_selection[i],padding[i],auto_prior[i]
                   ]

        """STEP 3: Return configurations"""
        return configs


    def create_args(num, arg):
        """Modifies arg into a list of lenght num which contains arg num times
        if arg has length 0, into a list of length None"""
        if arg is None:
            arg = [None] * num
        elif int(len(arg)) == 1:
            arg = [arg]* num
        return arg


    def create_BVAR_configurations(a_list, b_list, prior_mean_beta_list,
        prior_var_beta_list, S1_list, S2_list,
        nbh_sequence_list, restriction_sequence_list,
        intercept_grouping_list = None,
        general_nbh_sequence_list = None,
        general_nbh_restriction_sequence_list = None,
        nbh_sequence_exo_list = None, exo_selection_list = None,
        padding_list = None, auto_prior_list = None):
        """Create config list and store to file using pickle dump"""

        """STEP 1: Get the number of configs and adapt all 'None' entries"""
        num_configs = int(len(a_list))
        if intercept_grouping_list is None:
            intercept_grouping_list = [None] * num_configs
        if general_nbh_sequence_list is None:
            general_nbh_sequence_list = [None] * num_configs
        if general_nbh_restriction_sequence_list is None:
            general_nbh_restriction_sequence_list = [None] * num_configs
        if nbh_sequence_exo_list is None:
            nbh_sequence_exo_list = [None] * num_configs
        if exo_selection_list is None:
            exo_selection_list = [None] * num_configs
        if padding_list is None:
            padding_list = [None] * num_configs
        if auto_prior_list is None:
            auto_prior_list = [None] * num_configs

        """STEP 2: package everything into lists and save them with pickle"""
        configs=[None] * num_configs
        for i in range(0, num_configs):
            configs[i] = [a_list[i], b_list[i], prior_mean_beta_list[i],
                      prior_var_beta_list[i], S1_list[i], S2_list[i],
                      nbh_sequence_list[i], restriction_sequence_list[i],
                      intercept_grouping_list[i], general_nbh_sequence_list[i],
                      general_nbh_restriction_sequence_list[i],
                      nbh_sequence_exo_list[i], exo_selection_list[i],
                      padding_list[i],auto_prior_list[i]]

        """STEP 3: Return configurations"""
        return(configs)


    """*********************************************************************

                TYPE IV FUNCTIONS: Create plots

    *********************************************************************"""
    """Helper function: Smoothing"""
    def smooth(x,window_len=11,window='hanning'):
        """smooth the data using a window with requested size.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

            see also:

                numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
                scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """


        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            #raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
            print("Window type not admissible")

        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y



    """PLOT I: get the raw data together with the true CPs and return as a
                figure plt.figure() object.
        Options:
            indices     => list of indices in 1d. (data already be flattened)
                            s.t. the corresponding TS will be plotted
            time_range  => range of time over which we should plot the TS
            print_plt   => boolean, decides whether we want to see the plot
                           or just create the object to pass to the next fct.
            legend      => boolean, whether or not we want a legend
            legend_labels => gives you the labels for the TS as a list of
                             strings. If you don't specify the labels,
                             default is 1,2,...
            legend_position => gives the
                             position of the legend, and default is upper left

    """
    def plot_raw_TS(self, data, indices = [0], print_plt = True,
                    show_MAP_CPs = False,
                    legend = False, legend_labels = None,
                    legend_position = None, time_range = None,
                    start_plot = None, stop_plot = None,
                    aspect_ratio = 'auto',
                    xlab = "Time",
                    ylab = "Value",
                    ax = None,
                    xlab_fontsize = 10,
                    ylab_fontsize = 10,
                    xticks_fontsize = 10,
                    yticks_fontsize = 10,
                    all_dates = None,
                    custom_linestyles = None,
                    custom_colors_series = None,
                    custom_colors_CPs = None,
                    custom_linewidth = 3.0,
                    custom_transparency = 1.0,
                    ylabel_coords = None,
                    true_CPs = None,
                    additional_CPs = None,
                    custom_colors_additional_CPs = None,
                    custom_linestyles_additional_CPs = None,
                    custom_linewidth_additional_CPs = None,
                    custom_transparency_additional_CPs = 1.0,
                    set_xlims = None,
                    set_ylims = None,
                    up_to = None):
        """Generates plot of the raw TS at the positions marked by *indices*, over
        the entire time range unless specificed otherwise via *time_range*. It
        prints the picture to the console if *print_plt* is True, and puts a
        legend on the plot if *legend* is True"""


        """STEP 1: Default is to take the entire time range"""
        T = data.shape[0] #self.results[self.names.index("T")]
        if time_range is None:
            time_range = np.linspace(1,T,T, dtype=int)

        """STEP 2: If we do want a legend, the labels are 1,2,3... by default
                   and we plot in the upper left corner by default."""
        num = int(len(indices))
        if legend:
            if (legend_labels is None):
                legend_labels = [str(int(i)) for i in np.linspace(1,num,num)]
            if legend_position is None:
                legend_position = 'upper left'
        else:
            legend_labels = []


        """STEP 3: Plot all the lines specified by the index object"""
       #S1, S2 = self.results[self.names.index("S1")], self.results[self.names.index("S2")]

        #print(self.results[self.names.index("data")].shape)
                #[time_range-1 ,:,:]).reshape((int(len(time_range)), S1*S2))))

        #NOTE: We do not store the data in the detector (anymore), so read
        #       it in separately and then pass it into the fct.
        #data = (self.results[self.names.index("data")]
        #        [time_range-1 ,:][:,indices])
        if custom_colors_series is None:
            custom_colors_series = self.colors
        if custom_colors_CPs is None:
            custom_colors_CPs = self.CP_color * 100

        if ax is None:
            figure, ax = plt.subplots()

        if all_dates is None:
            if start_plot is None or stop_plot is None:
                x_axis = time_range
            else:
                x_axis = np.linspace(start_plot, stop_plot, len(time_range))
            start, stop = time_range[0], time_range[-1]
        else:
            x_axis = all_dates
            start, stop = all_dates[0], all_dates[-1]

        #if we want to plot everything
        if up_to is None or up_to > len(data[:,0]):
            up_to = len(data[:,0])


        legend_handles = []
        for i in range(0, num): #num = len(indices)
            """The handle is like an identifier for that TS object"""
            handle = ax.plot(x_axis[:up_to],
                    data[:up_to,indices[i]], color = custom_colors_series[i])
            legend_handles.append(handle)
        if not all_dates is None:
            if isinstance(all_dates[0], datetime.date):
                ax.xaxis_date()

        T_ = len(time_range)

        """STEP 4: If we have true CPs, plot them into the figure, too"""
        if False: #DEBUG: We need to add CP option self.results[self.names.index("has true CPs")]:
            CP_legend_labels = []
            CP_legend_handles = []
            CP_locations = self.results[self.names.index("true CP locations")]
            CP_model_labels = self.results[self.names.index("true CP model labels")]
            CP_model_index = self.results[self.names.index("true CP model index")]
            #DEBUG: How do I retrieve model index, model label and locatoin
            #       from the results? I NEED TO STORE THEM THERE FIRST, TOO!
            for  (CP_loc, CP_ind, CP_lab) in zip(CP_locations,
                 CP_model_index, CP_model_labels):
                handle = ax.axvline(x=CP_loc, color = self.CP_color,
                        linestyle = self.linestyle[CP_ind])
                CP_legend_handles.append(handle)
                CP_legend_labels.append(CP_lab)
            #DEBUG: Could make this conditional on another boolean input
            legend_handles += CP_legend_handles
            legend_labels += CP_legend_labels

        if additional_CPs is not None:
            CP_object = additional_CPs
            CP_locations = [entry[0] for entry in CP_object]
            CP_indices = [entry[1] for entry in CP_object]

            if custom_linestyles_additional_CPs is None:
                custom_linestyles_additional_CPs = self.linestyle #['solid']*len(CP_locations)
            if custom_linewidth_additional_CPs is None:
                custom_linewidth_additional_CPs = 3.0
            if custom_colors_additional_CPs is None:
                custom_colors_additional_CPs = custom_colors_CPs

            CP_legend_labels = []
            CP_legend_handles = []
            CP_indices_until_now = []
            count = 0

            """Loop over the models in order s.t. you can color in the same
            fashion as for the model posterior"""
            M = int(len(np.unique(np.array(CP_indices))))
            for m in range(0, M):
                for  (CP_loc, CP_ind) in zip(CP_locations, CP_indices):
                    if m == CP_ind:
                        if CP_loc <= time_range[-1] and CP_loc >= time_range[0]:
                            CP_loc = ((CP_loc - time_range[0])/T_)*(stop-start) + start# carry CP forward
                            if CP_ind not in CP_indices_until_now:
                                handle = ax.axvline(x=CP_loc, color = custom_colors_additional_CPs[count],
                                    linestyle = custom_linestyles_additional_CPs[count],
                                    #dashes = [3,6,3,6,3,6,18],
                                    linewidth = custom_linewidth_additional_CPs,
                                    alpha = custom_transparency_additional_CPs)
                                CP_legend_handles.append(handle)
                                #CP_legend_labels.append(model_labels[CP_ind])
                                CP_indices_until_now.append(CP_ind)
                                count= count+1
                            elif CP_ind in CP_indices_until_now:
                                """display it in the same color"""
                                relevant_index = CP_indices_until_now.index(CP_ind)
                                handle = ax.axvline(x=CP_loc, color = custom_colors_additional_CPs[relevant_index],
                                    linestyle = custom_linestyles_additional_CPs[relevant_index],
                                    linewidth = custom_linewidth_additional_CPs,
                                    alpha = custom_transparency_additional_CPs)

        if show_MAP_CPs:
            #which CPs to consider
            if up_to == len(data[:,0]):
                #i.e., we have not specified up_to in the input
                CP_object = self.results[self.names.index("MAP CPs")][-2]
            else:
                if (len(self.results[self.names.index("MAP CPs")][up_to]) == 0
                  and
                  up_to < len(self.results[self.names.index("MAP CPs")]) - 2):
                    #get the first entry which is not empty if up_to entry is 0
                    count = up_to
                    bool_ = True
                    while bool_:
                        count = count + 1
                        if len(self.results[
                                self.names.index("MAP CPs")][count]) > 0:
                            bool_ = False
                    CP_object = self.results[self.names.index("MAP CPs")][count]
                elif (up_to >= len(self.results[
                        self.names.index("MAP CPs")]) - 2):
                    #we have a too large value for up_to
                    CP_object = self.results[self.names.index("MAP CPs")][-2]
                else:
                    #our value of up_to is in range
                    CP_object = self.results[self.names.index("MAP CPs")][up_to]


            CP_locations = [entry[0] for entry in CP_object]
            CP_indices = [entry[1] for entry in CP_object]
            model_labels = self.results[self.names.index("model labels")]
            """if no custom color, take standard"""
#            if custom_colors is None:
#                custom_colors = [self.CP_color]*len(CP_locations)
            if custom_linestyles is None:
                custom_linestyles = self.linestyle #['solid']*len(CP_locations)
            if custom_linewidth is None:
                custom_linewidth = 3.0

            CP_legend_labels = []
            CP_legend_handles = []
            CP_indices_until_now = []
            count = 0

            """Loop over the models in order s.t. you can color in the same
            fashion as for the model posterior"""
            M = len(self.results[self.names.index("model labels")])
            for m in range(0, M):
                for  (CP_loc, CP_ind) in zip(CP_locations, CP_indices):
                    if m == CP_ind:
                        if CP_loc <= time_range[-1] and CP_loc >= time_range[0]:
                            CP_loc = ((CP_loc - time_range[0])/T_)*(stop-start) + start# carry CP forward
                            if CP_ind not in CP_indices_until_now:
                                handle = ax.axvline(x=CP_loc, color = custom_colors_CPs[count],
                                    linestyle = custom_linestyles[count],
                                    linewidth = custom_linewidth,
                                    alpha = custom_transparency)
                                CP_legend_handles.append(handle)
                                CP_legend_labels.append(model_labels[CP_ind])
                                CP_indices_until_now.append(CP_ind)
                                count= count+1
                            elif CP_ind in CP_indices_until_now:
                                """display it in the same color"""
                                relevant_index = CP_indices_until_now.index(CP_ind)
                                handle = ax.axvline(x=CP_loc, color = custom_colors_CPs[relevant_index],
                                    linestyle = custom_linestyles[relevant_index],
                                    linewidth = custom_linewidth,
                                    alpha = custom_transparency)

        if not true_CPs is None:
            #true_CPs = [[location, color]]
            for entry in true_CPs:
                ax.axvline(x = entry[0], color = entry[1],
                                     linestyle = "-", linewidth = entry[2])

        """STEP 5: Plot the legend if we want to"""
        if not xlab is None:
            ax.set_xlabel(xlab, fontsize = xlab_fontsize)
        if not ylab is None:
            ax.set_ylabel(ylab,  fontsize = ylab_fontsize)
        if not ylabel_coords is None:
            ax.get_yaxis().set_label_coords(ylabel_coords[0], ylabel_coords[1])
        if not xticks_fontsize is None:
            ax.tick_params(axis='x', labelsize=xticks_fontsize) #, rotation=90)
        if not yticks_fontsize is None:
            ax.tick_params(axis='y', labelsize=yticks_fontsize) #, rotation=90)


        #set x/ylims
        if not set_xlims is None:
            ax.set_xlim(set_xlims[0], set_xlims[1])
        if not set_ylims is None:
            ax.set_ylim(set_ylims[0], set_ylims[1])

        ax.set_aspect(aspect_ratio)
        if legend:
            ax.legend(legend_handles, legend_labels, loc = legend_position)

        """STEP 6: If we are supposed to print this picture, do so. Regardless
                   of whether you print it, return the resulting object"""
        #if print_plt:
        #    plt.show()
        return ax #figure


    """PLOT II: get the 1-step-ahead predictions together with the estimated
                 CPs and return as a figure plt.figure() object.
        Options:
            indices     => list of indices in 1d. (data already be flattened)
                            s.t. the corresponding TS will be plotted
            time_range  => range of time over which we should plot the TS
            print_plt   => boolean, decides whether we want to see the plot
                           or just create the object to pass to the next fct.
            legend      => boolean, whether or not we want a legend
            legend_labels => gives you the labels for the TS as a list of
                             strings. If you don't specify the labels,
                             default is 1,2,...
            legend_position => gives the
                             position of the legend, and default is upper left
            show_var    => bool indicating whether or not the square root of
                           the diagonal of the posterior covariance should be
                           plotted around the mean predictions
            show_CPs    => bool indicating whether or not the MAP CPs should be
                           included in the plot

    """
    def plot_predictions(self, indices = [0], print_plt = True, legend = False,
                         legend_labels = None,
                         legend_position = None, time_range = None,
                         show_var = True, show_CPs = True,
                         ax = None, aspect_ratio = 'auto',
                         set_xlims = None,
                         set_ylims = None):
        """Generates plot of the pred TS at the positions marked by *indices*,
        over entire time range unless specificed otherwise via *time_range*. It
        prints the picture to the console if *print_plt* is True, and puts a
        legend on the plot if *legend* is True. Posterior variances around the
        predicted TS are shown if *show_var* is True. The MAP CPs are shown
        if show_CPs = True."""

        """STEP 1: Default is to take the entire time range"""
        T = self.results[self.names.index("T")]
        if time_range is None:
            time_range = np.linspace(1,T,T, dtype=int)
        if ax is None:
            figure, ax = plt.subplots()

        """STEP 2: If we do want a legend, the labels are 1,2,3... by default
                   and we plot in the upper left corner by default."""
        num = int(len(indices))
        if legend and legend_labels is None:
            legend_labels = [str(int(i)) for i in np.linspace(1,num,num)]
        if legend and legend_position is None:
            legend_position = 'upper left'
        if not legend and legend_labels is None:
            legend_labels = []


        """STEP 3: Plot all the predicted means specified by the index object,
                   and also the predictive variance if *show_var* is True"""
        S1, S2 = self.results[self.names.index("S1")], self.results[self.names.index("S2")]
        means = (self.results[self.names.index("one-step-ahead predicted mean")]
                [time_range-1 ,:,:]).reshape((int(len(time_range)), S1*S2))[:,indices]
        if show_var:
            std_err = np.sqrt(
                    self.results[self.names.index("one-step-ahead predicted variance")]
                [time_range-1 ,:][:,indices])

        #figure = plt.figure()
        legend_handles = []
        for i in range(0, num):
            """The handle is like an identifier for that TS object"""
            handle, = ax.plot(time_range, means[:,i], color = self.colors[i])
            legend_handles.append(handle)
            """If required, also plot the errors around the series"""
            if show_var:
                ax.plot(time_range, means[:,i]+ std_err[:,i], color = self.colors[i],
                         linestyle = ":")
                ax.plot(time_range, means[:,i]-std_err[:,i], color = self.colors[i],
                         linestyle = ":")

        """STEP 4: If we have CPs, plot them into the figure, too"""
        if show_CPs:
            CP_object = self.results[self.names.index("MAP CPs")][-2]
            #print(CP_object)
            CP_locations = [entry[0] for entry in CP_object]
            CP_indices = [entry[1] for entry in CP_object]
            model_labels = self.results[self.names.index("model labels")]
            CP_legend_labels = []
            CP_legend_handles = []
            for  (CP_loc, CP_ind) in zip(CP_locations, CP_indices):
                handle = ax.axvline(x=CP_loc, color = self.CP_color,
                        linestyle = self.linestyle[CP_ind])
                CP_legend_handles.append(handle)
                CP_lab = model_labels[CP_ind]
                CP_legend_labels.append(CP_lab)
            #DEBUG: Could make this conditional on another boolean input
            legend_handles += CP_legend_handles
            legend_labels += CP_legend_labels


        #set x/ylims
        if not set_xlims is None:
            ax.set_xlim(set_xlims[0], set_xlims[1])
        if not set_ylims is None:
            ax.set_ylim(set_ylims[0], set_ylims[1])

        """STEP 5: Plot the legend if we want to"""
        if legend:
            ax.legend(legend_handles, legend_labels, loc = legend_position)

        """STEP 6: If we are supposed to print this picture, do so. Regardless
                   of whether you print it, return the resulting object"""
        #if print_plt:
        #    ax.show()
        ax.set_aspect(aspect_ratio)
        return ax

    """PLOT III: plot the prediction errors

        Options:
            time_range  => range of time over which we should plot the TS
            print_plt   => boolean, decides whether we want to see the plot
                           or just create the object to pass to the next fct.
            legend      => boolean telling you whether you should put the legend
                            in the plot
            show_MAP_CPs    => bool indicating whether or not the MAP CPs should be
                           included in the plot
            show_real_CPs   => bool indicating whether or not the true CPs
                                should be included in the plot
            show_var    => bool indicating whether you should show the pred.
                            std. err. around the pred. error
    """
    def plot_prediction_error(self, data,  indices=[0], time_range = None,
                              print_plt=False,
                              legend=False,
                              show_MAP_CPs = False,
                              show_real_CPs = False, show_var = False,
                              custom_colors = None,
                              ax=None, xlab = "Time", ylab = "Value",
                              aspect_ratio = 'auto', xlab_fontsize = 10,
                              ylab_fontsize = 10,
                              xticks_fontsize = 10,
                              yticks_fontsize = 10,
                              ylabel_coords = None,
                              set_xlims = None,
                              set_ylims = None,
                              up_to = None):

        """STEP 1: Obtain the time range if needed, else set it to 1:T"""
        T = self.results[self.names.index("T")]
        S1 = self.results[self.names.index("S1")]
        S2 = self.results[self.names.index("S2")]
        if time_range is None:
            time_range = np.linspace(1,T,T, dtype=int)
        if ax is None:
            figure, ax = plt.subplots()
        num = int(len(indices))
        if data.ndim == 3:
            data = data.reshape(T, S1*S2)
        if custom_colors is None:
            custom_colors = self.colors
        #indices = np.array(indices)

        """STEP 2: Obtain the prediction errors"""
        dat = data[time_range-1,:][:, indices]
        pred = ((self.results[self.names.index(
                "one-step-ahead predicted mean")]).reshape(
                        T, S1*S2)[time_range-1,:][:,indices])
        pred_errors = (dat-pred)


        #if we want to plot everything
        if up_to is None or up_to > len(dat[:,0]):
            up_to = len(dat[:,0])

        if show_var:
            std_err = np.sqrt(
                    self.results[self.names.index("one-step-ahead predicted variance")]
                [time_range-1 ,:][:,indices])*1

        """STEP 3: Plot all prediction errors and the true/estimated CPs & variances and legend
        labels if needed"""
        legend_labels = [str(int(i)) for i in np.linspace(1,num,num)]
        legend_handles = []
        #figure = plt.figure()
        count = 0
        for i in range(0, num):
            """The handle is like an identifier for that TS object"""
            handle, = ax.plot(time_range[:up_to], pred_errors[:up_to,i],
                               color = custom_colors[count])
            count = count+1
            legend_handles.append(handle)
            """If required, also plot the errors around the series"""
            if show_var:
                ax.plot(time_range[:up_to], pred_errors[:up_to,i]+ std_err[:up_to,i],
                         color = custom_colors[count],
                         linestyle =  (0, (3,1,1,1)))
                ax.plot(time_range[:up_to], pred_errors[:up_to,i]-std_err[:up_to,i],
                         color = custom_colors[count],
                         linestyle =  (0, (3,1,1,1)))
                count = count+1

        if show_real_CPs:
            CP_legend_labels = []
            CP_legend_handles = []
            for  (CP_loc, CP_ind, CP_lab) in zip(self.true_CP_location,
                 self.true_CP_model_index, self.true_CP_model_label):
                handle = ax.axvline(x=CP_loc, color = self.CP_color,
                        linestyle = self.linestyle[CP_ind])
                CP_legend_handles.append(handle)
                CP_legend_labels.append(CP_lab)
            legend_handles+=CP_legend_handles
            legend_labels +=CP_legend_labels


        if show_MAP_CPs:

            if up_to == len(dat[:,0]):
                #i.e., we have not specified up_to in the input
                CP_object = self.results[self.names.index("MAP CPs")][-2]
            else:
                if (len(self.results[self.names.index("MAP CPs")][up_to]) == 0
                  and
                  up_to < len(self.results[self.names.index("MAP CPs")]) - 2):
                    #get the first entry which is not empty if up_to entry is 0
                    count = up_to
                    bool_ = True
                    while bool_:
                        count = count + 1
                        if len(self.results[
                                self.names.index("MAP CPs")][count]) > 0:
                            bool_ = False
                    CP_object = self.results[self.names.index("MAP CPs")][count]
                elif (up_to >= len(self.results[
                        self.names.index("MAP CPs")]) - 2):
                    #we have a too large value for up_to
                    CP_object = self.results[self.names.index("MAP CPs")][-2]
                else:
                    #our value of up_to is in range
                    CP_object = self.results[self.names.index("MAP CPs")][up_to]

            #CP_object = self.results[self.names.index("MAP CPs")][-2]
            CP_locations = [entry[0] for entry in CP_object]
            CP_indices = [entry[1] for entry in CP_object]
            model_labels = self.results[self.names.index("model labels")]
            CP_legend_labels = []
            CP_legend_handles = []
            for  (CP_loc, CP_ind) in zip(CP_locations, CP_indices):
                handle = ax.axvline(x=CP_loc, color = self.CP_color,
                        linestyle = self.linestyle[CP_ind])
                CP_legend_handles.append(handle)
                CP_legend_labels.append(model_labels[CP_ind])
            legend_handles+=CP_legend_handles
            legend_labels +=CP_legend_labels



        """STEP 4: Plot all CPs if needed, and plot the legend if needed"""
        if legend:
            ax.legend(legend_handles, legend_labels, loc = "upper left")
        if not xlab is None:
            ax.set_xlabel(xlab, fontsize = xlab_fontsize)
        if not ylab is None:
            ax.set_ylabel(ylab,  fontsize = ylab_fontsize)
        if not ylabel_coords is None:
            ax.get_yaxis().set_label_coords(ylabel_coords[0], ylabel_coords[1])
        if not xticks_fontsize is None:
            ax.tick_params(axis='x', labelsize=xticks_fontsize) #, rotation=90)
        if not yticks_fontsize is None:
            ax.tick_params(axis='y', labelsize=yticks_fontsize) #, rotation=90)


        #set x/ylims
        if not set_xlims is None:
            ax.set_xlim(set_xlims[0], set_xlims[1])
        if not set_ylims is None:
            ax.set_ylim(set_ylims[0], set_ylims[1])

        ax.set_aspect(aspect_ratio)
        #if print_plt:
        #    plt.show()
        return ax

    """PLOT IV: plot the model posterior

        Options:
            time_range  => range of time over which we should plot the TS
            print_plt   => boolean, decides whether we want to see the plot
                           or just create the object to pass to the next fct.
            legend      => boolean, whether or not to plot the legend
            show_MAP_CPs    => bool indicating whether or not the MAP CPs should be
                           included in the plot
            show_real_CPs   => bool indicating whether or not the true CPs
                                should be included in the plot


    """
    def plot_model_posterior(self, indices = [0],
                             plot_type = "trace", #plot types: trace, MAP,
                                 #MAPVariance1_trace, MAPVariance1_det
                                 #MAPVariance2_trace, MAPVariance2_det
                             y_axis_labels = None, #only needed for MAP type
                             print_plt = True, time_range = None,
                             start_plot = None, stop_plot = None,
                             legend=False,
                             period_time_list = None,
                             label_list = None,
                             show_MAP_CPs = True, show_real_CPs = False,
                             log_format = True, smooth = False, window_len = 126,
                             aspect = 'auto', xlab = "Time", ylab = "P(m|y)",
                             custom_colors = None,
                             custom_linestyles = None,
                             custom_linewidths = None,
                             ax = None,
                             start_axis = None, stop_axis = None,
                             xlab_fontsize= 10, ylab_fontsize = 10,
                             number_offset = 0.25,
                             number_fontsize=10,
                             period_line_thickness = 3.0,
                             xticks_fontsize = 12,
                             yticks_fontsize = 12,
                             ylabel_coords = None,
                             SGV = False,
                             log_det = False,
                             all_dates=None,
                             true_CPs = None,
                             set_xlims = None,
                             set_ylims=None,
                             up_to = None):
        """if no custom colors, use standard colors"""
        if custom_colors is None:
            custom_colors = self.colors
        if custom_linestyles is None:
            custom_linestyles = ["-"] * 9999
        if custom_linewidths is None:
            custom_linewidths = [3.0] * 9999

        """STEP 1: Obtain the time range if needed, else set it to 1:T"""
        T = self.results[self.names.index("T")]
        if time_range is None:
            time_range = np.linspace(1,T,T, dtype=int)
            start=1-1
            stop=T
        else:
            start = time_range[0]-1
            stop= time_range[-1]

        if ax is None:
            figure, ax = plt.subplots()

        if start_plot is None or stop_plot is None:
            start_plot, stop_plot = start, stop

        """STEP 1.5: If indices None, get the CP indices"""
        CP_object = self.results[self.names.index("MAP CPs")][-2]
        CP_locations = [entry[0] for entry in CP_object]
        CP_indices = [entry[1] for entry in CP_object]
        if indices is None:
            indices = CP_indices

        if (not start_axis is None) and (not stop_axis is None):
            ax.set_xlim(start_axis, stop_axis) #debug: use datetime to make this nicer

        """STEP 2: Obtain the model posteriors by summing over all run
        lengths"""
        m_rl_distr = self.results[self.names.index(
                "all model and run length log distributions")]
        M = (m_rl_distr[-1][:,0]).shape[0]
        #DEBUG: offset should be smallest lag length if starting point is smaller
        #       than smallest lag length
        offset = max(0, (np.size(time_range) -
                  len(m_rl_distr)))
        model_posterior = np.zeros((M, np.size(time_range)))

        #should up_to be absent, use the entire time range
        if up_to is None:
            up_to = np.size(time_range)

        for (t,i) in zip(range(start + offset, stop), range(0, np.size(time_range))): #m_rl_distr[time_range]:
            for m in range(0,M):
                if m<m_rl_distr[t-offset][:,:].shape[0]:
                    model_posterior[m,i] = misc.logsumexp(
                            m_rl_distr[t-offset][m,:])
        if not log_format:
            model_posterior = np.exp(model_posterior)
        #if smooth:
        #    print("why am I here")
            #for m in range(0,M):
            #    model_posterior[m,:] = EvaluationTool.smooth(
            #            model_posterior[m,:],
            #                   window_len = window_len)[int(0.5*window_len):
            #                    -int(0.5*window_len)+1]

        """STEP 3: Plot the model posteriors"""
        legend_labels = self.results[self.names.index("model labels")]
        legend_handles = []
        #figure = plt.figure()

        #"""get time range s.t. it is in datetime format"""
        #ax.xaxis.set_major_formatter('%Y-%m')
        #date_axis = False
        if (not all_dates is None):
            x_axis = all_dates #drange(start, stop, delta) #debug: need delta as input
            start, stop = mdates.date2num(all_dates[0]), mdates.date2num(all_dates[-1])
            #date_axis = True
        else:
            x_axis = np.linspace(start_plot, stop_plot, len(time_range))
            all_dates = x_axis #debug



        if plot_type == "trace":
            count = 0
            for m in range(0,M):
                if m in indices:
                    handle, =ax.plot(x_axis[:up_to], model_posterior[m,:up_to],
                                  color=custom_colors[count],
                                  linestyle = custom_linestyles[count],
                                  linewidth = custom_linewidths[count])
                    legend_handles.append(handle)
                    count = count+1
        elif plot_type == "MAP":
            MAPs = np.argmax(model_posterior[indices,:], axis=0)+1
            handle = ax.plot(x_axis[:up_to], MAPs[:up_to], linewidth = 3.0,
                             color = custom_colors[0])
            tick_num = len(indices)
            major_ticks = np.arange(1, tick_num+1, 1, dtype = int).tolist()
            ax.set_yticks(major_ticks)
            if not y_axis_labels is None:
                ax.set_yticklabels(y_axis_labels)
            else:
                ax.set_yticklabels(major_ticks)
        elif (plot_type == "MAPVariance1_trace" or
              plot_type == "MAPVariance1_det"): #MAPVariance1_det
            """Plot map variance by considering variance about the
                each model posterior probability over a window of fixed
                size and and summing it up"""
            if window_len is None:
                window_len = 10
            map_variances = np.zeros((len(indices), len(time_range)))
            map_cov_dets = np.zeros(len(time_range))
            eps = 0.05
            """for first obs. - window_len"""
            for t in range(0, window_len):
                map_variances[:,t] = np.var(model_posterior[indices,:(t+1)].
                             reshape(len(indices), t+1), axis=1)
                if plot_type == "MAPVariance1_det":
                    minval = max(t+1, len(indices)+1)
                    covs = np.cov(
                            model_posterior[indices,:minval])
                    deleted_indices = np.all(np.abs(covs) > eps, axis=1)
                    covs = NPD.nearestPD(covs[~deleted_indices][:,~deleted_indices])
                    sign, ldet = np.linalg.slogdet(covs)
                    map_cov_dets[t] = sign*np.exp(ldet)
                    if SGV:
                        map_cov_dets[t] = pow(map_cov_dets[t], 1/covs.shape[0])
            """for the remainder"""
            for t in range(window_len, len(time_range)):
                map_variances[:,t] = np.var(model_posterior[indices,
                    (t-window_len):t], axis=1)
                if plot_type == "MAPVariance1_det":
                    covs = np.cov(
                            model_posterior[indices,(t-window_len):t])
                    deleted_indices = np.all(np.abs(covs) > eps, axis=1)
                    covs = NPD.nearestPD(covs[~deleted_indices][:,~deleted_indices])
                    sign, ldet = np.linalg.slogdet(covs)
                    map_cov_dets[t] = sign*np.exp(ldet)
                    if SGV:
                        map_cov_dets[t] = pow(map_cov_dets[t], 1/covs.shape[0])
            """sum up over the rows"""
            map_var = np.sum(map_variances, axis = 0)
            if plot_type == "MAPVariance1_trace":
                handle = ax.plot(x_axis[:up_to], map_var[:up_to], linewidth = 3.0,
                             color = custom_colors[0])
            elif plot_type == "MAPVariance1_det":
                #det exponentiated with 1/p, p = dimension. Done for standardizing
                if log_det:
                    map_cov_dets = np.log(map_cov_dets)
                handle = ax.plot(x_axis[:up_to], map_cov_dets[:up_to], linewidth = 3.0,
                             color = custom_colors[0])
        elif (plot_type == "MAPVariance2_trace" or
              plot_type == "MAPVariance2_det"):
            """Plot map variance by considering variance about the
                each model posterior probability when seeing it as a
                multinomial, over a window of fixed
                size and and summing it up."""

            MAPs = np.argmax(model_posterior[indices,:], axis=0)
            if window_len is None:
                window_len = 10
            MVN_variance = np.zeros(len(time_range))
            MVN_cov_dets = np.zeros(len(time_range))
            diag_ind = np.diag_indices(len(indices))
            """for first obs. - window_len"""
            for t in range(0, window_len):
                """STEP 1: Calculate frequencies"""
                frequencies = np.array([collections.Counter(MAPs[:(t+1)])[i]/(t+1)
                        for i in range(0, len(indices))])
                """STEP 2: Calcuate MVN vars from that"""
                MVN_variance[t] = np.sum([f*(1-f)*(t+1)
                            for f in frequencies])
                """STEP 3: calculate covariances (MVN off-diagonals)"""
                if plot_type == "MAPVariance2_det":
                    covs = (t+1)* np.outer(-frequencies, frequencies)
                    covs[diag_ind] = MVN_variance[t]
                    deleted_indices = np.all(covs == 0, axis=1)
                    covs = covs[~deleted_indices][:,~deleted_indices]
                    MVN_cov_dets[t] = np.linalg.det(covs)
                    if SGV:
                        MVN_cov_dets[t] = pow(MVN_cov_dets[t], covs.shape[0])
            for t in range(window_len, len(time_range)):
                """STEP 1: Calculate frequencies"""
                frequencies = np.array([collections.Counter(
                        MAPs[(t-window_len):t])[i]/window_len
                        for i in range(0, len(indices))])
                """STEP 2: Calcuate MVN vars from that"""
                MVN_variance[t] = np.sum([f*(1-f)*window_len
                            for f in frequencies])
                """STEP 3: calculate covariances (MVN off-diagonals)"""
                if plot_type == "MAPVariance2_det":
                    covs = window_len* np.outer(-frequencies, frequencies)
                    covs[diag_ind] = MVN_variance[t]
                    deleted_indices = np.all(covs == 0, axis=1)
                    covs = covs[~deleted_indices][:,~deleted_indices] #remove all 0-rows/cols
                    MVN_cov_dets[t] = np.linalg.det(covs)
                    if SGV:
                        MVN_cov_dets[t] = pow(MVN_cov_dets[t], covs.shape[0])
            """Plot"""
            if plot_type == "MAPVariance2_trace":
                handle = ax.plot(x_axis[:up_to], MVN_variance[:up_to],
                                 linewidth = 3.0,
                             color = custom_colors[0])
            elif plot_type == "MAPVariance2_det":
                if log_det:
                    MVN_cov_dets = np.log(MVN_cov_dets)
                handle = ax.plot(x_axis[:up_to], MVN_cov_dets[:up_to],
                                 linewidth = 3.0,
                             color = custom_colors[0])
        elif plot_type == "BF":
            """Plot Bayes Factors, hopefully we have only two models :D """
            """Assume equal prior"""
            if not log_format:
                BF = model_posterior[indices[0],:up_to]/model_posterior[indices[1],:up_to]
            else:
                BF = model_posterior[indices[0],:up_to] - model_posterior[indices[1],:up_to]

            #If we want to mark out the +/-5 parts
            if False:
                for i in range(0, len(BF)-1):
                    e = BF[i]
                    if abs(e) >= 5.0:
                        ax.plot([x_axis[i], x_axis[i+1]], [BF[i], BF[i+1]],
                                 linewidth = 3.0,color='green')
                    else:
                        ax.plot([x_axis[i], x_axis[i+1]], [BF[i], BF[i+1]],
                                 linewidth = 3.0, color='aqua')
            if True:
                handle = ax.plot(x_axis[up_to], BF, linewidth = 3.0,
                                 color = custom_colors[0])
                #gray shading
                ax.fill_between(x = [x_axis[0], x_axis[-1]],
                                y1 = [5, 5], y2 = [-5,-5],
                                color = "gray", alpha = 0.5)


        """STEP 4: Plot CPs if warranted"""
        if show_real_CPs:
            CP_legend_labels = []
            CP_legend_handles = []
            for  (CP_loc, CP_ind, CP_lab) in zip(self.true_CP_location,
                 self.true_CP_model_index, self.true_CP_model_label):
                if CP_loc >=start and CP_loc < stop:
                    handle = ax.axvline(x=CP_loc, color = self.CP_color,
                            linestyle = self.linestyle[CP_ind])
                    CP_legend_handles.append(handle)
                    CP_legend_labels.append(CP_lab)
            legend_handles+=CP_legend_handles
            legend_labels +=CP_legend_labels

        T_ = T  #DEBUG: Fix this once we plot model posterior for time-models
        if show_MAP_CPs:

            if up_to == np.size(time_range):
                #i.e., we have not specified up_to in the input
                CP_object = self.results[self.names.index("MAP CPs")][-2]
            else:
                if (len(self.results[self.names.index("MAP CPs")][up_to]) == 0
                  and
                  up_to < len(self.results[self.names.index("MAP CPs")]) - 2):
                    #get the first entry which is not empty if up_to entry is 0
                    count = up_to
                    bool_ = True
                    while bool_:
                        count = count + 1
                        if len(self.results[
                                self.names.index("MAP CPs")][count]) > 0:
                            bool_ = False
                    CP_object = self.results[self.names.index("MAP CPs")][count]
                elif (up_to >= len(self.results[
                        self.names.index("MAP CPs")]) - 2):
                    #we have a too large value for up_to
                    CP_object = self.results[self.names.index("MAP CPs")][-2]
                else:
                    #our value of up_to is in range
                    CP_object = self.results[self.names.index("MAP CPs")][up_to]

            #CP_object = self.results[self.names.index("MAP CPs")][-2]
            CP_locations = [entry[0] for entry in CP_object]
            CP_indices = [entry[1] for entry in CP_object]
            model_labels = self.results[self.names.index("model labels")]
            CP_legend_labels = []
            CP_legend_handles = []
            CP_indices_until_now = []
            count = 0
            for  (CP_loc, CP_ind) in zip(CP_locations, CP_indices):
                if CP_loc <= time_range[-1] and CP_loc >= time_range[0]:
                    CP_loc = ((CP_loc - time_range[0])/T_)*(stop-start) + start# carry CP forward
                    handle = ax.axvline(x=CP_loc, color = self.CP_color,
                            linestyle = self.linestyle[count])
                    if CP_ind not in CP_indices_until_now:
                        CP_legend_handles.append(handle)
                        CP_legend_labels.append(model_labels[CP_ind])
                        CP_indices_until_now.append(CP_ind)
                        count= count+1

        if not true_CPs is None:
            #true_CPs = [[location, color]]
            for entry in true_CPs:
                ax.axvline(x = entry[0], color = entry[1],
                                     linestyle = "-", linewidth = entry[2])
        """Annotations if wanted"""
        #Offset needs to be datetime object if we input datetime objects!
        if not period_time_list is None and not label_list is None:
            if plot_type == "MAP":
                ypos = len(indices)+0.2
                #if isinstance(number_offset, datetime.timedelta):
                #text_offset = 1.0
                #else:
                text_offset = number_offset + 0.2
            elif plot_type == "trace":
                ypos = 1+0.05
                text_offset = 0.25
            elif plot_type == "MAPVariance1_trace":
                ypos = np.max(map_var)*1.05
                text_offset = np.max(map_var)*0.1
            elif plot_type == "MAPVariance1_det":
                ypos = np.max(map_cov_dets)*1.05
                text_offset = np.max(map_cov_dets)*0.1
            elif plot_type == "MAPVariance2_trace":
                ypos = np.max(MVN_variance)*1.05
                text_offset = np.max(MVN_variance)*0.1
            elif plot_type == "MAPVariance2_det":
                ypos = np.max(MVN_cov_dets)*1.05
                text_offset = np.max(MVN_cov_dets)*0.1
            for period, label in zip(period_time_list, label_list):
                start_period, stop_period = period[0], period[1]

                """annotate the period"""
                ax.annotate("",
                    xytext=(start_period, ypos),
                        xycoords='data',
                    xy=(stop_period, ypos),
                        textcoords='data',
                    arrowprops=dict(arrowstyle="|-|",
                                    connectionstyle="arc3",
                                    linewidth = period_line_thickness,
                                    linestyle = "solid",
                                    color = "dimgray"),
                    )

                """annotate the label"""
                ax.annotate(label, xytext=(stop_period + number_offset, ypos - text_offset),
                        xycoords='data',
                    xy=(stop_period + number_offset, ypos - text_offset),
                        textcoords='data', fontsize = number_fontsize,
                        color = "dimgray")

            #debug
            stop_period = all_dates[-1]
            ax.annotate(label, xytext=(stop_period + number_offset,
                                         ypos - text_offset),
                        xycoords='data',
                    xy=(stop_period + number_offset, ypos - text_offset),
                        textcoords='data', fontsize = number_fontsize,
                        color = "dimgray")

#                if not event_time_list is None and not label_list is None:
#            if arrow_colors is None:
#                arrow_colors = ['black']*len(event_time_list)
#            count = 0
#            for event, label in zip(event_time_list, label_list):
#                ax.annotate(label, fontsize=number_fontsize, xy=(event, 1.2),
#                    xycoords='data', xytext=(event, -arrow_length),
#                    textcoords='data',
#                    arrowprops=dict(arrowstyle="->",
#                                    linewidth = arrow_thickness,
#                                    color = arrow_colors[count])
#                    )
#                count = count + 1

        """STEP 5: Plot legend & picture"""
        ax.set_aspect(aspect)
        if not xlab is None:
            ax.set_xlabel(xlab, fontsize = xlab_fontsize)
        if not xticks_fontsize is None:
            ax.tick_params(axis='x', labelsize=xticks_fontsize) #, rotation=90)
        if not yticks_fontsize is None:
            ax.tick_params(axis='y', labelsize=yticks_fontsize) #, rotation=90)
        if not ylab is None:
            ax.set_ylabel(ylab,  fontsize = ylab_fontsize)
        if not ylabel_coords is None:
            ax.get_yaxis().set_label_coords(ylabel_coords[0], ylabel_coords[1])

        #set x/ylims
        if not set_xlims is None:
            ax.set_xlim(set_xlims[0], set_xlims[1])
        if not set_ylims is None:
            ax.set_ylim(set_ylims[0], set_ylims[1])

        if legend:
            #plt.legend(handles=[blue_line])
            ax.legend(legend_handles, legend_labels, loc = "upper left")
        return ax


    """PLOT V: plot the run-length distribution for each time point, either
                 in log-format or in actual size.
        Options:
            time_range  => range of time over which we should plot the TS
            print_plt   => boolean, decides whether we want to see the plot
                           or just create the object to pass to the next fct.
            show_MAP_CPs    => bool indicating whether or not the MAP CPs should be
                           included in the plot
            show_real_CPs   => bool indicating whether or not the true CPs
                                should be included in the plot
            mark_median     => bool indicating if we want to mark the median
                               of the r-l distr
            log_format      => bool indicating if we want to display in log
                                format or not

    """
    def plot_run_length_distr(self, print_plt = True, time_range = None,
                              show_MAP_CPs = True, show_real_CPs = False,
                              mark_median = False,
                              mark_max = False,
                              log_format = True,
                              CP_legend = False,
                              CP_custom_legend_labels = None,
                              CP_exclude_indices = [],
                              CP_legend_fontsize = 10,
                              CP_transparence = 1.0,
                              buffer = 50,
                              upper_limit = None,
                              enforce_upper_limit = True,
                              print_colorbar = True,
                              orientation = "horizontal",
                              C1 = 0.0,
                              C2 = 1.0,
                              start=None,
                              stop=None,
                              all_dates = None,
                              event_time_list= None,
                              label_list = None,
                              custom_colors = None,
                              custom_linestyles = None,
                              aspect_ratio = 'auto',
                              xlab = 'Time',
                              ylab = 'run length',
                              ax = None, figure = None,
                              space_to_colorbar = 0.05,
                              colorbar_location = "top",
                              arrow_colors = None,
                              custom_linewidth = None,
                              arrow_length = 30,
                              arrow_thickness = 2.0,
                              number_fontsize = 10,
                              xlab_fontsize = 10,
                              ylab_fontsize = 10,
                              no_transform = False,
                              date_instructions_formatter = None,
                              date_instructions_locator = None,
                              ylabel_coords = None,
                              colorbar_ticks_num = None,
                              additional_legend_labels = [],
                              additional_legend_colors = [],
                              arrows_setleft_indices = None,
                              arrows_setleft_by = None,
                              zero_distance = None,
                              xticks_fontsize = 10,
                              yticks_fontsize = 10,
                              arrow_distance = None,
                              mark_max_linestyle = None,
                              mark_max_linewidth = None,
                              mark_max_color = None,
                              set_xlims = None,
                              set_ylims = None,
                              up_to = None
                              ):
        """plot the run-length distro, potentially inserting the MAP CPs or
        the real CPs. You can also trace the median of the distribution (which
        always needs to be computed in the log-format). Upper_limit gives you
        an r_max that you impose, i.e. you do not plot any r.l. larger than
        upper_limit."""



        """STEP 1: Default is to take the entire time range"""
        T = self.results[self.names.index("T")]
        storage_run_length_log_distr = self.results[self.names.index(
                "all run length log distributions")]
        offset = T - len(storage_run_length_log_distr)
        if time_range is None:
            time_range = np.linspace(offset,T,T-offset, dtype=int)
        elif time_range[0]<offset:
            #T_ = np.size(time_range) - time_range[0]
            time_range = np.linspace(offset, time_range[-1],
                                time_range[-1] - offset, dtype=int )

        """"new axis if needed"""
        if ax is None or figure is None:
            figure, ax = plt.subplots()
        """colorbar if needed"""
        divider = make_axes_locatable(ax)
        if orientation == "horizontal" and colorbar_location is not None:
            cax = divider.append_axes(colorbar_location, size = '5%', pad = space_to_colorbar)
        elif colorbar_location is not None:
            cax = divider.append_axes('right', size = '1%', pad = space_to_colorbar)

        """If we do not need to rescale"""
        if start is None or stop is None:
            start = time_range[0]
            stop = time_range[-1]

        """STEP 2: We need to get the maximum run-length to create 'pdfs' with
        the right dimensions in the next step"""
        r_max = 0
        storage_all_retained_run_lengths = self.results[self.names.index(
                "all retained run lenghts")]
        for run_lengths in storage_all_retained_run_lengths:
            r_max = max(r_max, np.max(run_lengths))
            """if we have upper limit, enforce"""
            if (not upper_limit is None) and (upper_limit > 0):
                r_max = max(int(upper_limit), r_max)
        """If RLs in log format, you may sometimes have to enforce the
        r_max you input manually, since comparisons/max don't work on that
        minute numerical scale"""
        if enforce_upper_limit and upper_limit is not None:
            r_max = upper_limit
        print("rmax = ", r_max)

        """STEP 3: We need to retrieve and appropriately transform the log
        run length distribution!"""

        T_ = np.size(time_range) #T_ = T if time_range None
        if log_format:
            #pdfs = np.zeros((T,T))
            pdfs = -np.inf * np.ones((r_max + buffer,T_))
        else:
            #pdfs = np.zeros((T,T))
            pdfs = np.zeros((r_max + buffer, T_))
        median = np.zeros(T_)
        maxima = np.zeros(T_)

        #set up_to to T_ if none
        if up_to is None:
            up_to = T_


        """STEP 4: Next, create the log-pdf for the run-length distro"""
        #r_max = 0
        #T_rl = int(len(storage_run_length_log_distr))
        minval, maxval = np.inf, -np.inf
        for t, ind in zip(time_range-1, range(0, up_to)): #range(offset, T_rl):

            """STEP 4.1: Retrieve the log rl distro and convert into proper
            distro if needed and get the maximum non-zero run length"""
            run_length_distr = np.array(storage_run_length_log_distr[t])
            run_length_distr_copy= np.array(storage_run_length_log_distr[t]).copy()
            retained_run_lengths = storage_all_retained_run_lengths[t]

            """STEP 4.1.1: If we want to only show the most recent r_max r.l.s,
            chop off the rest of the rld"""
            if (not upper_limit is None) and (upper_limit > 0):
                upper_limit = int(upper_limit)
                trimmed_rl = retained_run_lengths[
                        np.where(retained_run_lengths < upper_limit)]
                run_length_distr = run_length_distr[
                        np.where(retained_run_lengths < upper_limit)]
                retained_run_lengths = trimmed_rl


#            print(retained_run_lengths)
#            print(t)
#            print(r_max)
#            print(r_max + buffer)
            #if not log_format:
            #    run_length_distr = np.log(multiplicative * np.power(run_length_distr, exponential))
            #r_max = max(r_max, [ n for n,i in enumerate(run_length_distr)
            #    if i> -np.inf ][-1])

            """STEP 4.2: For each run-length, get the cdf for that r.l. and
            compute the index of the median for that r.l. distro"""
            if log_format:
                """STEP 4.2A: If we want the log cdf, use log sum exponential"""
                #pdfs[:np.size(run_length_distr),t] = np.exp(run_length_distr)
#                print("ind", ind)
#                print("size retained run lengths", retained_run_lengths.shape)
#                print("run_length_distr", run_length_distr.shape)
                if no_transform:
                    pdfs[retained_run_lengths,ind] = np.exp(run_length_distr)
                else:
                    pdfs[retained_run_lengths,ind] = C2*np.log(np.exp(run_length_distr  + C1))
            else:
                """STEP 4.2B: If we do not want the log format, take the
                exponential and cumulative sum"""
                #pdfs[:,t] = np.cumsum(np.exp(run_length_distr))
                #pdfs[:np.size(run_length_distr),t] = (np.exp(run_length_distr))
                pdfs[retained_run_lengths,ind] = np.exp(run_length_distr + C1)

            notinfindices = np.logical_and(
                    np.greater(pdfs[retained_run_lengths,ind], -np.inf),
                    np.less(pdfs[retained_run_lengths,ind],np.inf))
            if not np.sum(notinfindices) == 0:
                notinf = pdfs[retained_run_lengths,ind][notinfindices]
                minval = min(minval, np.nanmin(notinf))
                maxval = max(maxval, np.nanmax(notinf))

#            """STEP 4.3: Compute the median for the run-length (always use
#            the log-format for this!)"""
#            if not log_format:
#                run_length_distr = np.array(storage_run_length_log_distr[t])
#            """STEP 4.3.1: Compute the median for all non-zero (i.e., non -inf
#            entries in the run length posterior)"""
#
            if mark_max:
                maxima[ind] = np.argmax(pdfs[:,ind])#run_length_distr_copy)
            if mark_median:
                med = np.median( run_length_distr_copy[np.where(run_length_distr_copy
                                                       > -np.inf)])
                median[ind] = retained_run_lengths[
                       np.nanargmin(np.abs(run_length_distr_copy-med))]

        #"""STEP 5: Plot the run-length distro and its median"""
        #figure = plt.figure()
        #ax = figure.add_subplot(111)
        date_axis = False
        if (not all_dates is None):
            x_axis = all_dates #drange(start, stop, delta) #debug: need delta as input
            start, stop = mdates.date2num(all_dates[0]), mdates.date2num(all_dates[-1])
            date_axis = True
        else:
            x_axis = np.linspace(start, stop, T_)
        if mark_median:
            ax.plot(x_axis[:up_to], #np.linspace(start,stop,T_),
                     (median)[:up_to], color = self.median_color, #linewidth = 1,
                     linestyle = (0, (3,1,1,1)), linewidth = 2.0)
        if mark_max:
            if mark_max_linewidth is None:
                mark_max_linewidth = 2.0
            if mark_max_linestyle is None:
                mark_max_linestyle = (0, (3,1,1,1))
            if mark_max_color is None:
                mark_max_color = self.max_color
            ax.plot(x_axis[:up_to], #np.linspace(start,stop,T_),
                     (maxima)[:up_to], color = mark_max_color, #linewidth = 1,
                     linestyle = mark_max_linestyle,
                     linewidth = mark_max_linewidth)
        if date_axis:
            if date_instructions_formatter is None or date_instructions_locator is None:
                ax.xaxis_date()
#            else:
#                ax.xaxis_date()
#                ax.xaxis.set_major_locator(date_instructions_locator)
#                ax.xaxis.set_major_formatter(date_instructions_formatter)
#                ax.format_xdata = mdates.DateFormatter('%Y')
#                figure.autofmt_xdate()

        #pdfs = pdfs[:r_max + buffer, :]
        im = ax.imshow(pdfs, #extent=(0,T, T,0),#np.amax(r_max + self.cushion),0 ),#,
                        interpolation = None,
                        cmap='gray_r',
                        norm=LogNorm(),
                        aspect = aspect_ratio,
                        extent = (start, stop, r_max, 0))


        if CP_transparence is None:
            CP_transparence = 1.0

        """STEP 6: Plot real (or MAP) CPs if wanted"""
        """STEP 6A: Plot the MAP CPs stored in *results*"""
        if show_MAP_CPs:

            if up_to == T_:
                #i.e., we have not specified up_to in the input
                CP_object = self.results[self.names.index("MAP CPs")][-2]
            else:
                if (len(self.results[self.names.index("MAP CPs")][up_to]) == 0
                  and
                  up_to < len(self.results[self.names.index("MAP CPs")]) - 2):
                    #get the first entry which is not empty if up_to entry is 0
                    count = up_to
                    bool_ = True
                    while bool_:
                        count = count + 1
                        if len(self.results[
                                self.names.index("MAP CPs")][count]) > 0:
                            bool_ = False
                    CP_object = self.results[self.names.index("MAP CPs")][count]
                elif (up_to >= len(self.results[
                        self.names.index("MAP CPs")]) - 2):
                    #we have a too large value for up_to
                    CP_object = self.results[self.names.index("MAP CPs")][-2]
                else:
                    #our value of up_to is in range
                    CP_object = self.results[self.names.index("MAP CPs")][up_to]

            #CP_object = self.results[self.names.index("MAP CPs")][-2]
            CP_locations = [entry[0] for entry in CP_object]
            CP_indices = [entry[1] for entry in CP_object]
            model_labels = self.results[self.names.index("model labels")]
            """if no custom color, take standard"""
            if custom_colors is None:
                custom_colors = [self.CP_color]*len(CP_locations)
            if custom_linestyles is None:
                custom_linestyles = self.linestyle #['solid']*len(CP_locations)
            if custom_linewidth is None:
                custom_linewidth = 3.0

            CP_legend_labels = []
            CP_legend_handles = []
            CP_indices_until_now = []
            count = 0

            """Loop over the models in order s.t. you can color in the same
            fashion as for the model posterior"""
            M = len(self.results[self.names.index("model labels")])
            for m in range(0, M):
                for  (CP_loc, CP_ind) in zip(CP_locations, CP_indices):
                    if m == CP_ind and m not in CP_exclude_indices:
                        if CP_loc <= time_range[-1] and CP_loc >= time_range[0]:
                            CP_loc = ((CP_loc - time_range[0])/T_)*(stop-start) + start# carry CP forward
                            if CP_ind not in CP_indices_until_now:
                                handle = ax.axvline(x=CP_loc, color = custom_colors[count],
                                    linestyle = custom_linestyles[count],
                                    linewidth = custom_linewidth,
                                    alpha = CP_transparence)
                                CP_legend_handles.append(handle)
                                CP_legend_labels.append(model_labels[CP_ind])
                                CP_indices_until_now.append(CP_ind)
                                count= count+1
                            elif CP_ind in CP_indices_until_now:
                                """display it in the same color"""
                                relevant_index = CP_indices_until_now.index(CP_ind)
                                handle = ax.axvline(x=CP_loc, color = custom_colors[relevant_index],
                                    linestyle = custom_linestyles[relevant_index],
                                    linewidth = custom_linewidth,
                                    alpha = CP_transparence)

        """STEP 7: Plot the true CPs stored directly in EvaluationTool*"""
        if show_real_CPs:
            CP_legend_labels = []
            CP_legend_handles = []
            for  (CP_loc, CP_ind, CP_lab) in zip(self.true_CP_location,
                 self.true_CP_model_index, self.true_CP_model_label):
                if CP_loc <= time_range[-1] and CP_loc >= time_range[0]:
                    CP_loc = ((CP_loc - time_range[0])/T_)*(stop-start) + start#-time_range[0] #carry CP forward
                    handle = ax.axvline(x=CP_loc, color = custom_colors[CP_ind],
                            linestyle = self.linestyle[CP_ind])
                    CP_legend_handles.append(handle)
                    CP_legend_labels.append(CP_lab)

        """STEP 8: If we want a legend for the models corr. to the CPs"""
        if CP_legend:
            if not CP_custom_legend_labels is None:
                CP_legend_labels = CP_custom_legend_labels
#            if not additional_legend is None:
#                #additional_entries = []
                #additional_labels = []
#                for entry in additional_legend:
##                    fake_handle = ax.axvline(np.array([]),
##                                    color = entry[1],
##                                    linestyle = custom_linestyles[count-1],
##                                    linewidth = custom_linewidth)
##                    CP_legend_handles.append(fake_handle)#,
##                              #markersize=15))
##                    CP_legend_labels.append(entry[0])
#                    #Get artists and labels for legend and chose which ones to display
#                    handles, labels = ax.get_legend_handles_labels()
#                    display = (0,1,2)
#
#                    #Create custom artists
#                    simArtist = plt.Line2D((0,1),(0,0), color='k', marker='o', linestyle='')
#                    anyArtist = plt.Line2D((0,1),(0,0), color='k')
#
#                    #Create legend from custom artist/label lists
#                    ax.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
#                              [label for i,label in enumerate(labels) if i in display]+['Simulation', 'Analytic'])

                #CP_legend.handles.append
            additional_handles= []
            for color in additional_legend_colors:
                additional_handles.append(plt.Line2D((0,1),(0,0),
                                    color=color, linewidth = custom_linewidth))#, marker='-',
                                    #linestyle=''))
            handles, labels = CP_legend_handles, CP_legend_labels
            display = (0,1,2)
            ax.legend([handle for i,handle in enumerate(handles) if i in display]
                    +additional_handles,
                [label for i,label in enumerate(labels) if i in display]
                    + additional_legend_labels,
                loc = 'lower left', prop = {'size':CP_legend_fontsize})
#            ax.legend(CP_legend_handles, CP_legend_labels, loc = 'lower left',
#                      prop = {'size':CP_legend_fontsize})

        """STEP 9: Print if needed and return"""
        if print_colorbar:
            #print(minval)
            #print(maxval)
            #minval, maxval = -C2*np.exp(-80), maxval #C2 * np.exp(C1)
            #convert_min = minval/C2 - C1
            #convert_max = maxval/C2 - C1
            colbar = figure.colorbar(im, cax = cax, orientation=orientation,
                                     spacing = 'proportional')
            #print("Here2!")

            #DEBUG
            #colbar.set_ticks([0, 0.001, 0.01, 1])

            """Only rescale this if C1, C2 rescaled the log data"""
            if log_format and (C1 != 0.0 or C2 !=1.0):
                minval = 0.0075
                if colorbar_ticks_num is None:
                    theticks = [minval,
                                 minval + (maxval-minval)*pow(10,-4),
                                 minval + (maxval-minval)*5*pow(10,-3),
                                 minval + (maxval-minval)*pow(10,-1),
                                 maxval]
                else:
                    theticks = [minval,
                                 #minval + (maxval-minval)*pow(10,-4),
                                 minval + (maxval-minval)*5*pow(10,-3),
                                 minval + (maxval-minval)*pow(10,-1),
                                 maxval]

                colbar.set_ticks(theticks)#pow(10,1), pow(10,2), pow(10,3), pow(10,4), pow(10,5)])  #this color bar gives us the gradient of the rl distro
                 #rescaled + make next statement dependent on horizontal vs vertical
                labels =['-1000']
                for tick in theticks[1:]:
                    labels.append(str(int(tick/C2-C1)))
                colbar.ax.set_xticklabels(labels)

        if not event_time_list is None and not label_list is None:
            if arrow_colors is None:
                arrow_colors = ['black']*len(event_time_list)
            count = 0
            count_setlefts = 0
            for event, label in zip(event_time_list, label_list):


                if ((not arrows_setleft_indices is None) and
                    (not arrows_setleft_by is None) and
                    (not zero_distance is None) and
                    count_setlefts < len(arrows_setleft_indices)):
                    shifter = zero_distance #store 0 in here or datetime-delta of 0
                    if count == arrows_setleft_indices[count_setlefts]:
                        shifter = arrows_setleft_by[count_setlefts]
                        count_setlefts = count_setlefts+1
                    event_ = event - shifter
                else:
                    event_ = event
                if arrow_distance is None:
                    arrow_distance = 2


                ax.annotate(label, fontsize=number_fontsize, xy=(event, arrow_distance),
                    xycoords='data', xytext=(event_, -arrow_length),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    linewidth = arrow_thickness,
                                    color = arrow_colors[count])
                    )
                count = count + 1
#        ax.annotate('local max', xy=(2000, 2),  xycoords='data',
#            xytext=(0.8, 0.95), textcoords='axes fraction',
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            horizontalalignment='right', verticalalignment='top',
#            )
        if not xlab is None:
            ax.set_xlabel(xlab, fontsize = xlab_fontsize)
        if not ylab is None:
            ax.set_ylabel(ylab,  fontsize = ylab_fontsize)
        if not xticks_fontsize is None:
            ax.tick_params(axis='x', labelsize=xticks_fontsize) #, rotation=90)
        if not yticks_fontsize is None:
            ax.tick_params(axis='y', labelsize=yticks_fontsize) #, rotation=90)
        if not ylabel_coords is None:
            ax.get_yaxis().set_label_coords(ylabel_coords[0], ylabel_coords[1])
#        if print_plt:
#            plt.show()

        #set x/ylims
        if not set_xlims is None:
            ax.set_xlim(set_xlims[0], set_xlims[1])
        if not set_ylims is None:
            ax.set_ylim(set_ylims[0], set_ylims[1])

        return ax, figure


    """PLOT VI: plot the model-and-run-length distributions for each time point,
                and for each model. Do this either in log-format or in actual
                size.
        Options:
            time_range  => range of time over which we should plot the TS
            print_plt   => boolean, decides whether we want to see the plot
                           or just create the object to pass to the next fct.
            show_MAP_CPs    => bool indicating whether or not the MAP CPs should be
                           included in the plot
            show_real_CPs   => bool indicating whether or not the true CPs
                                should be included in the plot
            mark_median     => bool indicating if we want to mark the median
                               of the r-l distr
            log_format      => bool indicating if we want to display in log
                                format or not

    """
    def plot_model_and_run_length_distr(self, print_plt = True, time_range = None,
                              show_MAP_CPs = True, show_real_CPs = False,
                              mark_median = True, log_format = True,
                              CP_legend = False, buffer = 50):
        """plot the run-length distro, potentially inserting the MAP CPs or
        the real CPs. You can also trace the median of the distribution (which
        always needs to be computed in the log-format)"""

        """STEP 1: Default is to take the entire time range"""
        T = self.results[self.names.index("T")]
        if time_range is None:
            time_range = np.linspace(1,T,T)

        """STEP 2: We need to get the maximum run-length to create 'pdfs' with
        the right dimensions in the next step"""
        r_max = 0
        storage_all_retained_run_lengths = self.results[self.names.index(
                "all retained run lenghts")]
        for run_lengths in storage_all_retained_run_lengths:
            r_max = max(r_max, np.max(run_lengths))

        """STEP 2: We need to retrieve and appropriately transform the model
        and run length log distribution!"""
        storage_model_and_run_length_log_distr = self.results[self.names.index(
                "all model and run length log distributions")]
        T_ = np.size(time_range) #T_ = T if time_range None
        offset = T_ - len(storage_model_and_run_length_log_distr)
        M = len(storage_model_and_run_length_log_distr[-1][:,0])
        if log_format:
            #cdfs = (-np.inf)*np.ones((M, T,T))
            #pdfs = (-np.inf)*np.ones((M, T,T))
            pdfs = [(-np.inf)*np.ones((r_max + buffer,T))]*M
        else:
            #cdfs = np.zeros((M, T,T))
            #pdfs = np.zeros((M, T,T))
            pdfs = [np.zeros((r_max + buffer,T))]*M
        median = np.zeros((M, T))


        """STEP 3: Next, create the log-cdf for the run-length distro"""
        #r_max_list = []
        for m in range(0, M):

            r_max = 0 #need one for each model, or need the same for all models
            T_rl = int(len(storage_model_and_run_length_log_distr))
            for t in range(offset, T_rl):

                """STEP 3.1: Retrieve the log rl distro and convert into proper
                distro if needed and get the maximum non-zero run length"""
                #print(storage_model_and_run_length_log_distr[t][:,:].shape[0])
                #print(m)
                """This condition ensures that the model was already initialized
                at time t, i.e. that the model_and_run_length log distr contains
                a row for that model"""
                if m<storage_model_and_run_length_log_distr[t][:,:].shape[0]:
                    run_length_distr = storage_model_and_run_length_log_distr[t][m,:]
                    retained_run_lenghts = storage_all_retained_run_lengths[t]
                    if not log_format:
                        run_length_distr = np.exp(run_length_distr)

                    """STEP 3.2: For each run-length, get the cdf for that r.l. and
                    compute the index of the median for that r.l. distro"""
                    if log_format:
                        pdfs[m][retained_run_lenghts,t] = np.exp(run_length_distr)
                    else:
                        """STEP 3.2B: If we do not want the log format, take the
                        exponential and cumulative sum"""
                        pdfs[m][retained_run_lenghts,t] = np.exp(run_length_distr)

                    """STEP 3.3: Compute the median for the run-length (always use
                    the log-format for this!)"""
                    if not log_format:
                        run_length_distr = storage_model_and_run_length_log_distr[t][m,:]
                    if mark_median:
                        """STEP 3.3.1: Compute the median for all non-zero (i.e., non -inf
                        entries in the run length posterior)"""
                        med = np.median( run_length_distr[np.where(run_length_distr
                                                       > -np.inf)])
                        #print("np.sum(np.isnan(med)) :", np.sum(np.isnan(med)))
                        median[m,t] = retained_run_lenghts[
                                np.nanargmin(np.abs(run_length_distr-med))]

        """STEP 4: Plot the run-length distro and its median"""
        figure = plt.figure()
        #plt.suptitle("myTitle")

        """For each model, add a sub-plot s.t. they share the X-axis. Suppress
        the x-axis labels unless it is the bottom-level model. Also plot the
        names of the models over the plots using suptitle("myTitle")"""
        for m in range(0,M):
            ax = plt.subplot(M,1,1+m)
            #plt.imshow(cdfs[m,:,:],
            #            cmap='gray_r', norm=LogNorm())
            #pdfs[m] = pdfs[m][:(r_max_list[m]+buffer),:]
            plt.imshow(pdfs[m], #pdfs[m,:,:],
                        cmap='gray_r', norm=LogNorm())
            if mark_median:
                plt.plot((median), color = self.median_color, linewidth = 3)
            plt.gca().axes.get_xaxis().set_visible(False)



            """STEP 5: Plot real (or MAP) CPs if wanted"""
            """STEP 5A: Plot the MAP CPs stored in *results*"""
            if show_MAP_CPs:
                CP_object = self.results[self.names.index("MAP CPs")][-2]
                CP_locations = [entry[0] for entry in CP_object]
                CP_indices = [entry[1] for entry in CP_object]
                model_labels = self.results[self.names.index("model labels")]
                CP_legend_labels = []
                CP_legend_handles = []
                for  (CP_loc, CP_ind) in zip(CP_locations, CP_indices):
                    handle = plt.axvline(x=CP_loc, color = self.CP_color,
                        linestyle = self.linestyle[CP_ind])
                    CP_legend_handles.append(handle)
                    CP_legend_labels.append(model_labels[CP_ind])

            """STEP 5A: Plot the true CPs stored directly in EvaluationTool*"""
            if show_real_CPs:
                CP_legend_labels = []
                CP_legend_handles = []
                for  (CP_loc, CP_ind, CP_lab) in zip(self.true_CP_location,
                     self.true_CP_model_index, self.true_CP_model_label):
                    handle = plt.axvline(x=CP_loc, color = self.CP_color,
                        linestyle = self.linestyle[CP_ind])
                    CP_legend_handles.append(handle)
                    CP_legend_labels.append(CP_lab)

            ax.set_title("model number " + str(m))
            #plt.colorbar()  #this color bar gives us the gradient of the rl distro


        """STEP 6: If we want a legend for the models corr. to the CPs"""
        if CP_legend:
            plt.legend(CP_legend_handles, CP_legend_labels, loc = 'lower left')

        """STEP 7: Print if needed and return"""
        if print_plt:
            plt.show()
        return figure




#if __name__ == "__main__":
#    #import matplotlib.pyplot as plt
#    from BVAR_NIG_Sim import BVARNIGSim
#    from BVAR_NIG import BVARNIG
#    from cp_probability_model import CpModel
#
#    """STEP 0: Def9ine overall params"""
#    S1, S2, T = 2,2, 1500
#    result_file = ("C:\\Users\\Jeremias\\Documents\\Studieren - " +
#                    "Inhaltliches\\OxWaSP PC backup\\Modules\\SpatialProject" +
#                    "\\Code\\SpatialBOCD\\Test_results_EvTool.txt")
#    run_algo = False
#
#    if run_algo:
#        """STEP 1: Run simulation"""
#        mySim = BVARNIGSim(S1=S1, S2=S2, T=T, CPs = 3, CP_locations = [400,700, 1100],
#                 sigma2 = np.array([1,1, 0.5, 1]),
#                 nbh_sequences = [ [0,0], [0,0], [0,0], [4,4] ],
#                 restriction_sequences = [[0,0], [0,0], [0,0], [4,4]],
#                 segment_types=["BVAR", "MGARCH", "VMA", "BVAR"],
#                 intercept_groupings = None,
#                 coefs =[ np.array([10, 0.6, -0.35]),
#                          np.array([10, 0, 0]),
#                          np.array([10,0,0]),
#                          np.array([0.8, 0.25, 0.05, 0.3, 0.025])],
#                 burn_in = 100,
#                 padding = "row_col_mean")
#
#        data = mySim.generate_all_segments()
#        plt.plot(np.linspace(1,data.shape[0], data.shape[0]), data[:,0,0])
#        plt.show()
#
#
#        """STEP 2: Define the models"""
#        myBVAR = BVARNIG(prior_a=2, prior_b=pow(10,3.5),
#                     prior_mean_beta=np.zeros(1+1),
#                     prior_var_beta=100* np.identity(1+1),
#                     S1 = S1, S2 = S2,
#                     intercept_grouping = None,
#                     nbh_sequence=np.array([0]),
#                     nbh_sequence_exo=np.array([0]),
#                     exo_selection = [],
#                     padding = 'overall_mean',
#                     auto_prior_update=False,
#                     restriction_sequence = np.array([0])
#                     )
#        myBVAR2 = BVARNIG(prior_a=2, prior_b=pow(10,3.5),
#                     prior_mean_beta=np.zeros(1+4),
#                     prior_var_beta=100* np.identity(1+4),
#                     S1 = S1, S2 = S2,
#                     intercept_grouping = None,
#                     nbh_sequence=np.array([4,4]),
#                     nbh_sequence_exo=np.array([0]),
#                     exo_selection = [],
#                     padding = 'overall_mean',
#                     auto_prior_update=False,
#                     restriction_sequence = np.array([4,4])
#                     )
#
#        """STEP 3: Put them in the detector and run algo"""
#        model_universe = np.array([myBVAR, myBVAR2])
#        model_prior = np.array([0.5, 0.5])
#        intensity = 50
#        cp_model = CpModel(intensity)
#
#        myDetector = Detector(data, model_universe, model_prior, cp_model,
#                 S1, S2, T, exo_data=None, num_exo_vars=None, threshold = -50,
#                 store_rl = True, store_mrl = True)
#        myDetector.run()


#    """STEP 3: Put the detector inside of an EvaluationTool object and try
#               some fcts. (read/write and plotting)"""
#    myTool = EvaluationTool()
#    myTool.build_EvaluationTool_via_run_detector( myDetector,
#            true_CP_location=[0, 400, 700, 1100], true_CP_model_index = [0, 1, 2,0],
#            true_CP_model_label = ["BVAR", "MGARCH", "VMA", "BVAR"])
#
#    #Test 1: Can we store results?
#    myTool.store_results_to_HD(result_file)
#
#    #play with rl distr
#
#    t_ = 400
#    #check what cdf looks like
#    rld = myTool.results[myTool.names.index("all run length log distributions")]
#    non_inf_rld = rld[t_][np.where(rld[t_] > (-np.inf))]
#    L = np.size(non_inf_rld)
#    myA = -np.inf * np.ones(L)
#    for i in range(0, L):
#        myA[i] = misc.logsumexp(non_inf_rld[:(i+1)])
#    plt.plot(np.linspace(1,L,L), myA)
#
#
    #Test 2: Can we read them into a (new) Evaluation tool object?
    #myTool2 = EvaluationTool()
    #myTool2.build_EvaluationTool_via_results(result_file)

    #Test 3: can we plot what we want? Are the plots the same with Tool2?
#    myFig = myTool.plot_raw_TS(data, indices=[0], print_plt=True, legend=True,
#                       legend_labels = None, legend_position=None,
#                       time_range=None)
#    myFig.savefig("raw series.pdf")
#    #Q: Why are the CPs not in myTool2?
#    myTool2.plot_raw_TS(indices=[0], print_plt=True, legend=False,
#                       legend_labels = None, legend_position=None,
#                       time_range=None)
#    myFig2 = myTool.plot_predictions(indices=[0], print_plt=True, legend=True,
#                    legend_labels = None, legend_position="upper right",
#                    time_range=None,show_var=True, show_CPs=True)
#    myFig2.savefig("predictions.pdf")
#    myTool2.plot_predictions(indices=[0], print_plt=True, legend=False,
#                    legend_labels = None, legend_position=None,
#                    time_range=None,show_var=True, show_CPs=True)
#    myFig3 = myTool.plot_run_length_distr(print_plt = True, time_range = None,
#                              show_MAP_CPs = False, show_real_CPs = False,
#                              mark_median = False, log_format = True,
#                              CP_legend = False)
#    myFig3.savefig("runlengthdistro.pdf")
#    myTool2.plot_run_length_distr(print_plt = True, time_range = None,
#                              show_MAP_CPs = True, show_real_CPs = False,
#                              mark_median = False, log_format = True,
#                              CP_legend = False)
#
#    #NOT DEBUGGED YET, not sure what is wrong here.
#    myTool.plot_model_and_run_length_distr(print_plt = True, time_range = None,
#                              show_MAP_CPs = True, show_real_CPs = False,
#                              mark_median = True, log_format = True,
#                              CP_legend = False)
#    myTool2.plot_model_and_run_length_distr(print_plt = True, time_range = None,
#                              show_MAP_CPs = True, show_real_CPs = False,
#                              mark_median = True, log_format = True,
#                              CP_legend = False)
#
    #Test 4:

#
#    """plot the data from segment"""
#    plt.plot(np.linspace(1,data.shape[0], data.shape[0]), data[:,0,0])
#    plt.show()
#    plt.plot(np.linspace(1,data.shape[0], data.shape[0]), data[:,1,0])
#    plt.show()
#    plt.plot(np.linspace(1,data.shape[0], data.shape[0]), data[:,0,1])
#    plt.show()
#    plt.plot(np.linspace(1,data.shape[0], data.shape[0]), data[:,1,1])
#    plt.show()
#
#
#
#
