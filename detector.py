# Python 3.5.2 |Anaconda 4.2.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited: 2017-09-13
Author: Jeremias Knoblauch (J.Knoblauch@warwick.ac.uk)
Forked by: Luke Shirley (L.Shirley@warwick.ac.uk)

Description: Implements the Detector, i.e. the key object in the Spatial BOCD
software. This object takes in the Data & its dimensions, the CP prior,
a set of probability models, the priors over this probability model collection,
and a model prior probability over each of the models in the collection.
"""
import numpy as np
import scipy
from scipy import misc
import time

from BVAR_NIG_DPD import BVARNIGDPD
from BVAR_NIG import BVARNIG

class Detector:
    """key object in the Spatial BOCD
    software. This object takes in the Data & its dimensions, a set of
    probability models and their priors, and a model prior probability
    over each of the models in the collection/universe, and a CP prior.

    Attributes:
        data: float numpy array;
            a SxS spatial lattice with T time points and thus of
            dimension SxSxT
        model_universe: ProbabilityModel numpy array;
            the collection of models
            that might fit a given segment. All children of the
            ProbabilityModel parent class
        model_prior: float numpy array;
            a prior vector giving the prior belief for each entry
            in the *model_universe* numpy array. If model_prior[i-1] = p,
            then we have prior belief p that the i-th model is the generating
            process for a given segment
        cp_model: CpModel object;
            the CP model, usually specified as a geometric distribution
            over the time points.
        Q: int;
            the number of entries in the model_universe numpy array
        T, S1, S2: int;
            the dimensions of the spatial lattice (S1, S2) and the time (T),
            provided that the data is not put in as data stream
        predictive_probs: float numpy array;
            stores the predictive probs at each time point for each lattice
            point and for each potential model
        growth_probs: float numpy array;
            stores the probability of the run lengths being r = 1,2,..., t at
            time point t.
        cp_probs: float numpy array;
            stores the probability of the run lengt being r=0 at time point t
        evidence: float numpy array;
            stores the evidence (posterior probability) of having observed
            all values of the data up until time point t at each time point t
        run_length_distr: float numpy array;
            stores the run-length distribution for r=0,1,...,t at each time
            point t.

    """

    def __init__(self, data, model_universe, model_prior, cp_model,
                 S1, S2, T, exo_data=None, num_exo_vars=None, threshold=None,
                 store_rl = False, store_mrl = False, trim_type = "keep_K",
                 notifications = 50,
                 save_performance_indicators=False,
                 training_period = 200,
                 generalized_bayes_rld = False, #'kullback_leibler', 'power_divergence'
                 alpha_rld = None,
                 alpha_rld_learning = False,
                 alpha_param_learning = False,
                 alpha_param = None,
                 #SGD-updating of alpha_param and alpha_rld in case we use DPD
                 loss_der_rld_learning = None,
                 loss_param_learning = None,
                 step_size_rld_learning = None,
                 step_size_param_learning = None,
                 eps_param_learning = None,
                 alpha_param_opt_t=100,
                 alpha_rld_opt_t = 100):
        #add arguments: save_negative_log_likelihood, training_period
        """construct the Detector with the multi-dimensional numpy array
        *data*. E.g., if you have a SxS spatial lattice with T time points,
        then *data* will be SxSxT. The argument *model_universe* will provide
        you with a numpy array of ProbabilityModel objects, each of which is
        associated with a model that could fit the data and allows for online
        bayesian CP detection. The *model_prior* is a numpy array of floats,
        summing to one such that the i-th entry corresponds to the prior belief
        that the i-th model occurs in a segment. Lastly, *cp_model* is an
        object of class CpModel that stores all properties about the CP prior,
        i.e. the probability of one occuring at every time point.
        """
        #DEBUG: initialize correctly
        self.gradient = 0.0

        """store the inputs into object"""
        self.data = data.reshape(T, S1*S2)
        self.model_universe = model_universe
        self.model_prior = model_prior
        self.cp_model = cp_model
        if isinstance(model_universe, list):
            self.Q = int(len(model_universe))
        else:
            self.Q = model_universe.shape[0]
        self.T, self.S1, self.S2 = T, S1, S2
        self.threshold = threshold
        self.store_rl, self.store_mrl = store_rl, store_mrl
        self.not_all_initialized = True
        #self.MAP_first_reached = True
        self.m_star_old = 0
        self.first_model_initialized = False
        self.trim_type = trim_type #other option: "threshold"
        self.notifications = notifications
        self.save_performance_indicators = save_performance_indicators
        self.training_period = training_period
        self.negative_log_likelihood = []
        self.negative_log_likelihood_fixed_pars = []
        self.MSE = []
        self.MAE = []

        """Take care of exogeneous variables"""
        #DEBUG: At this point, assumes that we have an obs. for each location
        #       at later stage, might want to group areas and have only one
        #       exo observation per group per variable
        if exo_data is not None:
            self.exo_data = exo_data.reshape(T, S1*S2, num_exo_vars)
        else:
             self.exo_data = exo_data
        self.num_exo_vars = num_exo_vars


        """create internal data structures for most recent computed objects"""
        self.evidence = -np.inf
        self.MAP = None
        self.y_pred_mean = np.zeros(shape=(self.S1, self.S2))
        self.y_pred_var  = np.zeros(shape = (self.S1*self.S2, self.S1*self.S2))


        """create internal data structures for all computed objects"""
        self.model_and_run_length_log_distr = (-np.inf *
                                            np.ones(shape = (self.Q, self.T+1)))
        self.storage_run_length_log_distr = [] #np.zeros(shape=(self.T+1, self.T+1))
        self.storage_model_and_run_length_log_distr = []
        self.storage_all_retained_run_lengths = []
        self.storage_mean = np.zeros(shape = (self.T, self.S1, self.S2))
        self.storage_var = np.zeros((self.T, self.S1*self.S2))
                           #np.zeros(shape = (self.T, self.S1*self.S2,
                           #                  self.S1*self.S2))
        self.storage_log_evidence = -np.inf * np.ones(shape = self.T)
        #DEBUG: I have changed code s.t. log_MAP_storage grows
        #self.log_MAP_storage = -np.inf * np.ones(self.T)
        #Note: has two entries at time t=1, for r = 0 and r>0
        self.log_MAP_storage = np.array([0,0])#np.array([0,0])
        self.CPs = [[]] * self.T
        self.MAP_segmentation = [np.array([[],[]])]
        #self.segment_log_densities = np.zeros(shape = (self.Q, self.T) )

        """store the smallest & largest lag length"""
        self.smallest_lag_length = 99999
        for model in self.model_universe:
            if model.has_lags:
                self.smallest_lag_length = min(self.smallest_lag_length,
                                               model.lag_length)
            else:
                self.smallest_lag_length = 0
        self.max_lag_length = 0
        for model in self.model_universe:
            if model.has_lags:
                self.max_lag_length = max(self.max_lag_length,
                                               model.lag_length)

        """STEP 3: If we are in a generalized Bayes setting, modify each model
        object accordingly"""

        """STEP 3.1: If we use DPD for parameter inference and we want to
        optimize the alpha paramter across all models, set the initial value
        of alpha_param using either detector-input or the first model's val"""
        self.alpha_param_learning = alpha_param_learning
        if ((alpha_param is not None) and alpha_param_learning == "together"):
            self.alpha_param = alpha_param
            self.gradient_alpha_param_count = 0
            self.gradient_alpha_param = 0
        elif alpha_param_learning == "together":
            self.alpha_param = self.model_universe[0].alpha_param
            self.gradient_alpha_param_count = 0
            self.gradient_alpha_param = 0
            print("WARNING! You are using DPD for parameter inference " +
                  "and want to optimize alpha_param across all models, " +
                  "but you have not specified an initial value in the " +
                  "detector object. The alpha_param of the first " +
                  "model in the model universe was chosen instead")

        """STEP 3.2: Set whether we do alpha_param learning inside models.
        Next, set alpha_param for all models that are DPD in case we do
        joint optimization across all models. """
        if ((alpha_param_learning == "individual") or
             alpha_param_learning == "together"):
            if alpha_param_learning == "individual":
                self.gradient_alpha_param_count = np.zeros(self.Q)
                self.gradient_alpha_param = np.zeros(self.Q)
            """Set the alpha_param learning attribute for DPD models"""
            for model in self.model_universe:
                if isinstance(model, BVARNIGDPD):
                    model.alpha_param_learning = True
                    """Set the initial alpha_param value. This ensures that
                    the optimization step makes sense."""
                    if alpha_param_learning == "together":
                        model.alpha_param = self.alpha_param

        self.alpha_param_opt_t = alpha_param_opt_t
        self.alpha_rld_opt_t = alpha_rld_opt_t
        self.alpha_opt_count = 0

        """STEP 3.3: Set all phantities relating to Run-length distribution
        robustification using DPD"""
        self.alpha_rld_learning = alpha_rld_learning
        self.generalized_bayes_rld = generalized_bayes_rld
        self.alpha_rld = alpha_rld
        self.alpha_list = []
        if generalized_bayes_rld == "power_divergence": #or generalized_bayes is True:
            for model in self.model_universe:
                model.generalized_bayes_rld = "power_divergence"
                model.alpha_rld = self.alpha_rld
                model.alpha_rld_learning = (
                        self.alpha_rld_learning)
            self.jlp_scale = None #initialize the scaling of log probs
            self.gradient_alpha_rld_count = 0
            self.gradient_alpha_rld = 0

        """STEP 3.4: Read in all the SGD-related functions generating stepsize,
        epsilon (for finite differences) and the loss function"""

        """STEP 3.4.1: Default choices for our functions"""
        self.C = 1.0
        if (loss_der_rld_learning is None):
            loss_der_rld_learning = Detector.bounded_absolute_loss_derivative
        elif (loss_der_rld_learning == "squared_loss" or
            loss_der_rld_learning == "squared_loss_derivative"):
            loss_der_rld_learning = Detector.squared_loss_derivative
        elif (loss_der_rld_learning == "absolute_loss" or
              loss_der_rld_learning == "absolute_loss_derivative"):
            loss_der_rld_learning = Detector.absolute_loss_derivative

        if loss_param_learning is None:
            loss_param_learning = Detector.bounded_absolute_loss
        elif loss_param_learning == "squared_loss":
            loss_param_learning = Detector.squared_loss
        elif loss_param_learning == "absolute_loss":
            loss_param_learning = Detector.absolute_loss

        if step_size_rld_learning is None:
            step_size_rld_learning = Detector.step_size_gen_rld
        if step_size_param_learning is None:
            step_size_param_learning = Detector.step_size_gen_rld
        if eps_param_learning is None:
            eps_param_learning = Detector.eps_gen

        """STEP 3.4.2: Set the Detector attribute functions """
        self.loss_der_rld_learning = loss_der_rld_learning
        self.loss_param_learning = loss_param_learning
        #DEBUG: Get some default functions here that make sense!
        self.step_size_rld_learning = step_size_rld_learning
        self.step_size_param_learning = step_size_param_learning
        if step_size_param_learning is None:
            self.step_size_param_learning = (
                    Detector.default_step_size_param_learning)
        self.eps_param_learning = eps_param_learning

        self.all_retained_run_lengths = np.array([], dtype=int)


    def reinstantiate(self, new_model_universe):
        """clone this detector """
        """STEP 1: Copy all contents of this detector"""
        data, model_universe = self.data, new_model_universe
        model_prior, cp_model = self.model_prior, self.cp_model
        S1, S2, T = self.S1, self.S2, self.T
        exo_data, num_exo_vars = self.exo_data, self.num_exo_vars
        threshold = self.threshold
        store_rl, store_mrl = self.store_rl, self.store_mrl
        trim_type, notifications = self.trim_type, self.notifications
        save_performance_indicators = self.save_performance_indicators
        training_period = self.training_period

        #DEBUG: Needs updating.
        """STEP 2: Create the new detector, and return it"""
        new_detector = Detector(data, model_universe, model_prior, cp_model,
                 S1, S2, T, exo_data, num_exo_vars, threshold,
                 store_rl, store_mrl, trim_type,
                 notifications,
                 save_performance_indicators,
                 training_period)
        return new_detector


    def run(self, start=None, stop=None):
        """Start running the Detector from *start* to *stop*, usually from
        the first to the last observation (i.e. default).
        """

        """set start and stop if not supplied"""
        if start is None:
            start = 1
        if stop is None:
            stop = self.T

        """run the detector"""
        time_start = time.clock()
        time2 = 0.0
        for t in range(start-1, stop-1):
            if t % self.notifications == 0 and time2 != 0.0:
                print("Processing observation #" + str(int(t)))
                print("Last iteration took " + str(time.clock() - time2) +
                      " seconds")
            time2 = time.clock()
            self.next_run(self.data[t,:], t+1)
        self.execution_time = time.clock() - time_start


    def next_run(self, y, t):
        """for a new observation *y* at time *t*, run the entire algorithm
        and compute all quantities at the ProbabilityModel object and the
        Detector object layer necessary

        NOTE: The data structures in the ProbabilityModel objects grow from
        size t-1 (or t, for the sufficient statistics) to size t (or t+1)
        during 'next_run'.        #DEBUG: To be added: check if prior_mean, prior_var, ... are S1xS2. If
        #       they are not, assume that they are provided in vector and
        #       full covariance matrix form & compress them into internal form
        """


        """STEP 1: If t==1, initialize *joint_probabilities* and the
        predictive distributions.
        If t>1, update the *joint_probabilities* of (y_{1:t}, r_t =r|q_t =q)
        for all q in the model universe as well as the predictive
        distributions associated with each model.
        Note: Calls functions 'update_joint_log_probabilities' and
              'update_predictive_distributions'. The latter function is on
              ProbabilityModel level, and calls 'evaluate_predictive_log_distr'
              from the Model level. If we are interested in retrieving the
              negative log likelihood, the
              'one_step_ahead_predictive_log_probs' are also retrieved there."""

        """If we want the fixed-parameter NLL, we need to compute and store it
        before we update the joint log prob"""
        #if(self.save_performance_indicators and t>self.training_period):
            #self.compute_negative_log_likelihood_fixed_pars(y,t) #these need to
        #INSERT: alpha_param update here!
        """For all parameter-DPD models in our model universe, first update
        their value of alpha_param"""
        if self.max_lag_length + 3 < t and self.alpha_param_opt_t < t:
            #only do if MAP has changed last iteration!
            #DEBUG: Version of Code used for paper results!
            if t > 3:
                if self.CPs[t-2][-1][0] != self.CPs[t-3][-1][0]:
                    self.alpha_opt_count = self.alpha_opt_count  + 1
                    self.update_alpha_param(y,self.alpha_opt_count, True)
                else:
                    #I.e., do not update if the CPs have not changed.
                    self.update_alpha_param(y,self.alpha_opt_count, False)
#            if t > 3:
#                #if self.CPs[t-2][-1][0] != self.CPs[t-3][-1][0]:
#                self.alpha_opt_count = self.alpha_opt_count  + 1
#                self.update_alpha_param(y,self.alpha_opt_count, True)
                #else:
                    #I.e., do not update if the CPs have not changed.
                #    self.update_alpha_param(y,self.alpha_opt_count, False)

        # Jay: Something is not working here
        self.update_all_joint_log_probabilities(y, t)

        """STEP 1+: If we want to save the negative log likelihood, do it here
        because the run-length and model distro is not yet updated, but we
        already have all the evaluated log probs y_t|y_{1:t-1},r_t-1,m_t-1"""
        if(self.save_performance_indicators and t>self.training_period):
            #DEBUG: Implement function
            self.save_negative_log_likelihood(t)
            #self.save_negative_log_likelihood_fixed_pars(t)
            self.save_MSE(y,t)

        """STEP 2: Collect the model-specific evidences and update the overall
        evidence by summing them up. 'update_evidence' is a simple wrapper
        function for convenience"""
        #Jay: Everything is the same here
        self.update_log_evidence()

        """STEP 3: Trim the run-length distributions in each model. Next,
        update the distributions (q_t=q, r_t=r|y_{1:t}) for each
        run-length r and each model in the model universe q, store the
        result in *self.model_and_run_length_distr*"""
        #Jay: This isn't even used in the OLD one, LOL
        self.trim_run_length_log_distributions(t)
        #self.update_model_and_run_length_log_distribution(t)

        #Jay: This works during initialization only?
        self.update_run_length_log_distribution(t) #this is stored on detector level
                                                   #and computed from the model objects
        #DEBUG: This was meant to provide numerical stability, but doesn't work
#        if (not self.not_all_initialized and
#            self.generalized_bayes_rld == "power_divergence"):
#            self.rescale_DPD_run_length_log_distribution(t) #avoid numerical issues

        """STEP 5: Using the results from STEP 3, obtain a prediction for the
        next spatial lattice slice, which you preferably should either store,
        output, or write to some location"""
        # Jay: How the OLD one looks like
        # self.prediction_y(y, t)
        # self.storage(t) # Jay: This is exactly the same

        # Jay: original
        if not self.not_all_initialized:
            self.prediction_y(y, t)
            self.storage(t)

        """STEP 8: Update alpha if you do power-divergence based inference.
        Only start doing this once all models have been initialized"""
        if ((not self.not_all_initialized) and
            self.generalized_bayes_rld == "power_divergence" and
            self.alpha_rld_learning and
            self.alpha_rld_opt_t < t):
            #only do this step if MAP CP has changed!
            if t >= 3:
                if self.CPs[t-2][-1][0] != self.CPs[t-3][-1][0]:
                    #alpha opt count already updated in param update
                    #self.alpha_opt_count = self.alpha_opt_count  + 1
                    self.update_alpha_rld(y,self.alpha_opt_count,True)
                else:
                    self.update_alpha_rld(y,self.alpha_opt_count,False)

        """STEP 4: Check if all models have been initialized. This only needs
        to be checked for BVAR models. If they have all been initialized last
        round (i.e., self.not_all_initialized = False), then there is never any
        need to check in any of the succeeding rounds again"""
        if self.not_all_initialized:
            count_total, count_init = 0,0
            for model in self.model_universe:
                if model.has_lags:
                    if (t - model.lag_length >=1):
                        count_total += 1
                        count_init  += 1
                    else:
                        count_total +=1
            if count_total > count_init:
                self.not_all_initialized = True
            else:
                self.not_all_initialized = False



        """STEP 6: Using the results from STEP 3, obtain a MAP for the
        most likely segmentation & models per segment using the algorithm of
        Fearnhead & Liu (2007)"""
        #if not self.not_all_initialized:
        self.MAP_estimate(t)

        #NEEDS FURTHER INVESTIGATION
        """STEP 7: For each model in the model universe, update the priors to
        be the posterior expectation/variance"""

        # Jay: lol this doesn't even run because auto_prior_update is set False
        if not self.not_all_initialized:
            self.update_priors(t)

    def rescale_DPD_run_length_log_distribution(self, t):
        """STEP 1: Compute the mean of all joint log probs (note that since
        they are computed with the DPD, they will be positive!)"""
        if self.jlp_scale is None:
            log_scale_new = (np.max([model.joint_log_probabilities
                        for model in self.model_universe]) - 1)
#            log_scale_new = max(1.0, scipy.misc.logsumexp([
#                        model.joint_log_probabilities
#                        for model in self.model_universe]) - 1)
            rescaler_for_old_obs = None
        else:
            log_scale_old = self.jlp_scale
#            max_ = (np.max([model.joint_log_probabilities
#                        for model in self.model_universe]) - 1)
            min_ = (np.min([model.joint_log_probabilities
                        for model in self.model_universe]) - 1)
#            mean_ = ((1.0/(self.Q + len(self.all_retained_run_lengths))) *
#                     scipy.misc.logsumexp([model.joint_log_probabilities
#                        for model in self.model_universe]))
#            log_scale_new = scipy.misc.logsumexp(
#                    [model.joint_log_probabilities
#                        for model in self.model_universe])
            log_scale_new = min_ #0.5 * max(max_-min_, 2) #min(log_scale_old,
#                    (np.max([model.joint_log_probabilities
#                        for model in self.model_universe]) - 1))
#            log_scale_new = max(1.0, scipy.misc.logsumexp([model.joint_log_probabilities
#                        for model in self.model_universe]))
            rescaler_for_old_obs =  log_scale_old - log_scale_new

#         -1 = nothing will be negative

 #        only applied to most recent obs = np.log(scale_old/scale_new)
        self.jlp_scale = log_scale_new
#        log_scale_new = scipy.misc.logsumexp([model.joint_log_probabilities
#                        for model in self.model_universe])

#        """STEP 2: Use this mean to rescale all of them"""
        #for model in self.model_universe:
            #probabilitiy level call
            #model.rescale_DPD_run_length_distribution(log_scale_new,
            #                                          None, #rescaler_for_old_obs, #rescaler_for_old_obs, #rescaler_for_old_obs,
            #                                          t)


    def update_alpha_param(self, y, t, update=True):
        """Use the posterior expectation for alpha + eps and alpha - eps to
        approximate the gradient for Loss(PredError(alpha)) w.r.t. alpha.
        If alpha_param_learning = individual, then you optimize each DPD model
        independently. If learning = together, optimize together"""

        """STEP 1: If we learn jointly over all DPD models, get their model
        posterior probabilities s.t. we can get
         E[y_t|y_1:t-1, alpha_t-1 + eps] = sum(
                 E[y_t|y_1:t-1, m_t-1, alpha_t-1 + eps]) *
                 P(m_t-1|y_1:t-1,alpha_t-1 + eps)
            )
        where we have stored P(m_t-1|y_1:t-1,alpha_t-1 + eps) from before"""

        if self.alpha_param_learning == "together":
            """STEP 1A: If we optimize alpha_param over all models"""
            #number_DPD_models = 0
            eps = self.eps_gen(t-1) #as the eps is from the previous iteration
            DPD_model_indices = []
            list_model_log_evidences_p_eps = []
            list_model_log_evidences_m_eps = []
            list_post_mean_p_eps = []
            list_post_mean_m_eps = []

            """STEP 1A.1: count number of DPD models and retrieve both their
            model evidences and posterior expectations"""
            for (m, model) in zip(range(0, self.Q), self.model_universe):
                if isinstance(model, BVARNIGDPD):
                    """retrieve P(m_t-1, y_1:t-1|alpha_t-1 +/- eps)"""
                    list_model_log_evidences_p_eps.append(
                            model.model_log_evidence_p_eps)
                    list_model_log_evidences_m_eps.append(
                            model.model_log_evidence_m_eps)
                    """retrieve E[y_t|m_t-1, y_1:t-1, alpha_t-1 +/- eps]"""
                    list_post_mean_p_eps.append(model.
                            post_mean_p_eps)
                    list_post_mean_m_eps.append(model.
                            post_mean_m_eps)
                    """get index of this DPD model"""
                    DPD_model_indices.append(m)


            """STEP 1A.2: compute the posterior means for alpha +/- eps"""

            """STEP 1A.2.1: Get the model posteriors for alpha +/- eps"""
            total_evidence_p_eps = scipy.misc.logsumexp(
                    list_model_log_evidences_p_eps)
            total_evidence_m_eps = scipy.misc.logsumexp(
                    list_model_log_evidences_m_eps)
            model_posteriors_p_eps = np.exp(
                np.array(list_model_log_evidences_p_eps)
                - total_evidence_p_eps)
            model_posteriors_m_eps = np.exp(
                np.array(list_model_log_evidences_m_eps)
                - total_evidence_m_eps)
            """STEP 1A.2.2: Get the posterior mean"""
            post_mean_p_eps = (np.array(list_post_mean_p_eps)
                               * model_posteriors_p_eps[:,np.newaxis])
            post_mean_m_eps = (np.array(list_post_mean_m_eps)
                               * model_posteriors_m_eps[:,np.newaxis])

            """STEP 1A.3: Compute predictive loss and take gradient step"""
            #DEBUG: use more general loss functions
            loss_p_eps = self.loss_param_learning(post_mean_p_eps -
                                                  y.flatten(), self.C)
            loss_m_eps = self.loss_param_learning(post_mean_m_eps -
                                                  y.flatten(), self.C)
            self.gradient_alpha_param = (self.gradient_alpha_param +
                                         (loss_p_eps - loss_m_eps)/(2*eps))
            #DEBUG: Use more general step sizes
            self.gradient_alpha_param_count = (self.gradient_alpha_param_count
                                               + 1)

            #bound alpha_param between pow(10,-10) and 10
            if update:
                step_size = self.step_size_param_learning(t)
                #abs_increment = min(abs(step_size * self.gradient_alpha_param),
                #                    0.05)
                abs_increment = min(0.1,
                        step_size *
                        (1.0/self.gradient_alpha_param_count)*
                        self.gradient_alpha_param)
                self.alpha_param = min(
                        max(
                            pow(10,-10),
                            self.alpha_param -
                            abs_increment *
                            np.sign(self.gradient_alpha_param) #step_size * gradient_alpha_param
                        ),
                        10.0
                    )
                self.gradient_alpha_param = 0
                self.gradient_alpha_param_count = 0
            #print("detector alpha param", self.alpha_param)

                """STEP 1A.4: For each DPD model, update alpha_param"""
                for m in DPD_model_indices:
                    self.model_universe[m].alpha_param = self.alpha_param
                    self.model_universe[m].alpha_param_list.append(
                            self.alpha_param)


        elif self.alpha_param_learning == "individual":
            """STEP 1B: If we optimize alpha_param individually"""
            for (m, model) in zip(range(0, self.Q), self.model_universe):
                if isinstance(model, BVARNIGDPD):
                    #eps = self.eps_gen(t-1)
                    #DEBUG: use more general loss functions
                    loss_p_eps = np.sum(np.abs(model.post_mean_p_eps -
                                               y.flatten()))
                    loss_m_eps = np.sum(np.abs(model.post_mean_m_eps -
                                               y.flatten()))
                    self.gradient_alpha_param[m] = (self.gradient_alpha_param[m] +
                                                 (loss_p_eps - loss_m_eps)/
                                            (2 * model.eps))
                    self.gradient_alpha_param_count[m] = (
                            self.gradient_alpha_param_count[m]+ 1)

                    #bound alpha_param between pow(10,-10) and 10
                    #scale the step size by model complexity to counteract
                    #higher variance for more complex models
                    if update:
                        #print("PARAM gradient size:", self.gradient_alpha_param[m]/self.gradient_alpha_param_count[m])
                        step_size = self.step_size_param_learning(t)
                        abs_increment = min(0.1,
                                    step_size *
                                    (1.0/self.gradient_alpha_param_count[m])*
                                    self.gradient_alpha_param[m])
                        model.alpha_param = min(
                                    max(
                                        pow(10,-10),
                                        model.alpha_param -
                                        abs_increment *
                                        np.sign(self.gradient_alpha_param[m]) #step_size * gradient_alpha_param
                                    ),
                                    10.0
                                )
#                        model.alpha_param = min(
#                                max(
#                                    pow(10,-10),
#                                    model.alpha_param
#                                    + step_size * (1.0/(model.num_regressors+1)) *
#                                      self.gradient_alpha_param[m]
#                                ),
#                                10.0
#                            )
                        #print("model alpha param", model.alpha_param)
                        model.alpha_param_list.append(model.alpha_param)
                        self.gradient_alpha_param[m] = 0
                        self.gradient_alpha_param_count[m] = 0

    def update_run_length_log_distribution(self, t):
        """This function aggregates the models' individual model and run-length
        log probability appropriately s.t. we end up with the proper run-length
        log distribution for which the recursions are shown in Fearnhead & Liu"""

        """STEP 1: Get all growth probabilities & the CP probability"""

        """STEP 1.1: Get all models that have been initialized"""
        indices_initialized_models = []
        for (m, model) in zip(range(0, self.Q), self.model_universe):
            """if the model has no lags, give it 'lag 1'"""
            if model.has_lags:
                model_lag = model.lag_length
            else:
                model_lag = 0
            """if the model already contains joint_log_probabilities, use it"""
            if model_lag < t: #debug: model_lag =< t?
                indices_initialized_models.append(m)
        num_initialized = int(len(indices_initialized_models))

        """Check if this is the first time we have an initialized model. If so,
        we need to initialize the quantity self.run_length_log_distr. If we
        have no initialized model yet, leave the function immediately"""
        if (not self.first_model_initialized) and num_initialized > 0:
            """initialize run-length distro s.t. we can work with it"""
            self.first_model_initialized = True
            self.run_length_log_distr = 0 # log(1) = 0
        elif (not self.first_model_initialized):
            """skip the rest of the function"""
            return None


        """STEP 1.3: For all initialized models, retrieve the run lengths and
        aggregate them all into *all_run_lengths*"""
        has_both = False #boolean. If we have run lengths for t-1 and >t-1, flag
        all_run_lengths = np.array([])
        for model in self.model_universe[indices_initialized_models]:
            """Get all run lengths retained across models"""
            all_run_lengths = np.union1d(all_run_lengths,
                                         model.retained_run_lengths)
            """if we have the run-lengths for t-1 and >t-1, flag up and later
            place one index twice into the all_run_lengths"""
            if model.retained_run_lengths[-1] == model.retained_run_lengths[-2]:
                has_both = True

        """Add run length for >t-1 if necessary"""
        if has_both:
            all_run_lengths = np.append(all_run_lengths, t)#all_run_lengths[-1])
        """make sure that the run-lengths we retain can be used to access
        the array elements that we want."""
        all_run_lengths = all_run_lengths.astype(int)

        """STEP 1.4: Create the model and run-length distr s.t. we have no
        zero mass entries"""
        length_rls = int(len(all_run_lengths))
        model_rl_log_distributions = (-np.inf) * np.ones((
                self.Q, length_rls))

        """STEP 1.5: loop over the initialized models, and fill in the model
        and run-length distro"""
        for index in indices_initialized_models:
            model = self.model_universe[index]
            """get indices relative to all run lengths"""
            model_indices_indicators_relative_to_all_run_lengths = np.in1d(
                    all_run_lengths, model.retained_run_lengths)
            """If the current model has retained run length r>t-1, set it
            to t."""
            if (model.retained_run_lengths[-1] == model.retained_run_lengths[-2]):
                model_indices_indicators_relative_to_all_run_lengths[
                            -1] = True

            """get P(r_t,m_t|y_1:t) = P(r_t,m_t,y_1:t)/P(y_1:t)"""
            model_rl_log_distributions[index,
                    model_indices_indicators_relative_to_all_run_lengths] = (model.joint_log_probabilities - self.log_evidence)
                    #+ np.log(priors_initialized_models[m]))

        """STEP 1.4: we need to sum over the columns (i.e. the models).
        One needs log sum exp for this, since all is stored in log form.
        P(r_t|y_1:t) = \sum_m P(r_t,m_t|y_1:t)"""
        run_length_log_distr = misc.logsumexp(
                model_rl_log_distributions, axis=0)

        """STEP 2: Store to object, and if all run lengths are to be stored,
        also store it to another object"""
        self.model_and_run_length_log_distr = model_rl_log_distributions
        self.run_length_log_distr = run_length_log_distr
        self.all_retained_run_lengths = all_run_lengths

        """STEP 3: Store it if needed"""
        if self.store_mrl or self.store_rl:
            self.storage_all_retained_run_lengths.append(
                    self.all_retained_run_lengths)
        if self.store_rl:
            # Jay: this used to be storage_run_length_distr without the log
            self.storage_run_length_log_distr.append(self.run_length_log_distr)
        if self.store_mrl:
            self.storage_model_and_run_length_log_distr.append(
                    self.model_and_run_length_log_distr)


    #IMPLEMENTED FOR ALL SUBCLASSES IF predictive_probabilities WORK IN SUBLCASS
    #                                   update_joint_probabilities WORK
    def update_all_joint_log_probabilities(self, y, t):
        """Let the individual objects in *model_universe* compute their growth
        and CP probabilities, via 'update_joint_probabilities' in each.
        The strucutre is: (1) update the joint log probs (or initialize the
        model if first observation), which is done by calling probability model
        functions, and (2) update the predictive probabilities using the
        new observation y, which is done differently in each model object.
        """

        """STEP 0: Get which of the models are or will be
        initialized at t for redistributing the model prior later. """
        index_initialized = []
        for (m,model) in zip(range(0,self.Q),self.model_universe):
            if model.has_lags and ((t+1) - model.lag_length >= 1):
                index_initialized.append(m)
            elif  not model.has_lags:
                index_initialized.append(m)
        prior_rescaling_factor = np.sum(self.model_prior[index_initialized])
        #print()

        #if not self.first_model_initialized:
            #we need the initial run-length distro


        """If we need to do hyperparameter learning for the generalized
        Bayesian case (e.g., learning alpha for Power divergence)"""

        """Initialize these quantities. If we enter the next if-statement, they
        might be changed"""
        log_model_posteriors_der_m = None
        log_model_posteriors_der_sign_m = None
        log_CP_evidence_der = None
        log_CP_evidence_der_sign = None

        if (self.generalized_bayes_rld == "power_divergence"
            and self.alpha_rld_learning
            and t>1):
            #DEBUG: Need to be computed from all models
            """Compute the model posterior derivative w.r.t. alpha
            (in log form)"""


            """Check if at least one model is already initialized"""
            at_least_one_model_initialized = np.any([(model.has_lags and
                ((t) - model.lag_length)>1) or ( not model.has_lags )
                for model in self.model_universe])

            if at_least_one_model_initialized:
                """collect all derivatives of all models as well as the
                joint probabilities of all models. Sum over models per run-
                length afterwards"""
                all_log_probs = -np.inf*np.ones(
                        (self.Q, np.size(self.all_retained_run_lengths)))
                all_log_alpha_derivatives = -np.inf*np.ones(
                        (self.Q, np.size(self.all_retained_run_lengths)))
                all_log_alpha_derivatives_sign = np.zeros(
                        (self.Q, np.size(self.all_retained_run_lengths)))
                for m, model in zip(range(0,self.Q), self.model_universe):
                    #DEBUG: Unclear if the log derivatives joint log probs
                    #       are going to be one entry too many (for r=0)
                    #DEBUG: Indexing!

                    """Only retrieve quantities of models that have seen data
                    already"""
                    warmed_up = ((model.has_lags and ((t) - model.lag_length)>1) or
                        ( not model.has_lags ))
                    if warmed_up:

                        """get indices relative to all run lengths"""
                        model_indices_indicators_relative_to_all_run_lengths=(
                            np.in1d(self.all_retained_run_lengths,
                                    model.retained_run_lengths))
                        if (model.retained_run_lengths[-1] ==
                            model.retained_run_lengths[-2]):
                            model_indices_indicators_relative_to_all_run_lengths[-1] = True

        #                        print('rl', np.size(model_indices_indicators_relative_to_all_run_lengths))
        #                        print(model_indices_indicators_relative_to_all_run_lengths)
        #                        print(self.all_retained_run_lengths)
        #                        print('jlp', np.size(model.joint_log_probabilities))
        #                        print('alp', np.size(all_log_probs[m,:]))
                        #DEBUG: The initial log_alpha_derivative_joint_probs
                        #       has to be in the right size relative to lag l.
                        #DEBUG: Initialized to None, so initialize it when you
                        #       arrive here the first time by checking if = None
                        """Check if they have been initialized already. If not,
                        we need to do that now"""
                        if (model.log_alpha_derivatives_joint_probabilities
                            is None):
                            num_needed = np.sum(
                             model_indices_indicators_relative_to_all_run_lengths)
                            model.log_alpha_derivatives_joint_probabilities = (
                                    -np.inf * np.ones(num_needed))
                            model.log_alpha_derivatives_joint_probabilities_sign = (
                                    np.ones(num_needed))

                        """fill retrieved values into relevant position"""
                        all_log_alpha_derivatives[m,
                         model_indices_indicators_relative_to_all_run_lengths]=(
                            model.log_alpha_derivatives_joint_probabilities )
                        all_log_alpha_derivatives_sign[m,
                         model_indices_indicators_relative_to_all_run_lengths]=(
                         model.log_alpha_derivatives_joint_probabilities_sign)

                        #DEBUG: it seems that we don't have enough retained
                        #       indices in the relative ones
                        all_log_probs[m,
                         model_indices_indicators_relative_to_all_run_lengths]=(
                         model.joint_log_probabilities)

                """sum over the models for each run-length, needed for the
                derivative of P(m_t|m_t-1, r_t-1, y_1:t-1), see (4) in
                handwritten notes. Dimension is Rx1"""
                model_sums_derivatives, model_sums_derivatives_sign = (
                    scipy.misc.logsumexp(
                        a = all_log_alpha_derivatives,
                        b = all_log_alpha_derivatives_sign,
                        return_sign = True,
                        axis=0
                    ))
                model_sums = scipy.misc.logsumexp(
                        a = all_log_probs,
                        axis = 0
                    )

                """Get the two expressions that added together give you the
                derivative of the joint probabilities in log-form"""
                #Debug: In expr_1, we have MxR - M, make sure dimensions
                #       match. expr_2 has dimension R
                #DEBUG: check expr_2 and fix the dimension
                expr_1 = all_log_alpha_derivatives - model_sums
                sign_1 = all_log_alpha_derivatives_sign
                expr_2 = -2* model_sums + model_sums_derivatives + all_log_probs
                sign_2 = (-1) * model_sums_derivatives_sign

                #print("expr_1", expr_1)
                #print("expr_2", expr_2)
#                print("sign_1", sign_1.shape)
#                print("sign_2", sign_2.shape)

                expr, sign = scipy.misc.logsumexp(
                        a = np.array([expr_1, expr_2]),
                        b = np.array([sign_1, sign_2 *
                                      np.ones(self.Q)[:,np.newaxis]]),
                        return_sign = True,
                        axis=0
                    )

                #Note: We only have the growth-probabilities in here!
                #      the CP probability derivatives are computed below!
                log_model_posteriors_der = expr
                log_model_posteriors_der_sign = sign

                """Compute the CP evidence derivative w.r.t. alpha
                (in log form). It turns out that the computation decomposes
                s.t. we have
                    derivative(one-step-pred) * CP_evidence * q(m_t) +
                    one-step-pred * derivative(CP_evidence) * q(m_t).
                Since we don't  want to deal with the one-step-ahead
                predictives here, all we need to do is compute
                derivative(CP_evidence) and pass it on. The full
                computation is then done when we update the probs."""
                #DEBUG: Where does the evidence come from?
                #DEBUG: Unclear what we compute here and inside probability_model!
                _1, _2 = misc.logsumexp(
                        a = np.log(self.cp_model.hazard_vector(1, t)) +
                            all_log_alpha_derivatives, # +
                            #self.log_evidence,
                        b = all_log_alpha_derivatives_sign,
                        return_sign = True)
                log_CP_evidence_der = _1
                log_CP_evidence_der_sign = _2

#            else:
#                log_model_posteriors_der_m = None
#                log_model_posteriors_der_sign_m = None
#                log_CP_evidence_der = None
#                log_CP_evidence_der_sign = None

        #q = 0
        for (m,model) in zip(range(0,self.Q),self.model_universe):

            """STEP 1: Check if it is the first observation, and initialize the
            joint distributions in each model object if so"""
            #DEBUG: t is t-1!
            initialization_required = False
            if model.has_lags and ((t) - model.lag_length == 1):
                initialization_required = True
            elif  (not model.has_lags) and t == 1:
                initialization_required = True


            if initialization_required:
                """This command gives an initialization of (i) the
                *joint_probabilities*, (ii) the predictive distributions (i.e.,
                the sufficient statistics), and (iii) the *model_evidence*
                of each model."""

                """NOTE: We need to initialize BVAR at time t-lag_length!"""
                if model.has_lags or isinstance(model, BVARNIG):
                     """If we have a BVAR model, we need to pass data of
                     sufficient lag length to the initialization."""
                     #DEBUG: exo_selection in correct dimension? What is dim of
                     #       exo_data?
                     #DEBUG: exo data assumed to be contemporaneous!
                     """STEP I: Get endogeneous data"""
                     X_endo = self.data[:model.lag_length+1,:]
                     Y_2 = self.data[model.lag_length+1,:]
                     """STEP II: Get exogeneous data (if needed)"""
                     if model.exo_bool:
                         X_exo = self.exo_data[model.lag_length,
                                               model.exo_selection,:]
                         X_exo_2 = self.exo_data[model.lag_length+1,
                                                 model.exo_selection,:]
                     else:
                         X_exo = X_exo_2 = None
                     """STEP III: Use exo and endo data to initialize model"""
                     #new_prior = self.model_prior[m]/np.sum(self.)

                     model.initialization(X_endo, X_exo, Y_2, X_exo_2,
                                    self.cp_model,
                                    self.model_prior[m]/prior_rescaling_factor)
                else:
                     """Otherwise, just pass the first observation"""
                     #DEBUG: CONSTANT FITTING ADAPTION
#                     if isinstance(model, BVARNIG):
#                         """STEP I: Get endogeneous data"""
#                         X_endo = self.data[:model.lag_length+1,:]
#                         Y_2 = self.data[model.lag_length+1,:]
#                         """STEP II: Get exogeneous data (if needed)"""
#                         if model.exo_bool:
#                             X_exo = self.exo_data[model.lag_length,
#                                                   model.exo_selection,:]
#                             X_exo_2 = self.exo_data[model.lag_length+1,
#                                                     model.exo_selection,:]
#                         else:
#                             X_exo = X_exo_2 = None
#                         model.initialization(None, X_exo, Y_2, X_exo_2,
#                                    self.cp_model,
#                                    self.model_prior[m]/prior_rescaling_factor)
                     #else:

                     # Jay: This should run
                     model.initialization(y, self.cp_model,
                                              self.model_prior[m])
            else:
                """Make sure that we only modify joint_log_probs & predictive
                distributions for BVAR models for which t is large enough to
                cover the lag length."""

                #DEBUG: Use x_exo if BVAR

                """STEP 1: Within each ProbabilityModel object, compute its
                joint probabilities, i.e. the growth- & CP-probabilities
                associated with the model"""

                #"""get P(r_t,m_t|y_1:t) = P(r_t,m_t,y_1:t)/P(y_1:t)"""
                #model_rl_log_distributions[index,
                #    model_indices_indicators_relative_to_all_run_lengths] = (
                #    model.joint_log_probabilities -self.log_evidence )
                #    #+ np.log(priors_initialized_models[m]))

                """model_and_run_length_log_distr stores run-lengths for
                ALL models, so it will have -np.inf entries at
                run-lengths retained for other models. That's why we need
                to access its retained run lengths"""
                warmed_up = ((model.has_lags and ((t) - model.lag_length)>1) or
                    ( not model.has_lags ))
                #warmed_up_plus_one = (
                #    (model.has_lags and ((t) - model.lag_length)>2) or
                #    ( not model.has_lags  and t > 1))
                if warmed_up:
                    #print("self.model_and_run_length_log_distr[m,:] ",
                    #      self.model_and_run_length_log_distr)
                    #print("self.run_length_log_distr ",
                    #      self.run_length_log_distr)
                    #print("t ", t)
                    #print("lag length ", model.lag_length)
                    #print("joint log probs ", model.joint_log_probabilities)

                    #DEBUG: Incorrect! Alignment of run-length and mrl distro
                    #       does not hold!
                    log_model_posteriors = (
                            self.model_and_run_length_log_distr[m,:] -
                            self.run_length_log_distr)
                    log_model_posteriors = log_model_posteriors[np.where(
                            log_model_posteriors> -np.inf)]
                    #print("log model posteriors: ", log_model_posteriors)
                    """the log CP evidence is computed using the joint probs
                    as well as the hazard function/cp model"""
                    #DEBUG: needs to take into account vector-form of hazard
                    #       vector for more complicated hazard functions
                    #print("log evidence", self.log_evidence)
                    #print("model log evidences:", [m.model_log_evidence for m in self.model_universe])
                    #print("joint log probs", model.joint_log_probabilities)
                    #print("hazard", np.log(self.cp_model.hazard_vector(1, t)))
                    log_CP_evidence = misc.logsumexp(
                            np.log(self.cp_model.hazard_vector(1, t)) +
                            self.model_and_run_length_log_distr +
                            self.log_evidence)
                            #model.joint_log_probabilities)
                    #print("log CP evidence:", log_CP_evidence)
                    if (self.generalized_bayes_rld == "power_divergence"
                        and self.alpha_rld_learning):

                        """get indices relative to all run lengths"""
                        model_indices_indicators_relative_to_all_run_lengths=(
                            np.in1d(self.all_retained_run_lengths,
                                    model.retained_run_lengths))
                        if (model.retained_run_lengths[-1] ==
                            model.retained_run_lengths[-2]):
                            model_indices_indicators_relative_to_all_run_lengths[-1] = True

                        """Retrieve m-th entry of the calculated quantities"""
                        log_model_posteriors_der_m = (
                            log_model_posteriors_der[m,
                            model_indices_indicators_relative_to_all_run_lengths])
                        log_model_posteriors_der_sign_m = (
                            log_model_posteriors_der_sign[m,
                            model_indices_indicators_relative_to_all_run_lengths])
                        #DEBUG: needs to be one-dimensional!
                        #log_CP_evidence_der_m = (log_CP_evidence_der[m])
                        #log_CP_evidence_der_sign_m = (
                        #       log_CP_evidence_der_sign[m])



                """NOTE: If it is a BVAR model, we need the exogeneous vars,
                         and we also need to make sure that lag < t"""
                if(model.has_lags and ((t) - model.lag_length)>1):
                    """NOTE: This needs to be adjusted once we allow for
                    exogeneous variables"""


                    #We want the model posteriors to be passed in only from
                    #this model!

                    #DEBUG: Add a function for alpha_param learning step here.
                    #       Needs to happen before the joint log probs are
                    #       updated (as then, we can already use the new alpha
                    #       for the new joint log probs). Make it a probability
                    #       model level function that does nothing by default,
                    #       unless you extend it in your subclass.
                    if (t - model.lag_length)>2 and self.alpha_param_opt_t <= t:
                        #DEBUG: This calls aDPD_joint_log_prob_updater, which
                        #       in turns calls the integral computation.
                        model.alpha_param_gradient_computation(y=y, t=t,
                             cp_model = self.cp_model,
                             model_prior =
                                 self.model_prior[m]/prior_rescaling_factor,
                             log_model_posteriors = log_model_posteriors,
                             log_CP_evidence = log_CP_evidence,
                             eps = self.eps_param_learning(t))

                    model.update_joint_log_probabilities(
                        y=y,t=t, cp_model = self.cp_model,
                        model_prior =
                            self.model_prior[m]/prior_rescaling_factor,
                        log_model_posteriors = log_model_posteriors,
                        log_CP_evidence = log_CP_evidence,
                        log_model_posteriors_der = log_model_posteriors_der_m,
                        log_model_posteriors_der_sign =
                            log_model_posteriors_der_sign_m,
                        log_CP_evidence_der = log_CP_evidence_der,
                        log_CP_evidence_der_sign = log_CP_evidence_der_sign,
                        do_general_bayesian_hyperparameter_optimization = (
                                warmed_up))

                #DEBUG: CONSTANT FITTING ADAPTION
                elif( not model.has_lags ):
                    """NOTE: This does not allow for exogeneous variables"""
                    #impose condition s.t. BVAR is not called
                    if t > 2 and self.alpha_param_opt_t <= t:
                        model.alpha_param_gradient_computation(y=y, t=t,
                             cp_model = self.cp_model,
                             model_prior =
                                 self.model_prior[m]/prior_rescaling_factor,
                             log_model_posteriors = log_model_posteriors,
                             log_CP_evidence = log_CP_evidence,
                             eps = self.eps_param_learning(t))

                    # Jay: This should run
                    model.update_joint_log_probabilities(
                        y=y,t=t, cp_model = self.cp_model,
                        model_prior =
                            self.model_prior[m]/prior_rescaling_factor,
                        log_model_posteriors = log_model_posteriors,
                        log_CP_evidence = log_CP_evidence,
                        log_model_posteriors_der = log_model_posteriors_der_m,
                        log_model_posteriors_der_sign =
                            log_model_posteriors_der_sign_m,
                        log_CP_evidence_der = log_CP_evidence_der,
                        log_CP_evidence_der_sign = log_CP_evidence_der_sign,
                        do_general_bayesian_hyperparameter_optimization = (
                                warmed_up))



                """STEP 2: Update the predictive probabilities for each model
                inside the associated ProbabilityModel object. These are
                used in the 'update_joint_probabilities' call for the next
                observation. In the case of NaiveModel and other conjugate
                models, this step amounts to updating the sufficient
                statistics associated with the model"""

                """NOTE: If it is a BVAR model, we need the exogeneous vars,
                         and we also need to make sure that lag < t"""
                #DEBUG: CONSTANT FITTING ADAPTION
                if(model.has_lags and ((t) - model.lag_length)>1):

                    y_tm1 = self.data[t-2,:]
                    if model.exo_bool:
                        x_exo_t = self.exo_data[t,model.exo_selection,:]
                        x_exo_tp1 = self.exo_data[t+1,model.exo_selection,:]
                    else:
                        x_exo_t = x_exo_tp1 = None
                    if isinstance(model, BVARNIGDPD):
                        """If BVARNIGDPD model, we need the CP probability to
                        obtain the prior weights in the SGD step"""
                        model.update_predictive_distributions(y, y_tm1,
                                x_exo_t, x_exo_tp1, t, self.cp_model.hazard(0))
                    else:
                        """If not DPD model, we don't need CP probability"""
                        model.update_predictive_distributions(y, y_tm1,
                                 x_exo_t, x_exo_tp1, t)
                elif(not model.has_lags):
                    if isinstance(model, BVARNIG):
                        """If it is a BVARNIG model, call the more complicated
                        functions"""
                        y_tm1 = self.data[t-2,:]
                        if model.exo_bool:
                            x_exo_t = self.exo_data[t,model.exo_selection,:]
                            x_exo_tp1 = self.exo_data[t+1,
                                                      model.exo_selection,:]
                        else:
                            x_exo_t = x_exo_tp1 = None
                        if isinstance(model, BVARNIGDPD):
                            """If BVARNIGDPD model, we need the CP probability
                            to obtain the prior weights in the SGD step"""
                            model.update_predictive_distributions(y, y_tm1,
                                    x_exo_t, x_exo_tp1, t, \
                                    self.cp_model.hazard(0))
                        else:
                            """If not DPD model, we don't need CP probability"""
                            model.update_predictive_distributions(y, y_tm1,
                                     x_exo_t, x_exo_tp1, t)
                    else:
                        # Jay: This should run
                        model.update_predictive_distributions(y, t)


    def update_alpha_rld(self,y,t, update=True):
        """Using a loss function that can be specified as input, update alpha
        using stochastic gradient descent"""

        """STEP 1: Retrieve the logs of the alpha-derivatives in each model"""
        num_run_lengths = int(len(self.all_retained_run_lengths))
        all_log_alpha_derivatives = -np.inf*np.ones(
                (self.Q, num_run_lengths))
        all_log_alpha_derivatives_sign = np.zeros(
                (self.Q, num_run_lengths))
        all_log_probs = -np.inf*np.ones(
                (self.Q, num_run_lengths))
        #all_log_one_step_preds = -np.inf*np.ones((self.Q, num_run_lengths))

        for m, model in zip(range(0,self.Q), self.model_universe):
            #DEBUG: Unclear if the log derivatives joint log probs
            #       are going to be one entry too many (for r=0)

            """get indices relative to all run lengths"""
            model_indices_indicators_relative_to_all_run_lengths = np.in1d(
                    self.all_retained_run_lengths, model.retained_run_lengths)
            """If the current model has retained run length r>t-1, set it
            to t."""
            if (model.retained_run_lengths[-1] == model.retained_run_lengths[-2]):
                model_indices_indicators_relative_to_all_run_lengths[
                            -1] = True
            all_log_alpha_derivatives[m,
                model_indices_indicators_relative_to_all_run_lengths] = (
                model.log_alpha_derivatives_joint_probabilities )
            all_log_alpha_derivatives_sign[m,
             model_indices_indicators_relative_to_all_run_lengths] = (
             model.log_alpha_derivatives_joint_probabilities_sign)
            all_log_probs[m,
             model_indices_indicators_relative_to_all_run_lengths] = (
             model.joint_log_probabilities)
            #needed if we base loss on P(y_t|y_1:t-1) PROLBEM: This happens AFTER the update, i.e.
            #we have our mrld updated, but the old predictives
            #all_log_one_step_preds[m,
            # model_indices_indicators_relative_to_all_run_lengths] = (
            # model.one_step_ahead_predictive_log_probs)

        #r0_log_prob
        """STEP 2: Sum over all the joint log probs' derivatives"""
        sum_derivatives, sum_derivatives_sign = scipy.misc.logsumexp(
                a = all_log_alpha_derivatives,
                b = all_log_alpha_derivatives_sign,
                return_sign = True)

        """STEP 3: obtain the log of the gradient of P(r_t, m_t|y_1:t).
        NOTE: np.abs(term_1_sign) gives you all the run-length entries per
        model that are non-zero, so multiplying by it ensure we leave zero
        entries zero!"""
        term_1 = all_log_alpha_derivatives - self.log_evidence
        term_1_sign = all_log_alpha_derivatives_sign
        term_2 = sum_derivatives - 2.0*self.log_evidence + all_log_probs
        term_2_sign = (-1) * sum_derivatives_sign * np.abs(term_1_sign)

        run_length_and_model_log_der, run_length_and_model_log_der_sign = (
            scipy.misc.logsumexp(
                a = np.array([term_1, np.abs(term_1_sign) * term_2]),
                b = np.array([term_1_sign, term_2_sign]),
                return_sign = True,
                axis = 0
            ))

        """STEP 4: Lastly, get the gradient of the posterior expectation
        NOTE: This need not be the gradient. We could equivalently use any
        posterior property Q that can be computed for each (r_t, m_t) in closed
        form (i.e. we get a weighted posterior using the MRLD(r,m) * Q(r,m))"""
        if True: #self.loss_type == "posterior_expectation":
            """STEP 4.1: Get the deviation from posterior expectation"""
            resid =  self.y_pred_mean.flatten() - y.flatten()
            #self.y_pred_var

            """STEP 4.2: Get the derivative of the posterior expectation"""
            post_mean_der = np.zeros(shape=(self.S1 * self.S2))
            for (m, model) in zip(range(0, self.Q), self.model_universe):

                """Get the number of stored run lengths for this model"""
                num_rl = np.size(model.retained_run_lengths)

                """get indices relative to all run lengths"""
                model_indices_indicators_relative_to_all_run_lengths = np.in1d(
                        self.all_retained_run_lengths, model.retained_run_lengths)
                """If the current model has retained run length r>t-1, set it
                to t."""
                if (model.retained_run_lengths[-1] == model.retained_run_lengths[-2]):
                    model_indices_indicators_relative_to_all_run_lengths[
                                -1] = True

                """weighing expectation with model & run-length probabilities'
                derivatives"""
                post_mean_der = (post_mean_der +
                    np.sum(
                    np.reshape(model.get_posterior_expectation(t),
                        newshape = (num_rl, self.S1 * self.S2)) *
                    #overflow in exp!
                    (np.exp(run_length_and_model_log_der[m,
                        model_indices_indicators_relative_to_all_run_lengths]) *
                        run_length_and_model_log_der_sign[m,
                        model_indices_indicators_relative_to_all_run_lengths])
                        [:,np.newaxis],
                        axis = 0)
                    )

        if False: #self.loss_type = "posterior_expectation"
            post_prob_log = -np.inf
            post_prob_der_log = -np.inf
            post_prob_der_log_sign = 1.0


            for (m, model) in zip(range(0, self.Q), self.model_universe):
                """Get the number of stored run lengths for this model"""
                num_rl = np.size(model.retained_run_lengths)

                """get indices relative to all run lengths"""
                model_indices_indicators_relative_to_all_run_lengths = np.in1d(
                        self.all_retained_run_lengths, model.retained_run_lengths)
                """If the current model has retained run length r>t-1, set it
                to t."""
                if (model.retained_run_lengths[-1] == model.retained_run_lengths[-2]):
                    model_indices_indicators_relative_to_all_run_lengths[
                                -1] = True
                #DEBUG: Just done to ensure that we finish, root problem unclear
                if (np.sum(model_indices_indicators_relative_to_all_run_lengths)
                    > np.size(model.one_step_ahead_predictive_log_probs)):
                    one_steps = np.insert(model.one_step_ahead_predictive_log_probs,
                                      0, model.r0_log_prob)
                else:
                    one_steps = model.one_step_ahead_predictive_log_probs
                """rlm log:"""
#                print("mrl", self.model_and_run_length_log_distr[m][
#                            model_indices_indicators_relative_to_all_run_lengths].shape)
#                print("one steps", one_steps.shape)
                sum_ = scipy.misc.logsumexp(
                    a = np.array([
                        self.model_and_run_length_log_distr[m][
                            model_indices_indicators_relative_to_all_run_lengths],
                        one_steps
                    ]))
                post_prob_log = scipy.misc.logsumexp(
                        a = np.array([post_prob_log, sum_])
                        )
                """rlm log derivative"""
                sum_, sign_ = scipy.misc.logsumexp(
                    a = np.array([
                        run_length_and_model_log_der[m,
                            model_indices_indicators_relative_to_all_run_lengths],
                        one_steps
                        ]),
                    b = np.array([
                        run_length_and_model_log_der_sign[m,
                            model_indices_indicators_relative_to_all_run_lengths],
                        np.ones(num_rl)]),
                    return_sign = True)
                post_prob_der_log, post_prob_der_log_sign = scipy.misc.logsumexp(
                    a = np.array([post_prob_der_log, sum_]),
                    b = np.array([post_prob_der_log_sign, sign_]),
                    return_sign = True)

#                rlm_log_der = (np.exp(run_length_and_model_log_der[m,
#                        model_indices_indicators_relative_to_all_run_lengths])*
#                        run_length_and_model_log_der_sign[m,
#                        model_indices_indicators_relative_to_all_run_lengths])
#                """rlm log"""  #probably should leave everything in log form
#                rlm_log = self.model_and_run_length_log_distr[m,
#                    model_indices_indicators_relative_to_all_run_lengths]
#                post_prob= (post_prob +
#                    np.exp(model.one_step_ahead_predictive_log_probs) *
#                    np.exp(rlm_log))
#                post_prob_der = (post_prob_der +
#                    np.exp(model.one_step_ahead_predictive_log_probs) *
#                    np.exp(rlm_log_der))

        #resid =  post_prob_log

        """STEP 4.3: Plug both into the loss to get the gradient"""
        self.gradient_alpha_rld = (self.gradient_alpha_rld  +
                self.loss_der_rld_learning(resid.flatten(),
                                           post_mean_der.flatten(),
                                           self.C)
                )
        self.gradient_alpha_rld_count = self.gradient_alpha_rld_count + 1
#        alpha_gradient = self.loss_der_rld_learning(resid.flatten(),
#                                                    post_mean_der.flatten(),
#                                                    self.C)
        #DEBUG: Make constant steps

        """Note: Set number of observations you make before taking a gradient
        step, i.e. we estimate the gradient with k observations"""
        #k=1

        """STEP 4.4: Gradient descent step"""
        #relative = pow(10, -1) * 5 * np.abs(self.alpha)
        #t_eff = len(self.alpha_list) #number of updates
        #C_1, C_2= 1, 1000 #C_1*((C_2+1)/(t_eff+C_2)) #ad hoc  C_1 * np.exp(-C_2 * t)#
        step_size = self.step_size_gen(t=t)#, alpha=self.alpha_rld)
        min_increment, max_increment = 0.0000, 5/self.T #5/self.T #0.001
        #grad_sign = np.sign(alpha_gradient)
        min_alpha, max_alpha = pow(10,-5), 5

        """Note: this will simply be equal to alpha_gradient for standard
        SGD, where we take a step at each obs"""
        #self.gradient = self.gradient + (1.0/k)*alpha_gradient
        if update:
            #print("RLD gradient size:", self.gradient_alpha_rld/self.gradient_alpha_rld_count)
            #print("gradient ", self.gradient)
            """Update step"""
            #NOTE: Since our loss functions have MINIMA we want to find, we need to
            #       do gradient DESCENT, not ASCENT! (for log probability,
            #       we do need to do ascent though)

            #DEBUG: bound the gradient step to avoid chaotic behaviour
            grad_sign = np.sign(self.gradient_alpha_rld)
            increment = max(min(max_increment,
                                step_size *
                                np.abs(self.gradient_alpha_rld) *
                                (1.0/ self.gradient_alpha_rld_count)),
                            min_increment)
            self.alpha_rld = min(
                    max(self.alpha_rld - increment*grad_sign, min_alpha),
                    max_alpha)
            self.alpha_list.append(self.alpha_rld)

            """STEP 4.5: Update the alphas inside the model objects"""
            for model in self.model_universe:
                #DEBUG: once we change alpha name in detector, we nee to
                #       do the same in models, too
                model.alpha_rld = self.alpha_rld
            self.gradient_alpha_rld_count = 0
            self.gradient_alpha_rld = 0


    @staticmethod
    def step_size_gen(t, alpha = None):
        """gives you sequence 1/n"""
        if alpha is not None:
            g0 = min(alpha, 1.0)#pow(10,-10) #2*(alpha)
            lamb = 10# max(2.0, alpha) #pow(10,10) #1.0 - alpha
            step_size = g0/(1.0 + g0 * lamb * t)
            return step_size
        else:
            g0 = 3.0 #pow(10,-10) #2*(alpha)
            lamb = 0.5 # max(2.0, alpha) #pow(10,10) #1.0 - alpha
            step_size = g0/(1.0 + g0 * lamb * t)
            return step_size
            #return pow(t, -1)
#        if alpha is None:
#            return (pow(t, -1))
#        else:
#            return (alpha*pow(t, -1))

    @staticmethod
    def step_size_gen_rld(t, alpha = None):
        #slowly decreasing magnitude, but small step sizes
        g0 = 0.05
        lamb = 0.5
        step_size = g0/(1.0 + g0 * lamb * t)
        return step_size

    @staticmethod
    def eps_gen(t):
        """gives you sequence (n)^-0.25"""
        return pow(t, -0.25)

    @staticmethod
    def squared_loss(resid,C):
        """multivariate squared loss function. Univariate output."""
        return 0.5*np.sum(np.power(resid, 2))

    @staticmethod
    def absolute_loss(resid,C):
        """just returns the absolute loss, i.e. sum( |Y_pred - Y_t|_i )"""
        return np.sum(np.abs(resid))

    @staticmethod
    def biweight_loss(resid, C):
        """Get biweight loss"""
        smallerC = np.where(resid < C)
        biggerC = int(len(resid) - len(smallerC))
        return(0.5*np.sum(np.power(resid[smallerC],2))+ 0.5*biggerC*pow(C,2))

    @staticmethod
    def bounded_absolute_loss(resid, C):
        """Get biweight loss"""
        smallerC = np.where(resid < C)
        biggerC = int(len(resid) - len(smallerC))
        return(np.sum(np.abs(resid[smallerC]))+ biggerC*C)

    @staticmethod
    def squared_loss_derivative(resid, post_mean_der,C):
        """Gives the deriative of sum((resid)^2) wrt alpha_rld"""
        return(np.sum(2 * resid * post_mean_der))

    @staticmethod
    def absolute_loss_derivative(resid, post_mean_der,C):
        """returns deriative of sum(|resid|) wrt alpha_rld"""
        return(np.sum(np.sign(resid) * post_mean_der))

    @staticmethod
    def biweight_loss_derivative(resid, post_mean_der, C):
        """gives derivative of the huber loss with constant C"""
        smallerC = np.where(resid < C)
        #note: Where resid > C, there we have a flat loss, i.e. gradient=0
        return(np.sum(2 * resid[smallerC] * post_mean_der[smallerC]))

    @staticmethod
    def bounded_absolute_loss_derivative(resid, post_mean_der, C):
        """gives derivative of the huber loss with constant C"""
        smallerC = np.where(resid < C)
        #note: Where resid > C, there we have a flat loss, i.e. gradient=0
        return(np.sum(np.abs(resid[smallerC]* post_mean_der[smallerC])))


    #step_size_gen eps_gen

#    @staticmethod
#    def log_loss_derivative(log_prob):
#        """returns 1/x"""
#        return 1.0/np.exp(log_prob)


    def save_negative_log_likelihood(self,t):
        """Get the negative log likelihood if you want to store it"""
        #DEBUG: Unclear why model and run length log distr has different length
        #       Maybe because of trimming?
        all_one_step_ahead_lklhoods_weighted = [
            (self.model_universe[m].one_step_ahead_predictive_log_probs +
             self.model_and_run_length_log_distr[m,
                        self.model_and_run_length_log_distr[m,:]> -np.inf])
             #self.model_universe[m].retained_run_lengths])
             #np.where(self.model_and_run_length_log_distr[m,:]> -np.inf)])
             for m in range(0, self.Q)]
        summed_up = -misc.logsumexp([item for entry in
            all_one_step_ahead_lklhoods_weighted for item in entry])
        self.negative_log_likelihood.append(summed_up)
        #logsumexp

    def compute_negative_log_likelihood_fixed_pars(self, y,t):
        """For comparison with GP, fix sigma^2 at the MAP when doing the
        one-step-ahead prediction. Just compute them, store them later once
        the run-length distro is updated"""
        for (m,model) in zip(range(0, self.Q), self.model_universe):
            if model.has_lags: #I.e., BVAR family
                #do stuff differently, retrieve
                model.save_NLL_fixed_pars(y,t)

    def save_negative_log_likelihood_fixed_pars(self, t):
        """For comparison with GP, fix sigma^2 at the MAP when doing the
        one-step-ahead prediction."""
        all_one_step_ahead_lklhoods_weighted = [
            (self.model_universe[m].one_step_ahead_predictive_log_probs_fixed_pars +
             self.model_and_run_length_log_distr[m,
                        self.model_and_run_length_log_distr[m,:]> -np.inf])
            for m in range(0, self.Q)]
             #self.model_universe[m].retained_run_lengths])
             #np.where(self.model_and_run_length_log_distr[m,:]> -np.inf)])

        summed_up = -misc.logsumexp([item for entry in
            all_one_step_ahead_lklhoods_weighted for item in entry])
        self.negative_log_likelihood_fixed_pars.append(summed_up)



    def save_MSE(self,y,t):
        """Get the MSE at t and store it"""
        #DEBUG: Much more natural to do when we already compute posterior expectation...
        #self.y_pred_mean - y
#        DEBUG = True
#        if (DEBUG and pow(self.y_pred_mean - y.reshape(self.S1, self.S2),2) > 25
#            and isinstance(self.model_universe[0], BVARNIGDPD)):
#            model = self.model_universe[0]
#            all_exp = model.get_posterior_expectation(t)
#            map_ = np.argmax(self.model_and_run_length_log_distr)
#            max_rl = model.retained_run_lengths[map_]
#            print("t = ", t)
#            print("run length distro max at", max_rl)
#            print("expectation at map", all_exp[map_])
#            print("params at map", model.a_rt[map_], model.b_rt[map_], model.beta_rt[map_,:], model.L_rt[map_,:,:])


        self.MSE.append(pow(self.y_pred_mean - y.reshape(self.S1, self.S2),2))
        self.MAE.append(abs(self.y_pred_mean - y.reshape(self.S1, self.S2)))


    #IMPLEMENTED FOR ALL SUBCLASSES IF model_evidence UPDATED CORRECTLY IN SUBLCASS
    def update_log_evidence(self):
        """Sum up all the model-specific evidences from the submodels"""
        self.log_evidence = scipy.misc.logsumexp([model.model_log_evidence
                                for model in self.model_universe])


    #DEBUG: Not needed!
    def update_model_and_run_length_log_distribution(self, t):
        """Using the updated evidence, calculate the distribution of the
        run-lengths and models jointly by accessing each model in order to
        retrieve the joint probabilities. These are then scaled by 1/evidence
        and the result is stored as the new *model_and_run_length_distr*

        NOTE: If one does NOT compute the MAP, then it is more efficient not to
        store this quantity, and rather compute it directly when the next
        observation y is predicted!
        """

        """For each model object in model_universe, get the associated
        *joint_probabilities* np arrays, divide them by *self.evidence*
        and store the result. Notice that at time t, we will have t+1 entries
        in the joint_probabilities for r=0,1,...t-1, >t-1"""


        """STEP 1: Get the longest non-zero run-length. Keep in mind that we
        may have r=t-1 and r>t-1, but both are stored as same run-length!"""
        r_max, r_max2= 0,0
        for q in range(0, self.Q):
            retained_q = self.model_universe[q].retained_run_lengths
            """STEP 1.1: Check if in model q, we have a larger maximum
            run-length than in previous models"""
            r_max2 = max(np.max(retained_q), r_max)
            if r_max2 >= r_max:
                r_max = r_max2
                """STEP 1.2: If a new maximum was found, check if one retains
                both r=t-1 and r>t-1 in this model. If so, advance r_max"""
                if ((retained_q.shape[0]>1) and (retained_q[-1] ==
                    retained_q[-2])):
                    r_max = r_max + 1

        #r_max = np.max( [model.retained_run_lengths
        #                 for model in self.model_universe])

        """STEP 2: Initialize the model-and-run-length log distribution"""
        self.model_and_run_length_log_distr = (-np.inf *
                np.ones(shape=(self.Q, r_max + 1))) #Easier: t+1

        """STEP 3: Where relevant, fill it in"""
        for i in range(0, self.Q):

            """STEP 3.1: Get the retained run-lengths"""
            model = self.model_universe[i]
            retained = model.retained_run_lengths
            if ((retained.shape[0]>1) and (retained[-1] == retained[-2])):
                retained = np.copy(model.retained_run_lengths)
                retained[-1] = retained[-1] + 1

            """STEP 3.2: Update the model-and-run-length-log-distribution
            corresponding to the current model being processed"""
            #DEBUG: Make sure that BVARNIG just has -inf, -inf in there.
            self.model_and_run_length_log_distr[i,retained]=(
                    model.joint_log_probabilities - model.model_log_evidence)

        """STEP 4: If we want to store the run-length distribution at each
        time point, do so here"""
        if self.store_mrl:
            #DEBUG: Needs implementation!
            print("storage for model-and-run-length log distr not implemented")

    #DEBUG: Make sure you call posterior expectation + variance for BVAR only
    #       once suff. t for pred. achieved. I.e., we want 0-probability
    #       assigned to BVAR-predictions before suff. lag length
    def prediction_y(self,y, t):
        """Using the information of all potential models and run-lengths,
        make a prediction about the next observation. In particular,
        you want the MAP/Expected value as well as the standard deviation/
        variance to put CIs around your predicted value"""

        """for each model in the model universe, extract the posterior
        expectation and the posterior variance for each run length,
        weighted by model_and_run_length_distr, and sum them up.
        This yields the posterior mean and posterior variance under
        model uncertainty"""

        #Q: Start prediction only once ALL models initialized?
        #A: seems to be the correct way.

        post_mean , post_var = (np.zeros(shape=(self.S1, self.S2)),
            np.zeros(shape=(self.S1*self.S2, self.S1*self.S2)))
        for (m, model) in zip(range(0, self.Q), self.model_universe):
            """simple reweighting of the means of each (q,r) combination by
            the appropriate probability distribution"""

            """Get the number of stored run lengths for this model"""
            num_rl = np.size(model.retained_run_lengths)

            #DEBUG: I want to circumvent the exp-conversion!
            #       Callable via LogSumExp.logsumexp(...)
            """weighing expectation with model & run-length probabilities"""
            post_mean = (post_mean +
                    np.sum(
                            np.reshape(model.get_posterior_expectation(t),
                                newshape = (num_rl, self.S1, self.S2)) *
                            np.exp(model.joint_log_probabilities  -
                                self.log_evidence)[:,np.newaxis, np.newaxis],
                    axis = 0))

            """weighing the variance with model & run-length probabilities"""
            post_var = (post_var +
                    np.sum(
                        np.reshape(model.get_posterior_variance(t),
                            newshape = (num_rl, self.S1*self.S2,
                                        self.S1*self.S2)) *
                        np.exp(model.joint_log_probabilities  -
                             self.log_evidence)[:, np.newaxis,np.newaxis],
                    axis = 0) )

        """lastly, store the new posterior mean & variance in the relevant
        object"""
        self.y_pred_mean, self.y_pred_var = post_mean, post_var


    def storage(self, t):
        """helper function, just stores y_pred into storage_mean & storage_var
        so that we can always access the last computed quantity"""

        #Q: Best to only store it once all models initialized?
        #A: Probably best (and easiest)

        self.storage_mean[t-1, :, :]  = self.y_pred_mean
        #self.storage_var[t-1, :, :] = self.y_pred_var
        self.storage_var[t-1, :] = np.diag(self.y_pred_var)

        self.storage_log_evidence[t-1] = self.log_evidence


    def trim_run_length_log_distributions(self, t):
        """Trim the distributions within each model object by calling a trimmer
        on all model objects in the model universe. Pass the threshold down"""
        for model in self.model_universe:
            #NOTE: Would be ideal to implement this on probability_model level!
            if model.has_lags:
                if model.lag_length<t:
                    model.trim_run_length_log_distrbution(t, self.threshold,
                                                          self.trim_type)
            else:
                model.trim_run_length_log_distrbution(t, self.threshold,
                                                      self.trim_type)


    #DEBUG: Relocate to probability_model
    def update_priors(self, t):
        """update the priors for each model in the model universe, provided
        that the boolean auto_prior_update is true"""
        for (m, model) in zip(range(0,self.Q), self.model_universe):
            if model.auto_prior_update == True:
                """We need to weigh quantities acc. to rld"""
                model_specific_rld = (self.model_and_run_length_log_distr[m,:] -
                    misc.logsumexp(self.model_and_run_length_log_distr[m,:]))
                model.prior_update(t, model_specific_rld)

    def MAP_estimate(self, t):
        """Using the information of all potential models and run-lengths,
        get the MAP segmentation & model estimate.
        """

        """STEP 1: get all quantities we need for one iteration of the MAP
        segmentation. *log_MAP_current* and *log_MAP_proposed* will track the
        values of MAP_t. The *initializer* bool will indicate if the highest
        current MAP_t value as from initialization or a recursive update.
        *r_cand*,*m* is the run-length/model associated with log_MAP_proposed,
        and *r_star*, *m_star* hold the optimal run-length/model at the end
        of the current iteration"""
        log_MAP_current, log_MAP_proposed = -np.inf, -np.inf #-9999999, -9999999
        initializer, CP_initialization = False, False
        CP_proposed = [-99, -99]
        r_cand, r_star = -99,-99
        m, m_star = 0, self.m_star_old

        """STEP 1.5: I need a different version of 'all run lengths' than
        the one used for mrl distro"""
        all_rl = np.array([])
        for model in self.model_universe:
            all_rl = np.union1d(all_rl, model.retained_run_lengths)
            if model.retained_run_lengths[-1] == model.retained_run_lengths[-2]:
                    #DEBUG: changed r_more from '+1' to 't'
                    #r_more = np.array([model.retained_run_lengths[-1]+1])
                    r_more = np.array([t])
                    all_rl = np.union1d(all_rl, r_more)
        all_rl = all_rl.astype(int)

        """STEP 2: Loop over the model universe and check for each model if the
        *log_MAP_proposed* quantity implied by that model will beat the current
        maximum achieved."""
        for (m,model) in zip(range(0, self.Q), self.model_universe):

            if model.has_lags:
                model_lag = model.lag_length
            else:
                #DEBUG: changed model lag
                #model_lag = 1
                #DEBUG: CONSTANT FITTING ADAPTION
                model_lag = 0

            """STEP 2.1: If you have model_lag = t, this means that the model
            is initialized at this time point t, implying that *initializer*
            is set to True and you only check run-lengths 0 and <0.
            NOTE: If model_lag>t, then you just skip the round altogether."""
            if model_lag+1  == t:
                initializer = True
                log_MAP_0 = model.joint_log_probabilities[0] #+
                             #np.log(self.model_prior[m]))#r=0
                log_MAP_larger_0 = model.joint_log_probabilities[1] #+
                                    #np.log(self.model_prior[m]))#r<0
                if log_MAP_0 > log_MAP_larger_0:
                    r_cand = 0 #t-1
                    log_MAP_proposed = log_MAP_0
                else:
                    r_cand = t+1 #t
                    log_MAP_proposed = log_MAP_larger_0

            """STEP 2.2: If you have model_lag < t, the model has already been
            initialized. This means the *initializer* is set to False and you
            check all run-lengths stored inside the model object"""
            if model_lag+1  < t:
                initializer = False

                #log_densities = -np.inf*np.ones(t)
                """Get log densities s.t. we have one entry for every retained
                run length (and thus for every log MAP storage entry)"""
                log_densities = -np.inf*np.ones(np.size(all_rl))
                        #self.all_retained_run_lengths))

                """Get the run-lengths as indices. If we store r=t-1 and r>t-1
                at once, make sure that this is captured in the object."""
                model_run_lengths = model.retained_run_lengths.copy() #+ model_lag - 1
                if model_run_lengths[-1] == model_run_lengths[-2]:
                    #DEBUG: changed from '+1' to 't'
                    model_run_lengths[-1] = t #model_run_lengths[-1]+1
                    #print("model rl after mod:", model_run_lengths)
                #all_rl = self.all_retained_run_lengths.copy()
                #if (all_rl[-1] == all_rl[-2]):
                #    all_rl[-1] = all_rl[-1] + 1

                """All run-lengths that are kept of the current model are
                stored away in the correct entry of log_densities"""
                index_indicators = np.in1d(all_rl,
                                           model_run_lengths)


                #DEBUG: Renormalized version
                log_densities[index_indicators] = (
                        model.joint_log_probabilities
                        - self.log_evidence)

                """Solve the maximization problem"""
                MAP_factor = np.flipud(self.log_MAP_storage)[all_rl]
                candidates = (log_densities + MAP_factor)
                        #DEBUG: MAP factor replaces t-r_all-2 accessing
                        #(
                        #self.log_MAP_storage[t-all_rl-2]))#self.all_retained_run_lengths-2])) #self.all_retained_run_lengths])) #[model.retained_run_lengths]) )
                log_MAP_proposed = np.max(candidates)

                #DEBUG: Unclear hat happens if this selects r>t-1!
                #DEBUG: self.all_retained_run_lengths instead (?)
                r_cand = (all_rl)[np.argmax( candidates )]
                #if r_cand = t, then set r_cand  = t-1
                if r_cand == t-1:
                    r_cand = t-2
                if r_cand == t:
                    r_cand = t-1


            """STEP 2.3: If the highest MAP_t value that could be achieved
            with the current model is higher than that of any previous models,
            store the new maximum"""
            if log_MAP_proposed > log_MAP_current:
                log_MAP_current = log_MAP_proposed
                CP_initialization = initializer
                m_star = m
                r_star = r_cand
                CP_proposed = [t-r_star,m_star]


            m += 1

        """STEP 3: Record the new CP and model if any. I.e., store the new
        MAP-maximum in log format and append to the list of CPs and models"""
        #DEBUG: Now, we just append a value.
        self.log_MAP_storage = np.append(self.log_MAP_storage, log_MAP_current)#[t-1] = log_MAP_current #problem log map storage
            #indices are not aligned with the run-lengths?!

        if CP_initialization: #or (t == r_star + self.model_universe[m_star].lag_length + 1):
            """STEP 3.1: If we choose a model that has just been initialized,
            replace the entry of CPs at the relevant position if necessary."""
            #self.CPs.append([CP_proposed])
            self.CPs[t-1] = [CP_proposed]
        elif log_MAP_current > -np.inf: #-9999999:
            #DEBUG: This makes no sense. We append a new model only if the lag
            #       length and
            """We are at time t=1,2,3... and e want to get the segmentations
            from before time t, i.e. from times t-1, t-2, ..., with run-
            length r_t via t-r_t, where r_t=0,1,2... Now note that we hence
            have to access the optimal segmentation for time [(t-1) - r_t],
            which is stored at index [(t-1) - r_t - 1] = t-r_t-2."""

            """STEP 3.2: If we choose the same model as in the last time period
            with the new run-length being 1+old run-length, then we don't want
            to have a new list but just copy the last one. If we choose a
            completely different model, we want to overwrite CPs."""
            if (r_star  == self.r_star_old + 1 and m_star == self.m_star_old):
                self.CPs[t-1] = self.CPs[t-2]
            else:
                self.CPs[t-1] = self.CPs[t-2 - r_star].copy() + [CP_proposed]

        self.r_star_old = r_star
        self.m_star_old = m_star



    @staticmethod
    def default_step_size_param_learning(t, alpha = None):
        """Used if no other learning rate specified"""
        if alpha is None:
            return (pow(t, -1))
        else:
            return (alpha*pow(t, -1))
