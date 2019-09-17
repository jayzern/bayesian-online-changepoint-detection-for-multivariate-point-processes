#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:14:05 2018

@author: jeremiasknoblauch

Description: Implements the VB-approximation of the Density Power Divergence
             parameter inference for the NIG model inside BOCPDMS
"""


import numpy as np
from scipy import special
from scipy import linalg
from scipy import stats
import scipy
from probability_model import ProbabilityModel
#from nearestPD import NPD
from BVAR_NIG import BVARNIG


class BVARNIGDPD(BVARNIG):


    """Implements the constructor using the superclass BVARNIG"""
    def __init__(self,
             prior_a,
             prior_b,
             S1,
             S2,
             alpha_param = 0.5,
             prior_mean_beta=None,
             prior_var_beta=None,
             prior_mean_scale=0,
             prior_var_scale=100,
             nbh_sequence=None,
             restriction_sequence = None,
             intercept_grouping = None,
             general_nbh_sequence = None,
             general_nbh_restriction_sequence = None,
             exo_selection = None,
             padding = 'overall_mean',
             auto_prior_update=False,
             hyperparameter_optimization = "online",
             general_nbh_coupling = "strong coupling",
             non_spd_alerts =False,
             VB_window_size = 25,
             full_opt_thinning = None,
             full_opt_thinning_schedule = None,
             SGD_batch_size = 3,
             anchor_batch_size_SCSG = 30,
             anchor_batch_size_SVRG = None,
             first_full_opt = 3
            ):
        """Take inputs, relabel them"""
        prior_a = prior_a
        prior_b = prior_b
        S1, S2 = S1,S2
        prior_mean_beta=prior_mean_beta
        prior_var_beta=prior_var_beta
        prior_mean_scale=prior_mean_scale
        prior_var_scale=prior_var_scale
        nbh_sequence=nbh_sequence
        restriction_sequence = restriction_sequence
        intercept_grouping = intercept_grouping
        general_nbh_sequence = general_nbh_sequence
        general_nbh_restriction_sequence = general_nbh_restriction_sequence
        exo_selection = exo_selection
        padding = padding
        auto_prior_update=auto_prior_update
        hyperparameter_optimization = hyperparameter_optimization
        general_nbh_coupling = general_nbh_coupling
        non_spd_alerts =non_spd_alerts
        """instantiate object"""
        super(BVARNIGDPD, self).__init__(
            prior_a = prior_a ,
            prior_b = prior_b ,
            S1 = S1, S2 = S2,
            prior_mean_beta=prior_mean_beta,
            prior_var_beta=prior_var_beta,
            prior_mean_scale=prior_mean_scale,
            prior_var_scale=prior_var_scale,
            nbh_sequence=nbh_sequence,
            restriction_sequence = restriction_sequence,
            intercept_grouping = intercept_grouping,
            general_nbh_sequence = general_nbh_sequence,
            general_nbh_restriction_sequence = general_nbh_restriction_sequence,
            exo_selection = exo_selection,
            padding = padding,
            auto_prior_update=auto_prior_update,
            hyperparameter_optimization = hyperparameter_optimization,
            general_nbh_coupling = general_nbh_coupling,
            non_spd_alerts =non_spd_alerts)

        """add features to object that are unique to DPD version"""
        self.VB_window_size = VB_window_size
        if VB_window_size is None: #in this case, we have W = max(RL)
            self.VB_window_size = 101 #extend/prune in units of 100
            self.dynamic_VB_window_size = True
            self.max_rl = 1
        else:
            self.dynamic_VB_window_size = False
            self.max_rl = 1
        self.alpha_param_learning = False #initialized via Detector init
        self.alpha_param = alpha_param
        self.full_opt_thinning = full_opt_thinning

        """If we want to have a different opt. schedule"""
        #if full_opt_thinning_schedule is not None:
        self.full_opt_thinning_schedule = full_opt_thinning_schedule

        self.SGD_approx_goodness = SGD_batch_size
        self.first_full_opt = first_full_opt
        self.failed_opt_count = 0
        self.opt_count = 0

        #This ensures that the trimmer only trims the _p_eps/_m_eps quantities
        #if they are ever created.
        self.a_rt_p_eps = None
        #Similarly to a_rt_p_eps being initialized as None, we use this for
        #the trimmer only
        self.predictive_variance_log_det = None
        self.alpha_param_list = []
        self.N_j = None
        if anchor_batch_size_SCSG is None:
            if self.dynamic_VB_window_size == False:
                self.anchor_batch_size_SCSG = self.VB_window_size
            else:
                self.anchor_batch_size_SCSG = 100
        else:
            self.anchor_batch_size_SCSG = anchor_batch_size_SCSG
        if anchor_batch_size_SVRG is None:
            if self.dynamic_VB_window_size == False:
                self.anchor_batch_size_SVRG = self.VB_window_size
            else:
                self.anchor_batch_size_SVRG = 100
        else:
            self.anchor_batch_size_SVRG = anchor_batch_size_SVRG

        #bounds for optimization



    #initialization needs to ininitalize different objects to be tracked,
    #e.g. L (the Cholesky decomp of Sigma), and a storage for the X'X
    #instead of a storage for the M and so on. Note that big parts of
    #the BVARNIG object can be copied 1:1
    def initialization(self, X_endo, X_exo, Y_2, X_exo_2, cp_model, model_prior,
        padding_columns_computeXX = None, padding_column_get_x_new = None):
        """Initialize the model (i.e. t=lag length) with some inputs from the
        containing Detector object. The padding_column arguments are only
        needed for the demo Csurf object. This method differs from object
        instantiation/creation, as it processes the very first (collection of)
        observation(s), thus creating the objects and quantities we will trace
        through time.

        NOTE I:    The exo_selection list is applied inside detector, so X_exo
                    will already contain everything relevant
        NOTE II:   The tag #QR ADAPTION means that the lines following the tag
                    could/can be adapted to QR-updating (rather than Woodbury)

        X_endo: S1xS2x(L+1) numpy array, float:
            is the S1xS2x(L+1) array of the last L observations before t
            as well as the observation at t at position L.
        Y_2: S1xS2 np array, float:
            will be endogeneous regressors at time t+1, which means Y_t.
        X_exo: S1xS2xnum_exo np array, float:
            will contain exogeneous variables at time t (NOT IMPLEMENTED)
        X_exo_2: S1xS2xnum_exo np array, float:
            will contain exogeneous variables at time t+1 (NOT IMPLEMENTED)
        cp_model: CpModel object:
            gives the hazard function inside an object
        model_prior: float:
            Passes the prior of the Detector object into the model object
        padding_columns_computeXX, padding_column_get_x_new:
            deprecated, leave None.
        """

        print("Initializing BVAR DPD object")

        """STEP 1: Take the data-stream that was partitioned appropriately
        inside the Detector object and reshape/rename it for further processing
        Y1 = Y_t, Y2 = Y_{t+1}, X1_endo = Y_1:t-1, with t = L-1."""
        Y1 = X_endo[-1,:].flatten()
        Y2 = Y_2.flatten()
        X1_endo = X_endo[:self.lag_length,:].reshape(self.lag_length,
                        self.S1, self.S2)

        """In case there are no exogeneous variables in this model, take
        the relevant precautions."""
        if self.exo_bool:
            #RESHAPE will not corr. to real dims of exo vars
            X1_exo = (X_exo[-1,:,:].reshape(
                self.num_exo_regressors, self.S1, self.S2))
        else:
            X1_exo = None

        """STEP 2: Format the quantities we wish to trace through time (i.e.
        typically sufficient statistics), and correctly compute them using
        neighbourhood structure"""

        """STEP 2.1: Quantities for time point t, i.e. dimension does not
        depend on how many run-lengths we retain.

        Quantities will hold:
        XX
            Y_1:t-1'Y_1:t-1, i.e. the cross-product of regressors at time t.
        XY
            Y_1:t-1'Y_t, i.e. the cross-product of regressors and obs. at t
        X_t
            Y_1:t-1, i.e. regressors at time t
        X_tp1
            Y_2:t, i.e. regressors at time t+1 ('t plus (p) 1')
        YY
            Y_t'Y_t, i.e. observation cross-product

        """
        self.XX = np.zeros(shape=(self.num_regressors,self.num_regressors))
        self.XY = np.zeros(self.num_regressors)
        self.X_t = np.zeros(shape=(self.S1*self.S2, self.num_regressors))
        self.X_tp1 = np.zeros(shape=(self.S1*self.S2, self.num_regressors))
        self.YY = np.inner(Y1, Y1)

        """STEP 2.2: Create structures holding the last VB_window_size
        cross-prods XX and YY. Also create structures holding the variational
        parameter estimates.

        Quantities will hold;
        XX_t
            At time t, r-th entry holds the cross-product of X_{t-r}'X_{t-r}
            for any r < VB_window_size, so dimension is
            VB_window_size x num_regressors x num_regressors
            NOTE: VERY DIFFERENT from the XX_rt in super class BVAR_NIG, which
                  contains the SUMS over cross-products from t-r to t!
        YY_t
            As XX_t, but with Y'Y, so its dimension is VB_window_size x 1
            NOTE: VERY DIFFERENT from YY_rt in BVAR_NIG, same reason as above
        """
        self.XX_t = np.zeros(shape=(self.VB_window_size,
                                    self.num_regressors, self.num_regressors))
        self.YY_t = np.zeros(self.VB_window_size)
        self.XY_t = np.zeros((self.VB_window_size, self.num_regressors))
#        self.LBFGSB_anchor = np.zeros((self.VB_window_size,
#            int(self.num_regressors * (self.num_regressors + 1) * 0.5) +
#            self.num_regressors + 2))

        self.XX_t[0,:,:] = self.XX
        self.XY_t[0,:]   = self.XY
        self.YY_t[0]     = self.YY


        """STEP 2.3: Inverse and Determinant related quantities

        #DEBUG: We will want to implement efficient updating for the inv/det
        #       of (X'X* alpha + Sigma^-1) when Sigma^-1 changes!
        #       should be doable via Woodbury or something, but I will brute
        #       force it for the time being.
        #       The gain would be that one needs to have an S1*S2 x S1*S2
        #       inversion for the update, instead of pxp (which presumably
        #       will be larger)
        """

        #DEBUG: FILL IN IF THERE IS A COMPUTATIONALLY EFFICIENT WAY

        """STEP 2.4: Variational parameters at time point t and for run-
        length r, i.e. dimension depends on how many run-lengths one retains

        Quantities will hold:
        beta_rt
            beta_rt[r,:] stores the variational beta param, being the
            MAP-estimate of the approximate posterior at time t, run-length r.
            Dimension is num_run_lengths x num_regressors
        L_rt
            L_[r,:,:] stores the Cholesky decomposition of the inverse variance
            matrix, i.e. L[r,:,;]L[r,:,:]' = Sigma^{-1}. Here Sigma corresponds
            to the variational posterior covariance matrix estimate at time t
            for run-length r.
            Dimension is num_run_lengths x num_regressors x num_regressors
        a_rt
            a_rt[r] stores the variational param for the Inverse Gamma (IG) a
        b_rt
            b_rt[r] stores the variational param for the IG b
        """
        self.beta_rt = np.zeros(shape=(2,self.num_regressors))
        self.L_rt = np.zeros(shape = (2, self.num_regressors,
                                      self.num_regressors))
        self.a_rt = np.zeros(2)
        self.b_rt = np.zeros(2)

        """Initialize a_1, b_1, L_1, beta_1 with the priors. This
        means taking them as starting values for optimization"""
        self.a_rt[0], self.b_rt[0] = self.a, self.b
        self.beta_rt[0,:] = self.prior_mean_beta
        self.successful_terminations = np.array([False, False])


        """STEP 2.5: Retained run lengths, storing which run-lengths you retain
        at time t. Careful with this, as retained_run_lengths[i] = j means that
        the i-th longest run-length you retain is j"""
        self.retained_run_lengths = np.array([0,0])


        """STEP 3: Compute prior- and data-dependent quantities:
        Computation of X_t, X_tp1, X'X,  X'Y, and Y'Y from scratch."""

        """STEP 3.1: Gives X_t, X'X, Y'Y"""
        self.compute_X_XX_XY_YY( Y1, X1_endo, X1_exo,
                                padding_columns_computeXX,
                                compute_XY = True)
        """STEP 3.2: Gives X_{t+1}"""
        self.X_tp1 = self.get_x_new(Y2, X_exo_2 ,1,padding_column_get_x_new)

        """STEP 3.3: Fill the first entry of XX_t, YY_t"""
        self.XX_t[0,:,:], self.YY_t[0] = self.XX, self.YY
        self.XY_t[0,:] = self.XY

        """STEP 4: Using the results of STEP 3, compute some computationally
        burdensome results, like XX_rt's inverses and prior inv + det"""

        """STEP 4.1: Computation of the prior inverse, which will be needed
        at each iteration to inform the chaingepoint probabilities"""
        self.L_0_inv = np.linalg.cholesky(self.prior_var_beta)
        #syntax: the 1 stands for 'lower triangular', since the np cholesky
        #        decomposition returns lower triangular matrices
        self.L_0 = scipy.linalg.lapack.clapack.dtrtri(self.L_0_inv, 1)[0]
        self.prior_var_beta_inv = np.matmul(np.transpose(self.L_0),
                                            (self.L_0))

        self.L_rt[0,:,:] = self.L_0
        """STEP 5: Get L_1, b_1, a_1, beta_1 using our optimization. Set the
        parameters for run-length >0 equal to those for run-length = 0"""
        #NOTE: Wrong cp_prob, but basically amounts to prior_weight = 1 for SGD
        self.anchor_params = [0] # EXTENDED in the first iteration
        #self.SVRG_anchor = []
        self.SVRG_anchor_gradient_indices = [np.array([0])] #NOT extended in the first iteration
        self.SVRG_anchor_sum = [np.zeros(int(self.num_regressors*
                    (self.num_regressors+1)*0.5) + self.num_regressors + 2)] #NOT extended in the first iteration
        self.SVRG_anchor_gradients = [np.array([0])] #NOT extended in the first iteration

        self.full_VB_optimizer(r = 0, num_obs = 1, t=self.lag_length + 1,
                               cp_prob = 1.0, alpha = self.alpha_param)
        """Take care of r=t-1 & r>t-1 manually (usually done in the loops
        later on)"""
        self.anchor_params[-1] = self.anchor_params[0].copy() #r=t-1 & r>t-1
        self.SVRG_anchor_gradient_indices[0] =  (
                 self.SVRG_anchor_gradient_indices[-1].copy())
        self.SVRG_anchor_sum[-1]  = self.SVRG_anchor_sum[0].copy()
        self.SVRG_anchor_gradients[-1] = self.SVRG_anchor_gradients[0].copy()

        #print("LBFGSB anchor", self.LBFGSB_anchor)

        self.a_rt[1], self.b_rt[1] = self.a_rt[0], self.b_rt[0]
        self.beta_rt[1,:] = self.beta_rt[0,:]
        self.L_rt[1,:,:] = self.L_rt[0,:,:]

        #DEBUG: We also need to initialize VB params for alpha +/- eps!

        """STEP 6: Use the parameters to get the value of the predictive
        distribution for RL = 1"""

        """COMPUTATIONAL NOTE:
        PossiblyInefficient, probably: Can we not use that we have the Cholesky
        decomposition of V^-1 and (I + XVX')^-1 = I - X(V^-1 + X'X)^-1X'
        to somehow update (V^-1 + X'X)^-1 using V^-1's cholesky decomp
        plus some updating?
        See here: https://stackoverflow.com/questions/8636518/
                            dense-cholesky-update-in-python?rq=1
        and this c-based module: https://github.com/modusdatascience/choldate
        NOTE: Will be O(S * p^2), so probably better to use direct inversion,
        since we can either invert directly O(S^3) or via Woodbury (O(p^3))"""

        L_inv = scipy.linalg.lapack.clapack.dtrtri(self.L_rt[0,:,:], 1)[0]
        XL = np.matmul(self.X_t, L_inv)

        """Note: Q_0 and R_0 are the decomposition of the posterior predictive
        variance (without the scaling factor a/b)"""
        Q_0, R_0 = np.linalg.qr(
                        np.identity(self.S1*self.S2)
                        + np.matmul(
                            XL, np.transpose(XL)
                        )
                )
        #syntax: the 0 stands for 'upper triangular', since the QR
        #        decomposition returns upper triangular matrices
        R_0_inv = scipy.linalg.lapack.clapack.dtrtri(R_0, 0)[0]
        C_0_inv = (self.a_rt[0]/self.b_rt[0]) * (
                      #inverse of QR = R^-1 * Q^T, as Q orthogonal
                      np.matmul(R_0_inv, np.transpose(Q_0))
            )
        """LINEAR ALGEBRA NOTE:
        Q is unitary, so |det(Q)| = 1. Also, we have a positive definite
        matrix, so det(QR) > 0. Note also that we want the log determinant of
        covariance matrix, not of its inverse!"""
        C_0_log_det = ((self.S1*self.S2) *
                       (np.log(self.b_rt[0]) - np.log(self.a_rt[0]))
                       + abs(np.sum(np.log(np.diag(np.abs(R_0))))))

        """STEP 6.1: This step ensures that we center the MVT at zero, which
        makes the computations inside mvt_log_density easier"""
        resid = Y1 - np.matmul(self.X_t, self.beta_rt[0,:])

        """STEP 6.2: For the first observation, the predictive probability and
        model evidence are equivalent, as the model evidence is computed under
        posterior beliefs (captured by VB parameters) only."""
        self.model_log_evidence = ( np.log(model_prior) +
                BVARNIG.mvt_log_density(resid, C_0_inv, C_0_log_det,
                                        2*self.a_rt[0],
                                        self.non_spd_alerts))

        """STEP 9.2: Multiply the model evidence by the hazard rate/cp prior
        as well as the model prior to get the joint log probs for run-length
        equalling 0 or being >0 (i.e., first CP occured before first obs)"""

        """Numerical stability: Ensure that we do not get np.log(0)=np.inf
        by perturbation"""
        if cp_model.pmf_0(1) == 0:
            epsilon = 0.000000000001
        else:
            epsilon = 0

        """get log-probs for r_1=0 or r_1>0. Typically, we assume that the
        first observation corresponds to a CP (i.e. P(r_1 = 0) = 1),
        but this need not be the case in general."""
        r_equal_0 = (self.model_log_evidence +
                     np.log(cp_model.pmf_0(0) + epsilon))
        r_larger_0 = (self.model_log_evidence +
                     np.log(cp_model.pmf_0(1)+ epsilon))
        self.joint_log_probabilities = np.array([r_equal_0, r_larger_0])

        """STEP 8.3: Get the derivative of the log probs, too, just
        initialize to 1  (since log(1) = 0), initialize with 2 columns
        (for 2 hyperparams: a,b). We may wish to extend this to more params"""
        self.model_specific_joint_log_probabilities_derivative = np.ones((2,2))
        self.model_specific_joint_log_probabilities_derivative_sign = np.ones(
                (2,2))

        """STEP 8.4: Similar to 8.3, but for alpha-optimization. Hence we only
        do this if we have to"""
        if self.alpha_rld_learning:
            self.log_alpha_derivatives_joint_probabilities = None #np.ones(3)
            self.log_alpha_derivatives_joint_probabilities_sign = None #np.ones(3)




    def full_VB_optimizer(self, r, num_obs, t, cp_prob, alpha = None):
        """run_length gives you the run-length of the parameters we want to
        optimize. I.e., if run_length = r, then we optimize using the last r+1
        entries of self.XX_t.
        Note: The num_obs variable gives the VALUE of the run- length, whiile
        r gives its INDEX in all our parameters. (E.g., we could have discarded
        run-length 2, in which case r=2 will refer to run-length 3, thus the
        corresponding num_obs = 3.

        NOTE: This function will overwrite a_rt, b_rt, beta_rt, L_rt, for
        finite-difference based hyperparameter optimization for alpha_param,
        you will need to save the old values for a, b, beta, L first!
        """

#        """STEP 1: Get the current values of the variational params"""
#        self.old_a, self.old_b, self.old_beta, self.old_L = (
#            self.a_rt[r], self.b_rt[r], self.beta_rt[r,:], self.L_rt[r,:,:])

        """STEP 2: For each variable to be optimized, take a step into the
        current direction along the gradient until we are either
        close enough to the optimum or have taken max_iter steps. Inside
        each of the optimizer functions, we compute:
            (1) new step sizes (unless count = 0, in which case we take some
                default step size that is not dependent on old gradients and
                old values)
            (2) new gradient values, stored in self.old_gradient_a,
                 self.old_gradient_b,  self.old_gradient_beta,
                 self.old_gradient_L

        Note:   max_count = maximum iterations if LOCAL optimizer used
                count     = current iteration count if LOCAL optimizer used
                increment = current increment if LOCAL optimizer used
                eps       = min. amount of change we need to make another
                            iterations if LOCAL optimizer used
                local_opt = whether to use LOCAL optimizer (NOT RECOMMENDED)
                LBFGSB    = whether to use L-BFGS-B as implemented in python
                            (RECOMMENDED)
        """
        if alpha is None:
            alpha = self.alpha_param

        """STEP 2.2: Set up quantities that will be needed in the calls to
        gradient functions"""
        p = self.num_regressors

        eps = pow(10,-5)
        self.bounds = ([( 1+  eps, self.a + int(num_obs*0.75))] +
                [(1+ eps, None)] + [(None, None)]*(p + int(p*(p+1)*0.5)))


#        print("r", r)
#        print("SVRG gradients", self.SVRG_anchor_gradients)

        """STEP 2.2: Use the setup to run the optimization as required"""
        need_full_opt = False
        if not self.full_opt_thinning_schedule is None:
            need_full_opt = (
                    (self.retained_run_lengths[r] == 0)
                    or
                    ((self.retained_run_lengths[r] in
                    self.full_opt_thinning_schedule) and
                    (self.retained_run_lengths[r] < self.VB_window_size)))
        elif not self.full_opt_thinning is None:
            need_full_opt = (
                (self.retained_run_lengths[r] % self.full_opt_thinning == 0)
                and (self.retained_run_lengths[r] < self.VB_window_size))
        if self.retained_run_lengths[r] == self.first_full_opt:
            need_full_opt = True

        #if ((self.retained_run_lengths[r] % self.full_opt_thinning == 0 or
        #     self.retained_run_lengths[r] == self.first_full_opt) and
        #    self.retained_run_lengths[r] < self.VB_window_size):
        if need_full_opt:
            #    or self.retained_run_lengths[r]==0):
            #Note: If r=0, then we want global opt.
            """Uses python's internal optimization (from scipy library)"""
            self.precomputations_VB_optimizer(r, num_obs, alpha, True)
            """max_iter = max number of iterations, max_ls_iter = max number
            of iterations in line search part, maxcor = maximum number of terms
            used for storing the approximation to the Hessian"""
            #print("LBFGSB, r = ", r)
            max_iter, max_ls_iter, maxcor = 15000, 20, 10 #default values
            self.LBFGSB_optimizer(r=r,num_obs=num_obs,alpha = alpha, t=t,
                #params = old_params, *args,
                max_iter=max_iter, max_ls_iter=max_ls_iter, maxcor=maxcor)
        #elif ((self.retained_run_lengths[r]) % self.full_opt_thinning != 0
        #        and self.retained_run_lengths[r] <= self.VB_window_size):
        elif (self.retained_run_lengths[r] <= self.VB_window_size):
            self.SGD_VB_optimizer(r, num_obs, t, cp_prob, alpha)
            #if (r) % self.full_opt_thinning == 0 or r ==1:
            #    print("SVRG, r = ", r)
            #DEBUG: This still does not do the right thing with the indices!
#            self.SVRG_VB_optimizer(r,num_obs,t,cp_prob,alpha,
#                                   anchor_approx_goodness=min(
#                                          self.VB_window_size-1,
#                                          self.anchor_batch_size_SVRG),
#                                    SCSG = False)

        else: #((self.retained_run_lengths[r]) % self.full_opt_thinning != 0
              #  and self.retained_run_lengths[r]+1 > self.VB_window_size):
            #self.SGD_VB_optimizer(r, num_obs, t, cp_prob, alpha)
            #DEBUG: We need r+1 because the entire SVRG function is coded up
            #       to basically take r-1

            """NOTE: Unclear if this should be called before or after opt"""
#            print("r", r)
#            self.SVRG_anchor_gradients = [
#                    e[np.where(e < self.VB_window_size)] for e in
#                    self.SVRG_anchor_gradients ]
#            self.SVRG_anchor_gradient_indices = [
#                    e[np.where(e < self.VB_window_size)] for e in
#                    self.SVRG_anchor_gradient_indices ]
            #DEBUG: We need to also prune the gradients themselves!
#            print("VB windo size", self.VB_window_size)
            #print("self.SVRG_anchor_gradient_indices", self.SVRG_anchor_gradient_indices)
            #print("max", np.max(np.array([np.max(e) for e in self.SVRG_anchor_gradient_indices])))

            self.SVRG_VB_optimizer(r,num_obs,t,cp_prob,alpha,
                           anchor_approx_goodness=min(
                            self.VB_window_size-1,
                            self.anchor_batch_size_SCSG),
                            SCSG = True)
            #now we need to make sure that the gradient indices < window size
        if self.retained_run_lengths[r] == 0 and self.N_j is not None:
#                print("N_j", self.N_j)
#                print("r", r)
#                print("retained rls", self.retained_run_lengths)
                rl_num = len(self.retained_run_lengths)
#                if self.retained_run_lengths[-1] == self.retained_run_lengths[-2]:
#                    rl_num = rl_num-1
                if len(self.N_j) < rl_num:
                    self.N_j = np.insert(self.N_j,0, 0)


            #self.SCSG_VB_optimizer(r,num_obs,t,cp_prob,alpha)


    def SVRG_VB_optimizer(self, r, num_obs, t, cp_prob,alpha,
                          #params, *args,
                          anchor_approx_goodness = None,
                          SCSG = False):
        """Implements the variance-reduction SGD modification proposed by
        Johnson & Zhang (NIPS, 2013) for the first stage of optimization where
        we use the full VB optimizer as an anchoring point/variance reduction
        point

        Note: anchor_approx_goodness gives the number of terms you use for
              your 'anchor', i.e. for the gradient approximation that keeps
              SGD from exploring too much of our space, thus reducing its
              variance
        """

        if SCSG == False:
            r_ = r -1
        elif SCSG == True:
            r_ = r #Note: SCSG is only entered for r > window_size!
        run_length = self.retained_run_lengths[r_]
        #+1
        p = self.num_regressors

        #REMARK: If SCSG = True, we create N_j!
        if SCSG == True and self.N_j is None:
            run_length_num = len(self.retained_run_lengths)
            if self.retained_run_lengths[-1] == self.retained_run_lengths[-2]:
                self.N_j = np.zeros(run_length_num)
            else:
                self.N_j = np.zeros(run_length_num+1)
#        if SCSG:
#            print("N_j", self.N_j)
#            print("r_", r_)
#            print("retained rls", self.retained_run_lengths)
#        elif (SCSG == True and
#              r_ == self.VB_window_size-1):
#            self.N_j = np.insert(self.N_j, 0, 0)
            #print(self.N_j)
#        elif SCSG == True and self.N_j is not None and r_ == 0:
#            #add a new entry
#            self.N_j = np.insert(self.N_j, 0, 0)

        #REMARK: Extend the objects fo r=0!


        """STEP 1: If this is the first step after a full optimization, you
        need to do some extra work"""
        #REMARK: Unchanged under SCSG!

        #REMARK: Unclear if we need to change this under SCSG (depends on
        #        how/when we set our new 'maximum'! If we again set it at
        #        run_length % thinning == 0 (and do nothing else there), this
        #        block can remain unchanged.)
        #REMARK: If we work with N_j[r], then this will simply be entered if
        #        we have that N_j[r] == 0!
        last_full_opt = False
        if not self.full_opt_thinning_schedule is None:
            last_full_opt = ((run_length - 1) in
                    self.full_opt_thinning_schedule)
        elif not self.full_opt_thinning is None:
            last_full_opt = (
                (run_length - 1) % self.full_opt_thinning == 0)
        #if self.retained_run_lengths[r] == self.first_full_opt:
        #    last_full_opt = True

        if ( #((run_length - 1) % self.full_opt_thinning == 0)
            last_full_opt or
            (SCSG == True and self.N_j[r_] == 0)):

            """STEP 1.1: Draw a sample from the retained observations,
            typically using the entire window size.
            Note: We fill XX_t, XY_t, ... from the front
            """
            #REMARK: For SCSG, we would need the anchor_approx_goodness here
            #        s.t. we can just sample our batch size for the variance
            #        reduction.
#            if SCSG:
#                print("reanchoring entered")
            if anchor_approx_goodness is None:
                rl_ = min(run_length, self.VB_window_size-1)
                indices = np.linspace(0,rl_, rl_+1, dtype = int)
            else:
                num_ind = min(min(anchor_approx_goodness, run_length),
                              self.VB_window_size-1)
                rl_ = min(run_length, self.VB_window_size-1)
                indices = np.random.choice(rl_, size = num_ind,
                                           replace = False)
                #DEBUG: If we want to do original SVRG, we have to
                #use the entire sample, but there are modifications men-
                #tioned by M. Jordan using only a subsample.

            """STEP 1.2: Store the drawn indices to make sure that you can
            still access them for STEP 2 at later SVRG iterations.
            Note: We will have to shift these indices backwards for each new,
            observation, so as to retain the correct data point for an index"""
            #DEBUG: I need this to be a list with at most window_size entries
#            if run_length == 1:
#                self.SVRG_anchor_gradient_indices.insert(0,indices)
#            else:
            #print("indices shape", indices.shape)
#            print("self.SVRG_anchor_gradient_indices[r_] init BEFORE", self.SVRG_anchor_gradient_indices[r_])
#            print("self.SVRG_anchor_gradient_indices init BEFORE", self.SVRG_anchor_gradient_indices)
            self.SVRG_anchor_gradient_indices[r_] = indices
#            print("self.SVRG_anchor_gradient_indices[r_] init AFTER", self.SVRG_anchor_gradient_indices[r_])
#            print("self.SVRG_anchor_gradient_indices init AFTER", self.SVRG_anchor_gradient_indices)

            """STEP 1.3: Store the gradients for that sample & compute
            the average of the stored gradients (if you store the entire window
            this doesn't need to be computed, you can just retrieve it from
            the optimization object created in LBFGSB_optimizer)"""

            """STEP 1.3.1: Repackage params for the grad_ELBO function"""
#            lower_ind = np.tril_indices(p,0)
#            params = np.zeros(int(p*(p+1)*0.5) + p + 2)
#            params[0]  = max(1.00001, self.a_rt[r])
#            params[1]  = max(1.00001, self.b_rt[r] )#+
#            params[2:(p+2)] = self.beta_rt[r,:]
#            params[(p+2):] = self.L_rt[r,:,:][lower_ind]

            #REMARK: Maybe we should change the name from LBFGSB_anchor to just
            #        'anchor', and then change the anchor within SVRG later on
            #        in the iterations
            params = self.anchor_params[r_]

            """STEP 1.3.2: Package remaining arguments in args"""
            prior_weight = 1.0/np.size(self.SVRG_anchor_gradient_indices[r_])
            args = (self.a, self.b, self.prior_mean_beta, self.prior_var_beta_inv,
                    alpha, len(self.SVRG_anchor_gradient_indices[r_]), p,
                    self.K, self.E3, self.Rinv, self.E2_m_ba,
                    self.digamma_diff_1, self.digamma_diff_2, self.gamma_ratio_2,
                    self.downweight1, self.downweight2,
                    self.XX_t, self.XY_t, self.YY_t,
                    self.L_0, self.S1*self.S2, prior_weight,
                    self.SVRG_anchor_gradient_indices[r_], False)
                        #false = we do not want the sum!

            """STEP 1.3.3: Perform the call to the negative ELBO gradient and
            store the individual gradients as well as their sum"""
            grads = BVARNIGDPD.get_grad_neg_ELBO_full(
                    params,*args)
            a_sum = (np.sum(grads,axis=0)/np.size(
                        self.SVRG_anchor_gradient_indices[r_]))
            self.SVRG_anchor_gradients[r_] = grads
            self.SVRG_anchor_sum[r_] = a_sum


        """STEP 2: Take SGD steps with variance reduction as in Johnson & Zhang
        (NIPS, 2013) with the minor modification that we always use the
        most recent observation.
        NOTE: We may want to take multiple SGD steps per time period, in which
        case we have to specify another input at BVARNIGDPD level.
        """
        #REMARK: Under SCSG, this has to be b_j, so it can/should still be
        #        constant. Given their recommendations, probably b_j = 2 or 3
        #        would be best (s.t. you don't have complete ordering)
        num_SGs = min(self.SGD_approx_goodness,
                      min(run_length, self.VB_window_size))
        num_steps = min(run_length, min(self.VB_window_size, 1)) #DEBUG: Let this potentially be another input!

        #DEBUG: Choose this smart, as in the papers
        #REMARK: should be b_j/B_j ^ -(2/3)
        #if SCSG == True:
        step_size = (pow(num_SGs/anchor_approx_goodness, 2.0/3.0))#*
                     #np.ones(int(p*(p+1)*0.5)+2+p))
        #step_size[0] = ab_scaling*(1/beta_l_downscaling)*  step_size[0]
        #step_size[1] = ab_scaling*(1/beta_l_downscaling)* step_size[1]
        #elif SCSG == False:v
        #    step_size = pow(num_SGs/self.VB_window_size, 2.0/3.0) #unclear what I should go for here
        lower_ind = np.tril_indices(p,0)

        #REMARK: If we want to have step_count > 1, then for SCSG, we have the
        #       additional challenge of splitting up the steps between
        #       different observations. Probably easiest would be to set an
        #       object-internal counter (for each run-length) which is de-cre
        #       mented for each SGD step taken s.t. you can extend it over
        #       multiple observations

        if SCSG == True:
            if self.N_j[r_] == 0:
                self.N_j[r_] = (np.random.geometric( #p = 1/100,
                    p = anchor_approx_goodness/(anchor_approx_goodness + num_SGs),
                    size= 1))
            num_steps_this_cycle = int(min(self.N_j[r_], num_steps))
        elif SCSG == False:
            num_steps_this_cycle = num_steps #SVRG
        #if num_steps > self.N_j:


        #if num_steps_this_cyle == self.N_j
        for step_count in range(0, num_steps_this_cycle):
            """STEP 2.0: Decrement N_j by one"""
            if SCSG == True:
                self.N_j[r_] = self.N_j[r_] - 1

            """STEP 2.1: Get the indices & gradients of SGs used next step"""
            #if SCSG == False:
            #    max_ind = min(run_length,self.VB_window_size)
            #elif SCSG == True:
            max_ind = self.SVRG_anchor_gradients[r_].shape[0]#min(run_length,
                        #min(self.VB_window_size-1, max(anchor_approx_goodness, 1)))
            indices = np.random.choice(max_ind, size = min(num_SGs, max_ind),
                                       replace = False)  #random draw of size num_SGs + most recent one



            """STEP 2.2: do the num_SG computations for parameter update"""

            #DEBUG: Unclear if we should take this step
            """STEP 2.2.2: Update mu (i.e. the anchor sum) by evaluating
            the new gradient on the anchor optimum, stored in LBFGSB_anchor
            Note: Only needs to be done in the very first SGD batch
            Note: 1 gradient eval.
            """
            if step_count == 0:
                indices[0] = 0 #i.e. for the first SGD batch, we always want
                                #the most recent obs. to be included.

                """STEP 2.2.2.1: Compute the gradients of new observation
                and store into the SRVG anchor gradients.
                Note: Since we invoke precomputations, the args will be
                      different after 2.2.2.1, so we have to declare them
                      here. They will also be different from those under
                      2.2.3 onwards.
                """

                """Note that we will want to keep the prior weight of the
                average gradient (the SVRG_anchor_sum) at 1 throughout"""
                grad_size = np.size(self.SVRG_anchor_gradients[r_])
                w = 1/(grad_size + 1)
                prior_weight = w #as we have a full prior in mu already
                args_new_obs = (self.a, self.b, self.prior_mean_beta,
                                    self.prior_var_beta_inv,
                                alpha, 1, p,
                                self.K, self.E3, self.Rinv, self.E2_m_ba,
                                self.digamma_diff_1, self.digamma_diff_2,
                                    self.gamma_ratio_2,
                                self.downweight1, self.downweight2,
                                self.XX_t, self.XY_t, self.YY_t,
                                self.L_0, self.S1*self.S2, prior_weight,
                                np.array([0]), False)
                """Note: because num_obs = 1 (i.e. specified_indices =
                np.array([0])), this will be the gradient for the last obs
                + 1/(n+1) * prior gradient of last obs"""
                new_obs_grad = BVARNIGDPD.get_grad_neg_ELBO_full(
                        self.anchor_params[r_], *args_new_obs).flatten()

                """STEP 2.2.2.2: Update mu (SVRG_anchor_sum)"""
                self.SVRG_anchor_sum[r_] = (
                          (grad_size*w)*self.SVRG_anchor_sum[r_]
                        + w * new_obs_grad
                    )
                #print("self.SVRG_anchor_sum[r]", self.SVRG_anchor_sum[r])
                """STEP 2.2.2.3: Update the SVRG_anchor_gradients with the
                new gradient term """
                #print("new obs grad", new_obs_grad)
                self.SVRG_anchor_gradients[r_] = np.insert(
                        self.SVRG_anchor_gradients[r_],
                        0,
                        new_obs_grad,
                        axis=0
                    )
                #print("self.SVRG_anchor_gradients[r_]", self.SVRG_anchor_gradients[r_])
                #print("AFTER grads:", self.SVRG_anchor_gradients[r_])

                """Note that 0 is just the newest index at this time
                point (which we have added in this very step 2.2.2),
                implying in particular that the index of all other
                gradients is shifted forward by one"""
                self.SVRG_anchor_gradient_indices[r_] = np.insert(
                        self.SVRG_anchor_gradient_indices[r_] + 1,
                        0,
                        0)



            """STEP 2.2.3: Get NEW gradients corresponding to indices
            we randomly drew (stored in indices)
            Note: num_SG gradient evaluations
            """

            """STEP 2.2.3.1: Get the params and args for gradient call"""
            p = self.num_regressors
            #lower_ind = np.tril_indices(p,0)
            params = np.zeros(int(p*(p+1)*0.5) + p + 2)
            params[0]  = max(1.00001, self.a_rt[r])
            params[1]  = max(1.00001, self.b_rt[r] )#+
            params[2:(p+2)] = self.beta_rt[r,:]
            params[(p+2):] = self.L_rt[r,:,:][lower_ind]

            #print("indices", indices)
            C = 1/(1 - (1/(1 + cp_prob))) # = 1/(1-p)
            prior_weight = (1.0/C) * pow(1/(1 + cp_prob), run_length+1)
            args_new_grad = (self.a, self.b, self.prior_mean_beta,
                                    self.prior_var_beta_inv,
                                alpha, np.size(indices), p,
                                self.K, self.E3, self.Rinv, self.E2_m_ba,
                                self.digamma_diff_1, self.digamma_diff_2,
                                    self.gamma_ratio_2,
                                self.downweight1, self.downweight2,
                                self.XX_t, self.XY_t, self.YY_t,
                                self.L_0, self.S1*self.S2, prior_weight,
                                indices, True)

            """STEP 2.2.3.2: Perform gradient call & average result"""
            #DEBUG: Recomputes the precomputation!
            new_batch_grad = ((1/np.size(indices))
                *BVARNIGDPD.get_grad_neg_ELBO_full(params, *args_new_grad))

            """STEP 2.2.4: Get OLD gradients corresponding to indices
            we randomly drew, including the new one under 2.2.2
            Note: NO evaluations, get them from memory!
            """
#            if SCSG:
#                print("ind before old batch", indices)
#                print("r_ at old batch", r_)
#                print("rl", self.retained_run_lengths[r_])
#                print("all rls", self.retained_run_lengths)
#                print("SVRG anchor grads at r_", self.SVRG_anchor_gradients[r_])
#                print("all SVRG anchor grads", self.SVRG_anchor_gradients)
            try:
                old_batch_grad  = np.mean(
                    self.SVRG_anchor_gradients[r_][indices], axis=0)
            except IndexError:
                print("ind before old batch", indices)
                print("r_ at old batch", r_)
                print("rl", self.retained_run_lengths[r_])
                print("all rls", self.retained_run_lengths)
                print("SVRG anchor grads at r_", self.SVRG_anchor_gradients[r_])
                #print("all SVRG anchor grads", self.SVRG_anchor_gradients)
                raise

            """STEP 2.2.5: Get the increment by subtracting old and adding
            new gradient averages + anchor sum"""
            increment = step_size * (new_batch_grad - old_batch_grad +
                             self.SVRG_anchor_sum[r_])
            #scale a,b, beta, L differently
            ab_scaling = 2.5
            beta_scaling = 0.5
            L_scaling = 1
            #increment[ = beta_l_scaling * increment
            increment[0] = ab_scaling*increment[0]
            increment[1] = ab_scaling*increment[1]
            increment[2:(p+2)] = beta_scaling * increment[2:(p+2)]
            increment[(p+2):] = L_scaling * increment[(p+2):]
            #print(increment)
            """Make sure that everything is within bounds when we take
            the step"""
#                print("old", old_batch_grad)
#                print("new", new_batch_grad)
#                #print("self.SVRG_anchor_sum[r]", self.SVRG_anchor_sum[r])
#                print("increment", increment)
#                print("bounds", self.bounds[0][0])
            if self.a_rt[r] - increment[0] > self.a + int(0.75*
                    self.retained_run_lengths[r]):
                """get ratio of how much a-gradient result would be more efficient
                than KL + modify gradients by downweighting accordinly"""
                ratio = (self.a_rt[r] + increment[0])/(self.a +
                        int(0.5*self.retained_run_lengths[r]))
                increment[0] = increment[0] * (1.0/ratio)
                increment[1] = increment[1] * (1.0/ratio)

            if self.a_rt[r] - increment[0] < self.bounds[0][0]:
                ratio = (self.a_rt[r] + increment[0])/self.bounds[0][0]
                increment[0] = increment[0] * (1.0/ratio)
                increment[1] = increment[1] * (1.0/ratio)

            if self.b_rt[r] - increment[1] < self.bounds[1][0]:
                increment[1] = self.bounds[1][0] - self.b_rt[r]

            """STEP 2.2.6: Perform the update, i.e. take a step and re-
            convert everything into matrix/vector objects"""
            if np.isnan(increment).any():
                print("We have a nan increment at position", np.where(increment == True))
            self.a_rt[r] = max(self.a_rt[r] - increment[0], 1.0001)
            self.b_rt[r] = max(self.b_rt[r] - increment[1], 1.0001)
            self.beta_rt[r,:] = self.beta_rt[r,:] - increment[2:(p+2)]
            self.L_rt[r,:][lower_ind] = (self.L_rt[r,:][lower_ind] -
                         increment[(p+2):])

            if step_count == 0 and SCSG == True:
                """We need to prune the too big values for SCSG"""
                #if SCSG == True:
                #DEBUG: incorrect.
                #print("before trimming:", self.SVRG_anchor_gradients[r_])
                l = len(self.SVRG_anchor_gradient_indices)
                self.SVRG_anchor_gradients = [
                        self.SVRG_anchor_gradients[i][
                        np.where(self.SVRG_anchor_gradient_indices[i] <
                        self.VB_window_size)] for i in range(0,l)
                    ]
#                    self.SVRG_anchor_gradients = [
#                             e[np.where(e < self.VB_window_size)] for e in
#                             self.SVRG_anchor_gradients ]
                self.SVRG_anchor_gradient_indices = [
                        e[np.where(e < self.VB_window_size)] for e in
                        self.SVRG_anchor_gradient_indices ]
                #print("after trimming:", self.SVRG_anchor_gradients[r_])

            #REMARK: For SCSG, we would need to have another step here,
            #        namely the updating of our 'best estimate' every
                #        N_j steps (where N_j is random).
            if SCSG == True and self.N_j[r_] == 0:
                """update the anchor. Gradient for this anchor will be computed
                at next iteration!"""
                self.anchor_params[r_][0] = self.a_rt[r_]
                self.anchor_params[r_][1] = self.b_rt[r_]
                self.anchor_params[r_][2:(p+2)] = self.beta_rt[r_,:]
                self.anchor_params[r_][(p+2):] = self.L_rt[r_,:,:][lower_ind]


    def LBFGSB_optimizer(self, r, num_obs, alpha, t,
                         #params, *args,
                         max_iter=15000, max_ls_iter=20, maxcor=10):
        """Use python's LBFGSB function to optimize while r<window_size"""
        #ELBO needs maximization!

        """STEP 1: Get relevant arguments, put VB params in vector"""
        p = self.num_regressors
        lower_ind = np.tril_indices(p,0)
        old_params = np.zeros(int(p*(p+1)*0.5) + p + 2)
#        if self.successful_terminations.any():
#            """Find the closest run-length which has parameters
#            with successful terminations and use it for init."""
#            if (self.successful_terminations[r:]).any():
#                """larger run-lengths will have more observations
#                and a better initialization, thus prefered"""
#                best_ind = r + np.argmax(
#                        self.successful_terminations[r:])
#            else:
#                """If none available, look at shorter R.L.s"""
#                best_ind = np.argmax(
#                        self.successful_terminations)
#            """Initialize params accordingly"""
#
#            """Try using the estimate based on the most data"""
#            old_params[0]  = max(1.00001, self.a_rt[best_ind] ) #self.a + int(self.retained_run_lengths[r]*0.5))#self.a_rt[r])
#            old_params[1]  = max(1.00001, self.b_rt[best_ind] )#+
#            old_params[2:(p+2)] = self.beta_rt[best_ind,:]
#            old_params[(p+2):] = self.L_rt[best_ind,:,:][lower_ind]
#        else:
        #DEBUG: Change this up, we may want to put a_rt[r] back as init.
        #       Unclear what role it plays, but it definitely will make opt.
        #       slower as we restart from far away every timeself.a + int(self.retained_run_lengths[r]*0.5*0.5)
        old_params[0]  = max(1.00001, self.a_rt[r]) #self.a + int(self.retained_run_lengths[r]*0.5*0.5)) #self.a + int(self.retained_run_lengths[r]*0.5))#self.a_rt[r])
        old_params[1]  = max(1.00001, self.b_rt[r] )#+
        old_params[2:(p+2)] = self.beta_rt[r,:]
        old_params[(p+2):] = self.L_rt[r,:,:][lower_ind]

        """STEP 2.3: Package remaining arguments in args"""
        args = (self.a, self.b, self.prior_mean_beta, self.prior_var_beta_inv,
                self.alpha_param, num_obs, p,
                self.K, self.E3, self.Rinv, self.E2_m_ba,
                self.digamma_diff_1, self.digamma_diff_2, self.gamma_ratio_2,
                self.downweight1, self.downweight2,
                self.XX_t, self.XY_t, self.YY_t,
                self.L_0, self.S1*self.S2, 1,
                None, True)

        DEBUG_gradients_optimizer = False
        if DEBUG_gradients_optimizer:

            old_params[0] = self.a_rt[r]
            old_params[1] = self.b_rt[r]

            #Build stuff we need to evaluate the gradient numerically
            num_grads = (self.num_regressors +
                int(self.num_regressors * (self.num_regressors+1) * 0.5) + 2)
            turb = pow(10,-3) #beta, L, a, b is order of perturb

            perturbation = np.zeros(num_grads)
            grad_beta = np.zeros(self.num_regressors)
            grad_L = np.zeros(int(self.num_regressors *
                                  (self.num_regressors+1) * 0.5))

            turb = pow(10,-3)
            perturbation[-1] = turb
            elbobp = self.get_ELBO( perturbation, r, num_obs)
            elbobm = self.get_ELBO( -perturbation, r, num_obs)
            grad_b = (elbobp - elbobm)/(2 * turb)
            #log_grad_b = (np.log(-elbobp) - np.log(-elbobm))/(2 * turb)

            #a
            turb = pow(10,-3)
            perturbation[-1] = 0
            perturbation[-2] = turb
            elboap = self.get_ELBO( perturbation, r, num_obs)
            elboam = self.get_ELBO( -perturbation, r, num_obs)
            grad_a = (elboap - elboam)/(2 * turb)
            #log_grad_a = (np.log(-elboap) - np.log(-elboam))/(2 * turb)

            #beta
            turb = pow(10,-3)
            perturbation[-2] = 0
            for i in range(0, self.num_regressors):
                perturbation[i] = turb
                elbobetap = self.get_ELBO( perturbation, r, num_obs)
                elbobetam = self.get_ELBO( -perturbation, r, num_obs)
                perturbation[i] = 0
                grad_beta[i] = (elbobetap - elbobetam)/(2 * turb)
                #log_grad_beta[i] = (np.log(-elbobetap) - np.log(-elbobetam))/(2 * turb)

            #L
            for i in range(0,int(self.num_regressors * (self.num_regressors+1) * 0.5)):
                perturbation[i + self.num_regressors] = turb
                elboLp = self.get_ELBO( perturbation, r, num_obs)
                elboLm = self.get_ELBO( -perturbation, r, num_obs)
                perturbation[i+ self.num_regressors] = 0
                grad_L[i] = (elboLp - elboLm)/(2 * turb)
                #log_grad_L[i] = (np.log(-elboLp) - np.log(-elboLm))/(2 * turb)

            #compute grads and the elbo
            grad2_a = BVARNIGDPD.get_grad_a(old_params,
                *args)
            grad2_b = BVARNIGDPD.get_grad_b( old_params,
                *args)
            grad2_beta = BVARNIGDPD.get_grad_beta( old_params,
                *args)
            grad2_L = BVARNIGDPD.get_grad_L( old_params,
                *args)
            elbo = BVARNIGDPD.ELBO_fun(old_params,
                *args)
            elbo_obj = self.get_ELBO(perturbation = np.zeros(
                    int(p*(p+1)*0.5) + p + 2), r=r, num_obs=num_obs)

            #comparison
            print("STATIC VS NUMERICAL")
            print("a diff:", abs(grad2_a - grad_a))
            print("b diff:", abs(grad2_b - grad_b))
            print("beta diff:", np.max(np.abs(grad2_beta - grad_beta)))
            print("L diff:", np.max(np.abs(grad2_L - grad_L)))
            print("elbo diff:", abs(elbo-elbo_obj))

        #DEBUG: Put this in initialization, and let user choose different opt
        #       methods in creator

        #DEBUG: Insert optimization for multiple starting values
        """STEP 3: Optimization call"""
        if self.retained_run_lengths[r] >= self.first_full_opt:
            #optimum_not_reached = True
            #num_trials = 0
            #while optimum_not_reached:
            opt_object = scipy.optimize.minimize(
                        fun = BVARNIGDPD.neg_ELBO_fun,
                        x0 = old_params,
                        args = args,
                        method = 'L-BFGS-B', #Other methods not fully explored, but
                                             #seem to work less well (+slower)
                        jac = BVARNIGDPD.get_grad_neg_ELBO_full,
                        bounds = self.bounds,
                        options = {
                                'maxiter':max_iter,
                                'maxls':max_ls_iter,
                                'maxcor':maxcor,
                                }
                )
            #num_trials = num_trials + 1
            """STEP 4: Update parameters with new values"""
            successful_optimization = opt_object.success
            if successful_optimization:
                self.opt_count = self.opt_count + 1
                self.successful_terminations[r] = True
                #optimum_not_reached = False
                new_params = opt_object.x
                self.a_rt[r],self.b_rt[r] = new_params[0], new_params[1]
                self.beta_rt[r,:] = new_params[2:(p+2)]
                self.L_rt[r,:,:][lower_ind] = new_params[(p+2):]
            else:
                self.opt_count = self.opt_count + 1
                self.failed_opt_count = self.failed_opt_count + 1
                new_params = np.zeros(int(p*(p+1)*0.5) + p + 2)
                new_params[0]  = max(1.00001, self.a_rt[r]) #self.a + int(self.retained_run_lengths[r]*0.5))#self.a_rt[r])
                new_params[1]  = max(1.00001, self.b_rt[r] )#+
                new_params[2:(p+2)] = self.beta_rt[r,:]
                new_params[(p+2):] = self.L_rt[r,:,:][lower_ind]
#                print("after opt", opt_object.x)
#                print("before opt", old_params)

                #print("Optimization could not terminate for run length",
                #  self.retained_run_lengths[r], "at time", t)
                #print("start parameters were", old_params)
#                else:
#                    new_params = np.zeros(int(p*(p+1)*0.5) + p + 2)
#                    new_params[0]  = max(1.00001, self.a_rt[r]) #self.a + int(self.retained_run_lengths[r]*0.5))#self.a_rt[r])
#                    new_params[1]  = max(1.00001, self.b_rt[r] )#+
#                    new_params[2:(p+2)] = self.beta_rt[r,:]
#                    new_params[(p+2):] = self.L_rt[r,:,:][lower_ind]
#                    if num_trials == 1 and self.successful_terminations.any():
#                        """Find the closest run-length which has parameters
#                        with successful terminations and use it for init."""
#                        if (self.successful_terminations[r:]).any:
#                            """larger run-lengths will have more observations
#                            and a better initialization, thus prefered"""
#                            best_ind = r + np.argmax(
#                                    self.successful_terminations[r:])
#                        else:
#                            """If none available, look at shorter R.L.s"""
#                            best_ind = np.argmax(
#                                    self.successful_terminations)
#                        """Initialize params accordingly"""
#
#                        """Try using the estimate based on the most data"""
#                        old_params[0]  = max(1.00001, self.a_rt[best_ind] ) #self.a + int(self.retained_run_lengths[r]*0.5))#self.a_rt[r])
#                        old_params[1]  = max(1.00001, self.b_rt[best_ind] )#+
#                        old_params[2:(p+2)] = self.beta_rt[best_ind,:]
#                        old_params[(p+2):] = self.L_rt[best_ind,:,:]
#                    elif ((num_trials == 1 and not
#                          (self.successful_terminations[r:]).any) or
#                            num_trials == 2):
#                        """Try using the priors if no other run-length has
#                        terminated successfully yet."""
#                        old_params[0]  = max(1.00001, self.a +
#                                  int(self.retained_run_lengths[r]*0.5*0.5)) #self.a + int(self.retained_run_lengths[r]*0.5))#self.a_rt[r])
#                        old_params[1]  = max(1.00001, self.b )#+
#                        old_params[2:(p+2)] = self.prior_mean_beta
#                        old_params[(p+2):] = self.L_0
#                        num_trials = 3 #skip this step
#                    if num_trials >2:
#                    optimum_not_reached = False
#                    self.failed_opt_count = self.failed_opt_count + 1
#                    print("Optimization could not terminate for run length",
#                      self.retained_run_lengths[r], "at time", t)
#                    print("start parameters were", old_params)
#                        print(opt_object.message)

        elif (self.retained_run_lengths[r] < self.first_full_opt and
             self.retained_run_lengths[r] > 0):
            #SGD step
            #inject noise to a_rt[0], b_rt[0], ...
#            self.a_rt[r] = self.a_rt[r] + np.random.normal(0,0.25)
#            self.b_rt[r] = self.b_rt[r] + np.random.normal(0,0.25)
#            self.beta_rt[r,:] = self.beta_rt[r,:] + np.random.normal(
#                    0,0.25, size = self.num_regressors)
#            self.L_rt[r,:,:] = np.array(np.random.normal(
#                    0,0.25, size = int(pow(self.num_regressors,2)))).reshape(
#                    self.num_regressors, self.num_regressors)[lower_ind]
            self.SGD_VB_optimizer(r=r, num_obs = 1, t=t, cp_prob=0.5,
                                  alpha = alpha, precompute = False)
            #get new params
            new_params = np.zeros(int(p*(p+1)*0.5) + p + 2)
            new_params[0]  = max(1.00001, self.a_rt[r]) #self.a + int(self.retained_run_lengths[r]*0.5))#self.a_rt[r])
            new_params[1]  = max(1.00001, self.b_rt[r] )#+
            new_params[2:(p+2)] = self.beta_rt[r,:]
            new_params[(p+2):] = self.L_rt[r,:,:][lower_ind]
        elif self.retained_run_lengths[r] == 0:
            #SGD step
            #inject noise to a_rt[0], b_rt[0], ...
#            self.a_rt[r] = self.a_rt[r] + np.random.normal(0,0.25)
#            self.b_rt[r] = self.b_rt[r] + np.random.normal(0,0.25)
#            self.beta_rt[r,:] = self.beta_rt[r,:] + np.random.normal(
#                    0,0.25, size = self.num_regressors)
#            self.L_rt[r,:,:] = np.array(np.random.normal(
#                    0,0.25, size = int(pow(self.num_regressors,2)))).reshape(
#                    self.num_regressors, self.num_regressors)[lower_ind]
#            self.SGD_VB_optimizer(r=r, num_obs = 1, t=t, cp_prob=0.5,
#                                  alpha = alpha, precompute = False)
            #get new params
            new_params = np.zeros(int(p*(p+1)*0.5) + p + 2)
            new_params[0]  = max(1.00001, self.a_rt[r]) #self.a + int(self.retained_run_lengths[r]*0.5))#self.a_rt[r])
            new_params[1]  = max(1.00001, self.b_rt[r] )#+
            new_params[2:(p+2)] = self.beta_rt[r,:]
            new_params[(p+2):] = self.L_rt[r,:,:][lower_ind]


        #DEBUG: Here we need to do more stuff for the SVRG procedure!
        #       I.e., we need to store the gradients for each

        #DEBUG!!! We need to store this optimum for the SVRG procedure FOR EACH
        #        RUN LENGTH! I.e., this needs to be a vector of size at most
        #        threshold that we also TRIM in the trimmer!
        #        Similarly, the gradients we store ALSO need to be run-length
        #        dependent!
        #DEBUG: unclear if needed
        #self.LBFGSB_opt = opt_object.fun
        """check if r + 1 > len(anchor). If so, append this run-length to the
        end. For r+1<=len(anchor), store it in index r"""
        #DEBUG: Try to extend everything here (instead of inside SVRG) and then
        #fill in SVRG
        if self.retained_run_lengths[r] == 0:
            #I.e., the new entry that is created at each iteration for r=0
            self.anchor_params.insert(0,new_params.copy())
            #extend them here (rather than in SVRG)
            self.SVRG_anchor_sum.insert(0, np.zeros(int(p*(p+1)*0.5) + p + 2))
            self.SVRG_anchor_gradient_indices.insert(0, np.array([]))
            self.SVRG_anchor_gradients.insert(0,np.zeros(int(p*(p+1)*0.5) + p + 2))

        else:
            #I.e., we overwrite an older anchor with the new optimum.
            self.anchor_params[r] = new_params.copy()
        #DEBUG: Unclear if needed
        #self.LBFGSB_grad = opt_object.jac
        #DEBUG: Unclear if needed
        #self.SVRG_value = new_params.copy()



    def SGD_VB_optimizer(self, r, num_obs, t, cp_prob, alpha = None,
                         precompute = True):
        """If we don't use all data points, do SGD by taking a step into the
        direction proposed by the latest observation"""
        if alpha is None:
            alpha = self.alpha_param

        """STEP 1: Precompute relevant quantities"""
        n = min(r, self.SGD_approx_goodness) #min(self.VB_window_size, r+1) #n = number of obs we use to obtain the gradient
        #DEBUG: In precomputations, subsample the observations!
        if precompute:
            self.precomputations_VB_optimizer(r=r, num_obs = n, alpha = alpha)
        """COMPUTATIONAL NOTE:
        Gives weight we attach to  prior in the gradient computation. Unlike
        sum over observations, the prior does not decompose over the number of
        observations! We need a prior weight c_t for which sum(c_t,t) ->  1
        as t -> infinity, and we want a convergence that is relatively linear
        s.t. we attach prior weight streched out over all observations.
        SOLUTION: Choose c_t = (1/(1+p))^t * C for some p>0 such that
         C = 1/sum((1/p)^t), and in particular for p = Prob(new CP)"""

        C = 1/(1 - (1/(1 + cp_prob))) # = 1/(1-p)
        prior_weight = (1.0/C) * pow(1/(1 + cp_prob),
                        self.retained_run_lengths[r]+1)
        #Things to try and fix this:
        #   (1) plug in a_rt[r] for self.a [+for all other priors], currently checking
        #   (2) use the entire window for each step (+ possibly with the full prior)
        #   (3) change step sizes, i.e. make them larger



        """params"""
        p = self.num_regressors
        lower_ind = np.tril_indices(p,0)
        old_params = np.zeros(int(p*(p+1)*0.5) + p + 2)
        #DEBUG: Change this up, we may want to put a_rt[r] back as init.
        #       Unclear what role it plays, but it definitely will make opt.
        #       slower as we restart from far away every time
        #DEBUG: Maybe inject this with randomness (normal errors)
        #err_scale = 0.25
        old_params[0]  = max(1.00001, self.a_rt[r])# +
                  #np.random.normal(0,err_scale, 1))
        old_params[1]  = max(1.00001, self.b_rt[r]) # +
                  #np.random.normal(0,err_scale, 1)) #self.a + int(num_obs*0.5)
        old_params[2:(p+2)] = self.beta_rt[r,:]
        old_params[(p+2):] = self.L_rt[r,:,:][lower_ind]

        """STEP 2: Package remaining arguments in args"""
        args = (self.a, self.b, self.prior_mean_beta, self.prior_var_beta_inv,
                self.alpha_param, n, p,
                self.K, self.E3, self.Rinv, self.E2_m_ba,
                self.digamma_diff_1, self.digamma_diff_2, self.gamma_ratio_2,
                self.downweight1, self.downweight2,
                self.XX_t, self.XY_t, self.YY_t,
                self.L_0, self.S1*self.S2, prior_weight,
                None, #For this case, better to sample the indices inside the
                        #precomputations, as we are only interested in the sum
                True
                )

        """args"""


        """STEP 2: Compute all gradients"""
        gradient_a = (1.0/n)*BVARNIGDPD.get_grad_a(old_params, *args)
        gradient_b = (1.0/n)*BVARNIGDPD.get_grad_b(old_params, *args)
        gradient_beta = (1.0/n)*BVARNIGDPD.get_grad_beta(old_params, *args)
        gradient_L = (1.0/n)*BVARNIGDPD.get_grad_L(old_params, *args)

        #DEBUG: trying (1)
#        self.a = a_backup
#        self.b = b_backup
#        self.prior_mean_beta = beta_mean_backup
#        self.prior_var_beta_inv = beta_var_inv_backup


        """STEP 3: Get the step size"""
        #DEBUG: Make this an input option with defaul as 1/t, but allowing
        #       for the form suggested in the SGD tutorial paper
        #step_size = 1.0/(t-self.lag_length) #so we start at 1/VB_window_size
        #the smaller alpha, the smaller the learning rate should be (the
        #steeper the gradients are => so g0 should be smaller, lamb larger)
        g0 = 2*(max(self.alpha_param,0.1)) #initial learning step size
        lamb = max( 0.01, 1.0 - self.alpha_param) #0.25 #learning rate decreases at rate (lamb * t)^-1
        step_size = g0/(1.0 + g0 * lamb * self.retained_run_lengths[r])
        premult = 1.0 #1.0/pow(self.alpha_param, 1/3)
        step_size_ab = premult * 3.0/(1.0+3.0*0.99*self.retained_run_lengths[r]) #separate learning rate for a and b makes huge difference

        """STEP 4: Update the parameters in question"""
        #DEBUG: Might want to do the Average SGD as in tut paper
#        print("grad a", gradient_a)
#        print("grad b", gradient_b)
#        print("step size", step_size)
        #DEBUG: Ensure that a and b don't move too crazily! In particular,
        #       we don't want a to suggest much higher efficiency than KL, and
        #       should assume the overshoot of the a-gradient to be like the
        #       overshoot of the b-gradient. Accordingly, it makes sense then
        #       to let b not increase by more than a (relatively)
        a_increment = gradient_a * step_size_ab
        b_increment = gradient_b * step_size_ab
#        print("a inc", a_increment)
#        print("b inc", b_increment)
#        print("a grad", gradient_a)
#        print("b grad", gradient_b)
        if self.a_rt[r] + a_increment > self.a + int(0.75*
                    self.retained_run_lengths[r]):
            """get ratio of how much a-gradient result would be more efficient
            than KL + modify gradients by downweighting accordinly"""
            ratio = (self.a_rt[r] + a_increment)/(self.a +
                    int(0.5*self.retained_run_lengths[r]))
            a_increment = a_increment * (1.0/ratio)
            b_increment = b_increment * (1.0/ratio)
        self.a_rt[r] = max(self.a_rt[r] + a_increment, 1.0001)
        self.b_rt[r] = max(self.b_rt[r] + b_increment, 1.0001)
        self.beta_rt[r,:] = self.beta_rt[r,:] + gradient_beta * step_size
        lower_ind = np.tril_indices(self.num_regressors, 0)
        self.L_rt[r,:,:][lower_ind] = (self.L_rt[r,:,:][lower_ind] +
                                         gradient_L * step_size)



    def precomputations_VB_optimizer(self, r, num_obs, alpha=None,
                                full = False, #store_sample_indices = False,
                                sample_indices = None,
                                params = None):
        """precompute computationally burdensome quantities that are needed
        for the VB optimization/ELBO derivative.
        NOTE: If store_sample_indices = True, then we sample and subsequently
              store them inside the object"""

        """Unless otherwise specified, work with self.alpha_param"""
        if alpha is None:
            alpha = self.alpha_param
        """Unless you get the parameters as a single vector, refer to the
        object's values for a,b,beta,L"""
        p = self.num_regressors
        lower_ind = np.tril_indices(p, 0)
        if params is not None:
            an, bn = params[0], params[1]
            betan = params[2:(p+2)]
            Lnvech = params[(p+2):]
            Ln = np.zeros((p,p))
            Ln[lower_ind] = Lnvech
        elif params is None:
            an, bn = self.a_rt[r], self.b_rt[r]
            betan, Ln = self.beta_rt[r,:], self.L_rt[r,:,:]

        """If we provide sample_indices, then we should also make sure num_obs
        aligns with the length of sample indices (assuming correct input)"""
        if not sample_indices is None:
            num_obs = np.size(sample_indices)

        """STEP 1: Compute quantities not depending on data"""

        """STEP 1.1: Auxiliary quantities"""
        d = self.S1*self.S2
        self.gamma_ratio_1 = an #DEBUG: I think this quantity is not
                                         #used anywhere, and that I always
                                         #use G(a+1)/G(a) = a directly
        self.gamma_ratio_2 = np.exp(
                scipy.special.gammaln(an +
                    0.5 * alpha * d)
                - scipy.special.gammaln(an)
            )
        self.digamma_diff_1 = (scipy.special.digamma(an + 1) -
                              scipy.special.digamma(an))
        self.digamma_diff_2 = (scipy.special.digamma(an +
                    0.5* alpha * d ) -
                              scipy.special.digamma(an))

        """STEP 1.2: Composite quantities not depending on data"""
        #RUNTIME OVERFLOW due to pow(b,a) since a becomes huge (absorb in fraction)
        self.E2_m_ba = -(
                - abs(np.prod(np.diag(Ln)))
                  * (self.gamma_ratio_2)
                  * pow(2 * np.pi, -0.5*d * alpha)
                  * (1.0/alpha)
            )
        #DEBUG: This is where the runtime overflow occurs! If b is too large
        #       then this quantity becomes 0
        self.downweight1 = pow(bn, -0.5*d*alpha)
        self.downweight2 = pow(bn, -0.5*d*alpha-1)

        """STEP 2: Data-dependent quantities"""

        """STEP 2.1: Compute E_3i terms. Use the cholesky decomposition, as
        we will also want to store the R^-1 terms, and cholesky means we
        can compute both determinant and inverse easily"""
        self.E3 = np.zeros(num_obs)
        self.Rinv = np.zeros((num_obs, self.num_regressors, self.num_regressors))

        """NOTE: Think about updating cholesky factorization instead. I.e.,
        we could update the L_rt cholesky with S1*S2 rank-1 updates of O(p^2),
        but we typically expect the direct re-computation of O(p^3) to be
        faster"""
        #sample between 0 and min(r-1, self.VB_window_size-1) + take most recent one
        #DEBUG: Don't sample completely randomly, but weigh obs. with recency!
        if sample_indices is None:
            """If we do not provide sample indices, sample inside this fct"""
            num_indices = min(r, self.VB_window_size-1)
            sample_indices = np.array([0], dtype = int) #i.e., most recent observation
            if num_indices > 0 and r+1 >= num_obs and full == False:
                sample_indices = np.insert(sample_indices, 0,
                        np.random.choice(num_indices,
                        min(num_obs-1, num_indices),
                        replace=False, p = None)+1)
            elif (r+1 == num_obs or full == True):
                sample_indices = np.linspace(0, num_obs-1, num_obs, dtype=int)

#        """Note: This is needed if we want to keep track of the terms
#        in the gradient, and if we wish to track the individual
#        gradient terms (as in SVRG and SCSG)"""
#        if store_sample_indices:
#            self.sample_indices = sample_indices

        #print("samp ind", sample_indices)

        ind = 0
        for i in sample_indices:
            """COMPUTATIONAL NOTE: Check if this could be done more efficiently
            usingrank-1 updates"""
            """COMPUTATIONAL NOTE: If this cannot be inverted, we need to
            inject some mass along the diagonal"""
            #print("Ln", Ln)
            #print("mult",np.matmul(Ln,
            #            np.transpose(Ln)) )
            try:
                if p>1:
                    Li = np.linalg.cholesky(np.matmul(Ln,
                            np.transpose(Ln)) +
                            alpha * self.XX_t[i,:,:])
                else:
                    Li = np.sqrt(Ln*Ln + alpha*self.XX_t[i,:,:])
            except np.linalg.LinAlgError as e:
                #if 'not positive definite' in str(e):
                    #scale = np.mean(np.abs(Ln[lower_ind]))
                    #err = np.random.normal(0, 1, self.num_regressors)
                try:
                    Li = np.linalg.cholesky(np.matmul(Ln, np.transpose(Ln)) +
                            #10*np.identity(self.num_regressors) +
                            #np.outer(err, err) +
                            np.sum(np.abs(betan))
                            *np.identity(p) +
                            alpha * self.XX_t[i,:,:])
                except np.linalg.LinAlgError as e:
                    Li = np.identity(p)
               # else:
                    #raise
            #syntax: the 1 stands for 'lower triangular', since the np cholesky
            #        decomposition returns lower triangular matrices
            if p>1:
                Li_inv =  scipy.linalg.lapack.clapack.dtrtri(Li, 1)[0]
            else:
                Li_inv = 1.0/Li
            self.Rinv[ind,:,:] = np.matmul(np.transpose(Li_inv), Li_inv)
            self.E3[ind] = abs(np.prod(np.diag(Li_inv)))
            ind = ind + 1

        """STEP 2.2: Compute the K-terms"""
        self.K = np.zeros(num_obs)

        """STEP 2.2.1: Compute the terms not depending on the observations"""
        L_x_beta = np.matmul(np.transpose(Ln), betan)
        vec_part_1 = np.matmul(Ln, L_x_beta)
        E4 = np.inner(L_x_beta, L_x_beta)

        """STEP 2.2.2: Compute the terms depending on observations and put
        the into the i-th entry of K"""
        ind = 0
        for i in sample_indices:
            """get the terms E_{5,i}, E_{6,i}, E_{7,i}"""
            vec = vec_part_1 + alpha * self.XY_t[i,:]
            E5to7i = -np.inner(vec, np.matmul(self.Rinv[ind,:,:], vec))
            self.K[ind] = (bn
                    + 0.5 * (alpha * self.YY_t[i] + E4 + E5to7i)
                )
            ind = ind + 1


    """evaluation needs to be formulated differently since we track
    different objects now"""
    def evaluate_predictive_log_distribution(self, y, t,
                                store_posterior_predictive_quantities = True,
                                alpha_direction = None):
        """Returns the log densities of *y* using the predictive posteriors
        for all possible run-lengths r=0,1,...,t-1,>t-1 as currently stored by
        virtue of the VB parameters.
        The corresponding density is computed for all run-lengths and
        returned in a np array

        Note: This is called BEFORE update_log_distr, so at time t, the
                quantities one tracks through time (like L_rt, a_rt, ...) will
                still only hold L(r,t), a(r, t), ... and so on (rather than
                L(r+1,t+1), a(r+1,t+1) ... ). Similarly, the regressors
                X_t will actually correspond to time point t-1, so we instead
                use the regressors stored inside X_tp1 = X_t+1 for evaluating
                the pred. density of y.
        Note: If store_posterior_predictive_quantities = False, C_t_inv and
                predictive_variance_log_det are not overwritten. This function-
                ality is needed for the alpha_param optimization!
        """

        """STEP 1: Preliminaries.
            - Get y into vector format,
            - get log_densities as container of log predictive densities
            - get C_t_inv[r,:,:] as the posterior precision for run-length r+1
        """

        y = y.flatten()
        run_length_num = self.retained_run_lengths.shape[0]
        log_densities = -np.inf * np.ones(shape=run_length_num)

        """STEP 2: Get the correct VB quantities. If None, use the kept ones.
        If > 0, use those of alpha + eps and if < 0 use those ofalpha - eps"""
        if alpha_direction is None:
            L_rt = self.L_rt
            beta_rt = self.beta_rt
            a_rt, b_rt = self.a_rt, self.b_rt
        elif alpha_direction < 0:
            L_rt = self.L_rt_m_eps
            beta_rt = self.beta_rt_m_eps
            a_rt, b_rt = self.a_rt_m_eps, self.b_rt_m_eps
        elif alpha_direction > 0:
            L_rt = self.L_rt_p_eps
            beta_rt = self.beta_rt_p_eps
            a_rt, b_rt = self.a_rt_p_eps, self.b_rt_p_eps

        """STEP 3: Create the storage containers for posterior predictive
        variance and its determinant if needed."""
        if store_posterior_predictive_quantities:
            self.C_t_inv = np.zeros((run_length_num+1, self.S1*self.S2,
                                 self.S1*self.S2))
            self.predictive_variance_log_det = np.zeros(run_length_num+1)
            self.C_t_inv[0,:,:] = self.C_t_inv_r0
            self.predictive_variance_log_det[0] = (
                self.predictive_variance_r0_log_det)

        """STEP 4: Loop over all retained run-lengths and fill log_densities[r]
        with the predictive log density for run-length r.
        NOTE: We cannot use retained_run_lengths to loop directly, since
                r=t-1 and r>t-1 both have a 0 in there."""
        for r in range(0,run_length_num):

            """STEP 2.1: QR decomposition of the posterior predictive
            variance (without the scaling factor a/b)"""
            if self.S1*self.S2 > 1 or self.num_regressors > 1:
                """Unless we only fit a single constant, do this"""
                L_inv = scipy.linalg.lapack.clapack.dtrtri(L_rt[r,:,:], 1)[0]
                XL = np.matmul(self.X_tp1, L_inv)

                Q, R = np.linalg.qr(
                                np.identity(self.S1*self.S2)
                                + np.matmul(
                                    XL, np.transpose(XL)
                                )
                        )

                """STEP 2.2: Invert R using triangular solver and compute inverse
                (+ log determinant)of the posterior predictive variance"""
                #syntax: the 0 stands for 'upper triangular', since the QR
                #        decomposition returns upper triangular matrices
                R_inv = scipy.linalg.lapack.clapack.dtrtri(R, 0)[0]
                C_t_inv =  (
                          #inverse of QR = R^-1 * Q^T, as Q orthogonal
                          np.matmul(R_inv, np.transpose(Q))
                          )
                predictive_variance_log_det = ((self.S1*self.S2) *
                           (np.log(b_rt[r]) - np.log(a_rt[r]))
                           + abs(np.sum(np.log(np.diag(np.abs(R))))))
            else:
                XL_sq = np.power(self.X_tp1 * 1.0/self.L_rt[r,:,:],2)
                C_t_inv = 1.0/(1 + XL_sq)
                predictive_variance_log_det = (
                        (np.log(b_rt[r]) - np.log(a_rt[r]))*(1+XL_sq) )
            #DEBUG: How are they not nan!?

            """STEP 2.3: If we need to store C_t_inv and its determinant, do so
            This will only NOT happen when we call the alpha-param finite
            difference gradient approximating optimization routine"""
            if store_posterior_predictive_quantities:
                self.C_t_inv[r+1,:,:] = C_t_inv
                self.predictive_variance_log_det[r+1] = (
                        predictive_variance_log_det)

            """STEP 2.3: Evaluate the predictive probability for r_t = r"""
                        #if predictive_variance_log_det is np.nan:
#            print("pred var is", predictive_variance_log_det)
#            print("XL squared", XL_sq)
#            print("C t inv", C_t_inv)
#            print("np.log(b_rt[r]) - np.log(a_rt[r])", np.log(b_rt[r]) - np.log(a_rt[r]))
            resid = y - np.matmul(self.X_tp1, beta_rt[r,:])
            log_densities[r] = (
                    BVARNIG.mvt_log_density(resid,
                        (a_rt[r]/b_rt[r]) * C_t_inv,
                        predictive_variance_log_det,
                        2*a_rt[r], self.non_spd_alerts))

        """STEP 3: return the full log density vector"""
        return log_densities


    def get_log_integrals_power_divergence(self, DPD_call = False):
        """get integrals for power div in log-form"""
        p = self.S1*self.S2
        #run_length_with_0 = np.insert(self.retained_run_lengths.copy() + 1, 0, 0)

        nu_1 =  2* np.insert(self.a_rt, 0, self.a) #(self.a + (run_length_with_0+1.0)*0.5) self.a_rt * 2
        #if DPD_call:
        #    nu_1 = 2 * self.a_rt
        nu_2 = nu_1 * self.alpha_rld + p* self.alpha_rld + nu_1

        C1 = (1.0 + self.alpha_rld) * (special.gammaln(0.5*(nu_1+p)) -
              special.gammaln(0.5*nu_1))
        C2 = (special.gammaln(0.5*(nu_2+p)) - special.gammaln(0.5*nu_2))


        if DPD_call:
            pred_var = np.insert(self.predictive_variance_log_det, 0,
                                 self.predictive_variance_r0_log_det)
        else:
            pred_var = self.predictive_variance_log_det

#        print("C1", C1)
#        print("C2", C2)
#        print("nu1", nu_1)
#        print("nu2", nu_2)
#        print("pred var", pred_var)
#        print("without insertion", self.predictive_variance_log_det)
#        #print("L_rt", self.L_rt)
#        #print("Ctinv", self.C_t_inv)
#        #print("rlw0", run_length_with_0)
#        print("whole xpr", C1 - C2 - nu_1*0.5*p*self.alpha_rld
#                - np.pi*0.5*p*self.alpha_rld #dets)
#                 -  self.alpha_rld * pred_var)
#        print("whole e^xpr", np.exp((C1 - C2 - nu_1*0.5*p*self.alpha_rld
#                - np.pi*0.5*p*self.alpha_rld
#                 -  self.alpha_rld * pred_var)))

        return (C1 - C2 - nu_1*0.5*p*self.alpha_rld
                - np.pi*0.5*p*self.alpha_rld #dets)
                 -  self.alpha_rld * pred_var) #self.predictive_variance_log_det )


    def evaluate_log_prior_predictive(self, y, t,
                            store_posterior_predictive_quantities = True):
        """Basically, this does is as 'evaluate_predictive_log_distribution',
        but using only the prior specs of BVARNIG object to get the
        predictive prob. """

        """STEP 1: Get QR decomposition"""
        #DEBUG: IS THIS CORRECT

        if self.S1*self.S2>1 or self.num_regressors > 1:
            """Do this unless we only fit a single constant"""
            XL = np.matmul(self.X_tp1, self.L_0_inv)
            Q, R = np.linalg.qr(
                            np.identity(self.S1*self.S2)
                            + np.matmul(
                                XL, np.transpose(XL)
                            )
                    )

            """STEP 2: Invert R using triangular solver and compute inverse
            (+ log determinant)of the posterior predictive variance"""
            #syntax: the 0 stands for 'upper triangular', since the QR
            #        decomposition returns upper triangular matrices
            R_inv = scipy.linalg.lapack.clapack.dtrtri(R, 0)[0]

            C_t_inv_r0 = (
                      np.matmul(R_inv, np.transpose(Q))
                      )
            predictive_variance_r0_log_det = ((self.S1*self.S2) *
                       (np.log(self.b) - np.log(self.a))
                       + abs(np.sum(np.log(np.diag(np.abs(R))))))
        elif self.S1*self.S2 == 1 and self.num_regressors == 1:
            XL_sq = np.power(self.X_tp1 * self.L_0_inv[0],2)
            C_t_inv_r0 = 1.0/(1 + XL_sq)
            predictive_variance_r0_log_det = (
                (np.log(self.b) - np.log(self.a))*(XL_sq+1) )


        """STEP 3: If needed, srtore the quantities. NOT needed if we call this
        as part of the alpha finite-difference optimization routine"""
        if store_posterior_predictive_quantities:
            self.C_t_inv_r0 = C_t_inv_r0
            self.predictive_variance_r0_log_det = (
                    predictive_variance_r0_log_det)

        """STEP 4: Compute the density (bound it from below!)"""
        resid = y - np.matmul(self.X_tp1, self.prior_mean_beta)
        return min(0.0, BVARNIG.mvt_log_density(resid,
                (self.a/self.b)*C_t_inv_r0,
                predictive_variance_r0_log_det, 2*self.a, True))

    #DEBUG: Deprecated/not needed (just here for in case it is faultily called)
    def save_NLL_fixed_pars(self, y,t):
        pass


    """Needs major overhaul, will do the heavy lifting i.t.o. optimization"""
    def update_predictive_distributions(self, y_t, y_tm1, x_exo_t, x_exo_tp1,t,
                                    cp_prob,
                                    #deprecated
                                    padding_column_tm1 = None,
                                    padding_column_t = None,
                                    r_evaluations = None):
        """Takes the next observation, *y*, at time *t* and updates the
        sufficient statistics, means & vars corresponding to all potential
        run-lengths r=0,1,...,t-1,>t-1.

        Inputs are:
        y_t
            Observations at time t
        y_tm1
            Observations at time t-1 ('t minus 1', i.e. tm1)
        x_exo_t, x_exo_tp1
            Exogeneous variables at t, t+1
        padding_column_tm1, padding_column_t, r_evaluations
            deprecated/leave None

        Quantities affected by the update are:
            always: XX, , XX_t, X_t, X_tp1, XY, XY_t, YY, YY_t,
                    beta_rt, L_rt, a_rt, b_rt, retained_run_lengths
            never:  C_t_inv_rt (instead updated in evaluate_predictive...)
        """

        """STEP 1: Store the new data point and relevant transformations"""

        """STEP 1.1: Get observations as vectors"""
        y_t, y_tm1 =y_t.flatten(), y_tm1.flatten()
        self.Y_new, self.Y_old = y_t, y_tm1

        """STEP 1.2: Updates X'X, X'Y, Y'Y, XX_rt, XY_rt, YY_rt"""
        self.regressor_cross_product_updates(y_t,  y_tm1, x_exo_t,
                                             t,
                                             #next two are deprecated arguments
                                             padding_column_tm1,
                                             rt_updates = False)

        """STEP 1.3: Retrieves the new regressors in X_tp1 and re-assigns the
        old value of X_tp1 to X_t"""
        self.X_tp1 = self.get_x_new(y_t, x_exo_tp1, t, padding_column_t)

        """STEP 1.4: Write into self.XX_t, self.XY_t & so on"""

        """STEP 1.4.0: Check if we have a dynamic window size (i.e., if we
        always keep all obs until the largest run length)"""

        """STEP 2.1.1: Extend run length"""
        self.retained_run_lengths =  self.retained_run_lengths + 1
        self.retained_run_lengths = np.insert(self.retained_run_lengths, 0, 0)
        run_length_num = self.retained_run_lengths.shape[0]
        self.max_rl = np.max(self.retained_run_lengths)

        rl = int(self.max_rl/100)
        wnd = int(self.VB_window_size/100)
        if self.dynamic_VB_window_size and rl>=wnd:
#            print("rl", rl)
#            print("wnd", wnd)
#            print("max rl", self.max_rl)
#            print("VB size", self.VB_window_size)
            old_window = self.VB_window_size
            self.VB_window_size = 100 +  self.VB_window_size
            XX_new = np.zeros((self.VB_window_size, self.num_regressors,
                               self.num_regressors))
            XY_new = np.zeros((self.VB_window_size, self.num_regressors))
            YY_new = np.zeros((self.VB_window_size))
#            anchor_new = np.zeros((self.VB_window_size,
#                int(self.num_regressors * (self.num_regressors + 1) * 0.5) +
#                self.num_regressors + 2))
            XX_new[:old_window,:,:] = self.XX_t.copy()
            XY_new[:old_window,:] = self.XY_t.copy()
            YY_new[:old_window] = self.YY_t.copy()
            #anchor_new[:old_window,:] = self.LBFGSB_anchor.copy()
            self.XX_t = XX_new.copy()
            self.XY_t = XY_new.copy()
            self.YY_t = YY_new.copy()
            #self.LBFGSB_anchor = anchor_new.copy()
#            print("YY_t size", np.size(self.YY_t))



        """STEP 1.4.1: For XX"""
        old_entries = self.XX_t[:-1,:,:].copy()
        self.XX_t[0,:,:] = self.XX
        self.XX_t[1:,:,:] = old_entries.copy()

        """STEP 1.4.2: For XY"""
        old_entries = self.XY_t[:-1,:].copy()
        self.XY_t[0,:] = self.XY
        self.XY_t[1:,:] = old_entries.copy()

        """STEP 1.4.3: For YY"""
        old_entries = self.YY_t[:-1].copy()
        self.YY_t[0] = self.YY
        self.YY_t[1:] = old_entries.copy()


        """STEP 2: Perform the VB optimization we will stick with (unlike the
        next step, where we aim at improving alpha_param by finite-difference
        approximation to the gradient)"""

        """STEP 2.1: Add a new entry to all quantities that will be updated.
        Parameters that will be newly estimated are initialized using the
        previous value"""

        """STEP 2.1.2: Extend VB parameters, initalize VB(r=0) with VB(r=1)"""
        self.successful_terminations = np.insert(self.successful_terminations.copy(), 0, False)
        self.a_rt = np.insert(self.a_rt.copy(), 0, self.a_rt[0].copy())
        self.b_rt = np.insert(self.b_rt.copy(), 0, self.b_rt[0].copy())
        self.beta_rt = np.insert(self.beta_rt.copy(), 0, self.beta_rt[0,:].copy(), axis=0)
        self.L_rt = np.insert(self.L_rt.copy(), 0, self.L_rt[0,:,:].copy(), axis=0)

        """STEP 2.1.3: Obtain the number of unique RLs so that for the case
        where we retain r = t-1 and r>t-1, we only do the following
        computations once (instead of twice)"""
        if self.retained_run_lengths[-1] == self.retained_run_lengths[-2]:
            unique_run_length_num = run_length_num -1
            have_double = True
        else:
            unique_run_length_num = run_length_num
            have_double = False


        """STEP 2.1.4: in case we later do alpha_param optimization, we need to
        retain the LBFGS anchors & so on here!"""
        if self.alpha_param:
            retained_pre_anchor_params = self.anchor_params.copy()
            retained_pre_SVRG_anchor_gradient_indices = (
                    self.SVRG_anchor_gradient_indices.copy())
            retained_pre_SVRG_anchor_gradients = self.SVRG_anchor_gradients.copy()
            retained_pre_SVRG_anchor_sum = self.SVRG_anchor_sum.copy()
            if self.N_j is not None:
                has_N_j_pre = True
                retained_pre_N_j = self.N_j.copy()
            else:
                has_N_j_pre = False


        for r in range(0, unique_run_length_num):
            """STEP 2.2: For all retained run-lengths, perform the retained
            optimization step based on a method of choice"""
            num_obs = self.retained_run_lengths[r] + 1
            #if num_obs <= self.VB_window_size: # and r > self.lag_length:
            #    """STEP 2.2.1: If r < K, use the global optimizer"""
            #    #print("enters full VB optimizer")
            self.full_VB_optimizer(r = r,
                                   num_obs = num_obs,
                                   t=t,cp_prob = cp_prob,
                                   alpha = self.alpha_param)
            #else:
            #    """STEP 2.2.2: If r>=K, use SGD"""
            #    n = 1 #n = number of observations we average for gradient
            #    self.SGD_VB_optimizer(r = r, num_obs = num_obs, t=t,
            #                          cp_prob = cp_prob,
            #                          alpha = self.alpha_param)

            """STEP 2.3: If we have retained r=t-1 and r>t-1, copy results"""
            if have_double and r == unique_run_length_num-1:
                self.a_rt[-1] = self.a_rt[-2]
                self.b_rt[-1] = self.b_rt[-2]
                self.beta_rt[-1,:] = self.beta_rt[-2,:].copy()
                self.L_rt[-1,:,:] = self.L_rt[-2,:,:].copy()
                self.anchor_params[-1] = self.anchor_params[-2].copy()
                self.SVRG_anchor_gradient_indices[-1] = (
                        self.SVRG_anchor_gradient_indices[-2].copy())
                self.SVRG_anchor_gradients[-1]=(
                        self.SVRG_anchor_gradients[-2].copy())
                self.SVRG_anchor_sum[-1] = self.SVRG_anchor_sum[-2].copy()
                #NOTE: No need to copy N_j here
                #also need to copy the LBFGS quantities!
                #also, need to make sure they are not permanently overwritten
                #when I use the alpha-optimization!


        """STEP 3: Check if we have alpha_param-optimization enabled. If so,
        do the optimization for alpha_old + eps and alpha_old - eps next,
        following exactly the same steps as before. Unlike before, save the
        old values from the previous optimization (which are the ones we are
        going to carry over) first, then overwrite them in the course of
        computing alpha_old +/- eps, and overwrite them back later"""

        if self.alpha_param_learning:
            """STEP 3.1: Save the old parameter values + joint probabilities"""
            retained_L_rt, retained_beta_rt, retained_a_rt, retained_b_rt = (
                    self.L_rt.copy(), self.beta_rt.copy(),
                    self.a_rt.copy(), self.b_rt.copy())
            retained_post_anchor_params = self.anchor_params.copy()
            retained_post_SVRG_anchor_gradient_indices = (
                    self.SVRG_anchor_gradient_indices.copy())
            retained_post_SVRG_anchor_gradients = self.SVRG_anchor_gradients.copy()
            retained_post_SVRG_anchor_sum = self.SVRG_anchor_sum.copy()
            if self.N_j is not None:
                retained_post_N_j = self.N_j.copy()
            else:
                retained_post_N_j = None
            self.anchor_params = retained_pre_anchor_params.copy()
            self.SVRG_anchor_gradient_indices= retained_pre_SVRG_anchor_gradient_indices.copy()
            self.SVRG_anchor_gradients = retained_pre_SVRG_anchor_gradients.copy()
            self.SVRG_anchor_sum = retained_pre_SVRG_anchor_sum.copy()
            if has_N_j_pre:
                self.N_j = retained_pre_N_j.copy()
            else:
                self.N_j = None
            self.eps = 0.01 * pow(t, -0.25) #DEBUG: Make this an option ?


            #PROBLEM: What if we want to optimize over multiple models for the
            #           same value of alpha? Then, the detector needs to call this!
            #           Solution: We could still do all the computations here, and
            #           simply store E[y_t|y_1:t-1, m_t, alpha + eps], so that
            #           inside the detector, we can then average these posterior
            #           expectations using P(m_t|y_1:t-1, alpha+eps) [NOTE that
            #           this implies that we need to store the model-specific
            #           alpha+eps evidence P(y_1:t-1|m_t,alpha+eps)]

            """STEP 3.2: For all retained run-lengths, perform the gradient
            step for alpha_param + eps based on a method of choice"""
            alpha_p = max(pow(10,-10), min(10.0, self.alpha_param + self.eps))
            for r in range(0, unique_run_length_num):
                """STEP 3.2.1: For all retained run-lengths, perform the
                alpha + eps optimization step based on a method of choice"""
                num_obs = self.retained_run_lengths[r] + 1
                #if num_obs <= self.VB_window_size:# and r > self.lag_length:
                """STEP 3.2.2: If r < K, use the global optimizer"""
                self.full_VB_optimizer(r = r,
                                    num_obs = num_obs,
                                    t=t,cp_prob = cp_prob,
                                    alpha = alpha_p)
#                else:
#                    """STEP 3.2.3: If r>=K, use SGD"""
#                    n=1 #number of obs we use for gradient computations
#                    self.SGD_VB_optimizer(r = r,
#                                      num_obs = num_obs, t=t,
#                                      cp_prob = cp_prob,
#                                      alpha = alpha_p)

                """STEP 2.3: If we have retained r=t-1 and r>t-1, copy results"""
                if have_double and r == unique_run_length_num-1:
                    self.a_rt[r+1] = self.a_rt[r]
                    self.b_rt[r+1] = self.b_rt[r]
                    self.beta_rt[r+1,:] = self.beta_rt[r,:].copy()
                    self.L_rt[r+1,:] = self.L_rt[r,:].copy()
#                    self.LBFGSB_anchor[r+1] = self.LBFGSB_anchor[r].copy()
#                    self.SVRG_anchor_gradient_indices[r+1] = (
#                            self.SVRG_anchor_gradient_indices[r].copy())
#                    self.SVRG_anchor_gradients[r+1]=(
#                            self.SVRG_anchor_gradients[r].copy())
#                    self.SVRG_anchor_sum[r+1] = self.SVRG_anchor_sum[r].copy()

            """STEP 3.2.4: Store the resulting parameters s.t. at the next
            iteration, you can update alpha_t based on the predictive loss
            NOTE: This will also require you to obtain the joint log probs
                    for each value of alpha!"""
            self.alpha_param_p_eps = self.alpha_param + self.eps
            self.L_rt_p_eps, self.beta_rt_p_eps,  = (
                    self.L_rt.copy(), self.beta_rt.copy())
            self.a_rt_p_eps, self.b_rt_p_eps = (
                    self.a_rt.copy(), self.b_rt.copy()  )
            self.anchor_params = retained_pre_anchor_params.copy()
            self.SVRG_anchor_gradient_indices= retained_pre_SVRG_anchor_gradient_indices.copy()
            self.SVRG_anchor_gradients = retained_pre_SVRG_anchor_gradients.copy()
            self.SVRG_anchor_sum = retained_pre_SVRG_anchor_sum.copy()
            if has_N_j_pre:
                self.N_j = retained_pre_N_j.copy()
            else:
                self.N_j = None


            """STEP 3.3: Reset the VB parameters s.t. we start optimization at
            the same starting values for both alpha + eps and alpha - eps"""
            self.L_rt, self.beta_rt, self.a_rt, self.b_rt = (
                    retained_L_rt.copy(), retained_beta_rt.copy(),
                    retained_a_rt.copy(), retained_b_rt.copy())
#            retained_LBFGSB_anchor = self.LBFGSB_anchor.copy()
#            retained_SVRG_anchor_gradient_indices = (
#                    self.SVRG_anchor_gradient_indices.copy())
#            retained_SVRG_anchor_gradients = self.SVRG_anchor_gradients.copy()
#            retained_SVRG_anchor_sum = self.SVRG_anchor_sum.copy()

            """STEP 3.4: For all retained run-lengths, perform the gradient
             step for alpha_param - eps based on a method of choice"""
            for r in range(0, unique_run_length_num):
                alpha_m = max(pow(10,-10), min(10.0, self.alpha_param - self.eps))
                """STEP 3.4.1: For all retained run-lengths, perform the
                alpha + eps optimization step based on a method of choice"""
                num_obs = self.retained_run_lengths[r] + 1
                #if num_obs <= self.VB_window_size: # and r > self.lag_length:
                #    """STEP 3.4.2: If r < K, use the global optimizer"""
                self.full_VB_optimizer(r = r,  num_obs = num_obs,
                                       t=t,cp_prob = cp_prob,
                                    alpha = alpha_m)
#                else:
#                    """STEP 3.4.3: If r>=K, use SGD"""
#                    n=1 #number of obs we use for gradient computations
#                    self.SGD_VB_optimizer(r = r, num_obs = num_obs,
#                                          t=t,cp_prob = cp_prob,
#                                          alpha = alpha_m)
                """STEP 2.3: If we have retained r=t-1 and r>t-1, copy results"""
                if have_double and r == unique_run_length_num-1:
                    self.a_rt[r+1] = self.a_rt[r]
                    self.b_rt[r+1] = self.b_rt[r]
                    self.beta_rt[r+1,:] = self.beta_rt[r,:].copy()
                    self.L_rt[r+1,:] = self.L_rt[r,:].copy()

            """STEP 3.4.4: Store the resulting parameters s.t. at the next
            iteration, you can update alpha_t based on the predictive loss
            NOTE: This will also require you to obtain the joint log probs
                    for each value of alpha!"""
            self.alpha_param_m_eps = self.alpha_param - self.eps
            self.L_rt_m_eps, self.beta_rt_m_eps,  = (
                    self.L_rt.copy(), self.beta_rt.copy())
            self.a_rt_m_eps, self.b_rt_m_eps = (
                    self.a_rt.copy(), self.b_rt.copy()  )
            self.anchor_params = retained_post_anchor_params
            self.SVRG_anchor_gradient_indices = (
                    retained_post_SVRG_anchor_gradient_indices)
            self.SVRG_anchor_gradients = retained_post_SVRG_anchor_gradients
            self.SVRG_anchor_sum = retained_post_SVRG_anchor_sum
            #if self.N_j is not None:
            self.N_j = retained_post_N_j

            """STEP 3.5 Write back the quantities we want"""
            self.L_rt, self.beta_rt, self.a_rt, self.b_rt = (
                    retained_L_rt, retained_beta_rt,
                    retained_a_rt, retained_b_rt)


    def alpha_param_gradient_computation(self, y, t, cp_model, model_prior,
                         log_model_posteriors, log_CP_evidence,
                         eps):
        """Implements the optimization of alpha_param w.r.t. a predictive loss
        function of the user's choice. This works by first updating the
        joint log probabilities for alpha + eps, alpha - eps and then using
        the parameter estimates from the previous iteration to do predictions
        in order to approximate the gradient by finite differences."""
        #NOTE: Once you have multiple classes with DPD, it might be worth
        #       putting this into probability_model and just having an
        #       attribute self.alpha_param_optimization in all objects

        #print("reached alpha param grad comp, alpha_param_learning = ", self.alpha_param_learning)

        if self.alpha_param_learning:

            """STEP 1: Set and store the new alpha + eps, alpha - eps"""
            self.alpha_param_p_eps = min(self.alpha_param + eps, 10)
            self.alpha_param_m_eps = max(pow(10,-10),
                                         self.alpha_param - eps)

            """STEP 2: compute the joint log probs for alpha + eps and get
            the posterior expectation corresponding to alpha + eps"""

            """STEP 2.1: Get the joint log probs for alpha + eps.
            If alpha_direction = +1, we investigate alpha + eps.
            If alpha_direction = -1, we investigate alpha - eps"""

            joint_log_probs = self.DPD_joint_log_prob_updater(
                alpha_direction = +1,
                y=y, t=t, cp_model=cp_model, model_prior=model_prior,
                log_model_posteriors = log_model_posteriors,
                log_CP_evidence = log_CP_evidence)

            """STEP 2.2: Obtain the model-specific RLD as
            P(r_t|y_1:t, m_t) = P(r_t,y_1:t,m_t)\P(y_1:t, m_t), where we get
            P(y_1:t, m_t) = P(y_1:t|m_t)P(m_t) by summing the joint probs"""
            self.model_log_evidence_p_eps = (
                scipy.misc.logsumexp(joint_log_probs))

            """STEP 2.3: Get the posterior expectation for alpha + eps, i.e.
            E[y_t+1|y_1:t, m_t, alpha_t + eps]. On the detector level, we
            aggregate back by computing
                E[y_t+1|y_1:t, m_t, alpha_t + eps] *
                P(m_t|y_1:t, alpha_t + eps)
            """
            self.post_mean_p_eps = np.sum(
                np.reshape(
                    np.matmul(
                        self.X_tp1,
                        np.insert(self.beta_rt_p_eps, 0,
                            self.prior_mean_beta, axis=0)[:,:,np.newaxis]
                        )
                        ,
                    newshape = (self.retained_run_lengths.shape[0] + 1,
                                self.S1, self.S2))
                * (np.exp(joint_log_probs - self.model_log_evidence_p_eps)
                        [:,np.newaxis, np.newaxis]),
                axis = 0).flatten()


            """STEP 3: compute the joint log probs for alpha - eps and get
            the posterior expectation corresponding to alpha - eps"""

            """STEP 3.1: Get the joint log probs for alpha + eps.
            If alpha_direction = +1, we investigate alpha + eps.
            If alpha_direction = -1, we investigate alpha - eps"""
            joint_log_probs = self.DPD_joint_log_prob_updater(
                alpha_direction = -1,
                y=y, t=t, cp_model=cp_model, model_prior=model_prior,
                log_model_posteriors = log_model_posteriors,
                log_CP_evidence = log_CP_evidence)

            """STEP 3.2: as before"""
            self.model_log_evidence_m_eps = (
                    scipy.misc.logsumexp(joint_log_probs))

            """STEP 3.3: Get the posterior expectation for alpha + eps"""
            self.post_mean_m_eps = np.sum(
                np.reshape(
                    np.matmul(
                        self.X_tp1,
                        np.insert(self.beta_rt_m_eps, 0,
                            self.prior_mean_beta, axis=0)[:,:,np.newaxis]
                        ),
                    newshape = (self.retained_run_lengths.shape[0] + 1,
                                self.S1, self.S2))
                * (np.exp(joint_log_probs - self.model_log_evidence_m_eps)
                        [:,np.newaxis, np.newaxis]),
                axis = 0).flatten()


    """Definitely needs adapting since Sigma is stored differently"""
    def get_posterior_variance(self, t, r_list=None):
        """get the predicted variance from the current posteriors at
        time point t, for all possible run-lengths."""

        post_var = np.zeros((np.size(self.retained_run_lengths),
                                     self.S1*self.S2, self.S1*self.S2))

        run_length_num = self.retained_run_lengths.shape[0]
        for r in range(0,run_length_num):

            """Q: Can we use the stored C_t_inv?
               A: Depends on whether M_inv_1_rt and X_tp1 inside prediction_y
                   are still the same as inside update_joint_log_probabilities,
                   but they won't be because update_predictive_distributions is
                   called inbetween to update X_t, X_tp1
            """
            #syntax: the 1 stands for 'lower triangular', since the np cholesky
            #        decomposition returns lower triangular matrices
            L_inv = scipy.linalg.lapack.clapack.dtrtri(self.L_rt[r,:,:], 1)[0]
            XL = np.matmul(self.X_tp1, L_inv)
            post_var[r,:,:] = (self.b_rt[r]/self.a_rt[r])*(
                    np.identity(self.S1*self.S2) +
                    np.matmul(XL, np.transpose(XL))
                )

        return post_var

    """Needs adaption since we store (and trim) different quantities"""
    def trimmer(self, kept_run_lengths, BAR_submodel = False):

        """If this BVAR model is a submodel of a BAR model, its joint log
        probs will never be updated/grown. In that case, all it is needed for
        is the evaluation of predictive probabilities"""
        if not BAR_submodel:
            self.joint_log_probabilities = (
                    self.joint_log_probabilities[kept_run_lengths])
        """If we optimize hyperparameters, we also want to discard the deriva-
        tives that are not warranted."""
        if self.hyperparameter_optimization:
            self.model_specific_joint_log_probabilities_derivative_sign = (
                self.model_specific_joint_log_probabilities_derivative_sign[:,
                        kept_run_lengths])
            self.model_specific_joint_log_probabilities_derivative = (
                self.model_specific_joint_log_probabilities_derivative[:,
                        kept_run_lengths])

        """None condition needed to ensure that they are initialized"""
        if (self.generalized_bayes_rld == "power_divergence" and
            self.alpha_rld_learning and
            self.log_alpha_derivatives_joint_probabilities is not None):
            self.log_alpha_derivatives_joint_probabilities = (
                    self.log_alpha_derivatives_joint_probabilities[
                            kept_run_lengths])
            self.log_alpha_derivatives_joint_probabilities_sign = (
                    self.log_alpha_derivatives_joint_probabilities_sign[
                            kept_run_lengths])

        """Discard all quantities of data that have been computed"""
        self.beta_rt = self.beta_rt[kept_run_lengths,:]
        self.L_rt = self.L_rt[kept_run_lengths,:,:]
        self.b_rt = self.b_rt[kept_run_lengths]
        self.a_rt = self.a_rt[kept_run_lengths]
        self.successful_terminations = self.successful_terminations[
                kept_run_lengths]

        """Discard the anchoring quantities that you don't need"""
        if True:
            #print(kept_run_lengths)
            keepers = np.where(kept_run_lengths == True)[0]
            self.anchor_params = [self.anchor_params[ind] for
                                  ind in keepers] #list
    #        self.SVRG_anchor = [self.SVRG_anchor[ind] for
    #                              ind in kept_run_lengths]
            self.SVRG_anchor_gradient_indices = [
                self.SVRG_anchor_gradient_indices[ind] for ind in keepers]
            self.SVRG_anchor_sum = [self.SVRG_anchor_sum[ind] for
                                  ind in keepers]
            if self.N_j is not None:
                self.N_j = np.array([self.N_j[e] for e in keepers])
#            print("the trimmer does:", [self.SVRG_anchor_gradients[ind] for
#                                  ind in kept_run_lengths] )
#            print("before trimmer applied:", self.SVRG_anchor_gradients)
#            print("kept rls:", kept_run_lengths)
#            print("length of SVRG_anchor grads", len(self.SVRG_anchor_gradients))
#            newList = []
#            listInd = 0
#            for ind in kept_run_lengths:
#                #baselevel = self.SVRG_anchor_gradients[ind]
#                newList.append([entry for entry in self.SVRG_anchor_gradients[ind]])
#                #self.SVRG_anchor_gradients[ind] =
#                listInd = listInd + 1
#            self.SVRG_anchor_gradients = newList
#            print("length anchor gradients:", len(self.SVRG_anchor_gradients))
#            print("maximum kept_rl + 1", np.max(kept_run_lengths) + 1)
#            print("kept rl", kept_run_lengths)
#            print("before trim", self.SVRG_anchor_gradients)
            self.SVRG_anchor_gradients = [self.SVRG_anchor_gradients[ind] for
                                  ind in keepers]
            #print("after trim", self.SVRG_anchor_gradients)

        """Once we have put something in them, prune the predictive quantities
        too"""
        if self.predictive_variance_log_det is not None:
            self.predictive_variance_log_det = self.predictive_variance_log_det[
                    kept_run_lengths]
            self.C_t_inv = self.C_t_inv[kept_run_lengths,:,:]

        """Note: If we do alpha param opt, also need to do this for
        the corresponding quantities"""
        if self.alpha_param_learning and self.a_rt_p_eps is not None:
            self.beta_rt_p_eps = self.beta_rt_p_eps[kept_run_lengths,:]
            self.L_rt_p_eps = self.L_rt_p_eps[kept_run_lengths,:,:]
            self.b_rt_p_eps = self.b_rt_p_eps[kept_run_lengths]
            self.a_rt_p_eps = self.a_rt_p_eps[kept_run_lengths]
            self.beta_rt_m_eps = self.beta_rt_m_eps[kept_run_lengths,:]
            self.L_rt_m_eps = self.L_rt_m_eps[kept_run_lengths,:,:]
            self.b_rt_m_eps = self.b_rt_m_eps[kept_run_lengths]
            self.a_rt_m_eps = self.a_rt_m_eps[kept_run_lengths]


        self.retained_run_lengths = (
                    self.retained_run_lengths[kept_run_lengths])
        self.model_log_evidence = scipy.misc.logsumexp(
                        self.joint_log_probabilities )


        """Check if you need to retain all data quantities by checking the
        largest run length index r* and then discarding all obs/data before the
        relevant CP location at t-r*. The way we do this here is to check if
        the largest run length is close to exceeding VB_window. If so, we
        extend the VB_window_size by another 100 entries"""
        self.max_rl = np.max(self.retained_run_lengths)
        rl = int(self.max_rl/100)
        wnd = int((self.VB_window_size-1)/100)
        if (self.dynamic_VB_window_size and
            wnd > rl and
            self.max_rl >= 100):
            #new_len = 100 * (rl + 1)
            self.XX_t = self.XX_t[:(self.max_rl),:,:].copy()
            self.XY_t = self.XY_t[:(self.max_rl),:].copy()
            self.YY_t = self.YY_t[:(self.max_rl)].copy()
            #self.LBFGSB_anchor = self.LBFGSB_anchor[:(self.max_rl),:].copy()
            self.VB_window_size = self.max_rl


    """HYPERPAR OPT"""

    """Simply returns a[r] and b[r] for all run-lengths retained at t plus
    the prior values.
    basically, all we need is a function inside BVARNIG returning a_vec, b_vec
    and then putting both as inputs into the three hyperparopt fcts. We can
    then compute a_vec, b_vec in BVARNIG, and simply set them = a_rt, b_rt in
    BVARNIGDPD """
    def get_param_rt(self):
        a_rt = np.insert(self.a_rt, 0, self.a)
        b_rt = np.insert(self.b_rt, 0, self.b)
        return (a_rt, b_rt)


    """OPTIMIZATION FUNCTIONS/GRADIENTS"""


    @staticmethod
    def neg_ELBO_fun(params, *args):
        """Wrapper function returning the negative of ELBO_fun, used inside
        standard optimization methods"""

        updates = BVARNIGDPD.precompute_VB(params, *args)
        """retrieve quantities"""
        K, E3, Rinv, E2_m_ba = updates[0], updates[1], updates[2], updates[3]
        digamma_diff_1, digamma_diff_2 = updates[4], updates[5]
        gamma_ratio_2,downweight1,downweight2=updates[6],updates[7],updates[8]
        """put them into args"""
        args = (args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                K, E3, Rinv, E2_m_ba,
                digamma_diff_1, digamma_diff_2,
                gamma_ratio_2, downweight1, downweight2,
                args[16], args[17], args[18], args[19],
                args[20], args[21], args[22], args[23]
                )

        return (-BVARNIGDPD.ELBO_fun(params, *args))

    @staticmethod
    def ELBO_fun(params, *args):
        """Evaluation of the ELBO, used inside the standard optimization"""

        """STEP 0.1: Additional arguments that we supply"""
        a0, b0 = args[0], args[1]
        beta0, Sig0inv = args[2], args[3]
        alpha, num_obs, p = args[4], args[5], args[6]
        K, E3, E2_m_ba = args[7], args[8], args[10]
        gamma_ratio_2 = args[13]
        L0 = args[19]
        d = args[20]

        """STEP 1: Get things in the right form"""
        an, bn = params[0], params[1]
        betan = params[2:(p+2)]
        vech_Ln = params[(p+2):]
        lower_ind = np.tril_indices(p, 0)
        Ln = np.zeros((p,p))
        Ln[lower_ind] = vech_Ln
        Linv = scipy.linalg.lapack.clapack.dtrtri(Ln, 1)[0]


        """STEP 2: Compute the E-quantities not depending on observations"""

        E1 = (- np.log(abs(np.prod(np.diag(L0))))
              - np.log(abs(np.prod(np.diag(Ln))))
              + 0.5 * (p - np.trace(
                              np.matmul(Sig0inv,
                                        np.matmul(Linv,
                                                  np.transpose(Linv))
                                        )
                            )
                       )
            )

        E2_m_ba = (gamma_ratio_2
                    * (1.0/alpha)
                    * pow(2 * np.pi, -0.5*d*alpha)
                    * abs(np.prod(np.diag(Ln)))
                    )


        #Ltbeta = np.matmul(np.transpose(Ln), betan)
        #E4 = np.inner(np.transpose(Ltbeta), Ltbeta)


        E8 = -(0.5
              * (1.0/bn)
              * an
              * (np.inner(beta0-betan ,
                          np.matmul(Sig0inv,
                                    beta0-betan))
                 + 2*(b0-bn)
                )
             )

        E9 = -(an*np.log(bn) + scipy.special.gammaln(a0)
                - a0*np.log(b0) - scipy.special.gammaln(an) )
        E10 = -(an-a0) * (scipy.special.digamma(an) - np.log(bn))
        E11 = -(  num_obs
                * gamma_ratio_2
                * pow(bn*2*np.pi, -0.5*d*alpha)
                * pow(1 + alpha, -0.5*d - 1)
            )


        """STEP 3: Compute data-dependent quantities"""
        sum_part = 0.0

        if num_obs > 0:
            log_sum = True
            if not log_sum:
                sum_part = (pow(bn,-0.5*d*alpha)*
                    np.sum(E3*
                        np.power(bn/K, an + 0.5*d*alpha)
                    ))
            else:
                sign = np.sign(E3)*np.sign(K)
                sum_, sign_ = scipy.misc.logsumexp(
                        a = np.log(np.abs(E3)) + (an + 0.5*d*alpha)*(np.log(bn) -
                                   np.log(np.abs(K))),
                        b = sign,
                        return_sign=True
                        )
                sum_part = np.exp(sum_ - np.log(bn) * (0.5*d*alpha))

        return (E1 + E8 + E9 + E10 + E11 + E2_m_ba*sum_part)



    @staticmethod
    def get_grad_neg_ELBO_full(params, *args):
        """Updates the quantities needed in the gradient computation for a,b,
        beta,Land then calls the relevant gradient functions to compute the
        gradients. The gradients are then put into a singular vector.

        Note 1: The params are stored in order:
            an          params[0]
            bn          params[1]
            betan       params[2:(p+2)]
            vech_Ln     params[(p+2):]

        Note 2: The *args are stored in order:
            a0                  float >0                            args[0]
            b0                  float >0                            args[1]
            beta0               np.array, dim=p                     args[2]
            Sig0inv             np.array, dim= p x p                args[3]
            alpha               float > 0                           args[4]
            num_obs             int >0                              args[5]
            p                   int > 0                             args[6]
            K                   np.array, dim=p                     args[7]
            E3                  np.array, dim=p                     args[8]
            Rinv                np.array, dim= max_r x p x p        args[9]
            E2_m_ba             float                               args[10]
            digamma_diff_1      float                               args[11]
            digamma_diff_2      float                               args[12]
            gamma_ratio_2       float                               args[13]
            downweight1         float                               args[14]
            downweight2         float                               args[15]
            XX_t                np.array, dim = VB_window x p x p   args[16]
            XY_t                np.array, dim = VB_window x p       args[17]
            YY_t                np.array, dim = VB_window           args[18]
            L0                  np.array, dim = p x p               args[19]
            d                   int                                 args[20]
            prior_weight        float > 0                           args[21]
            specified_indices   np.array, dim = b or B              args[22]
            return_sum          boolean                             args[23]

        Note 3: precompute_VB returns in the following order:
            K, E3, Rinv, E2_m_ba, digamma_diff_1, digamma_diff_2,
            gamma_ratio_2, downweight1, downweight2
        """

        """STEP 1: Precompute quantities and put them in args"""
        updates = BVARNIGDPD.precompute_VB(params, *args)
        """retrieve quantities"""
        K, E3, Rinv, E2_m_ba = updates[0], updates[1], updates[2], updates[3]
        digamma_diff_1, digamma_diff_2 = updates[4], updates[5]
        gamma_ratio_2,downweight1,downweight2=updates[6],updates[7],updates[8]
        """put them into args"""
        args = (args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                K, E3, Rinv, E2_m_ba,
                digamma_diff_1, digamma_diff_2,
                gamma_ratio_2, downweight1, downweight2,
                args[16], args[17], args[18], args[19],
                args[20], args[21], args[22], args[23]
                )


        """STEP 1: Get gradients of each function"""
        a_grad = -BVARNIGDPD.get_grad_a(params,*args)
        b_grad = -BVARNIGDPD.get_grad_b(params, *args)
        beta_grad = -BVARNIGDPD.get_grad_beta(params,*args)
        L_grad = -BVARNIGDPD.get_grad_L(params, *args)

        """STEP 2: Put together all the gradient (individual terms or sums
        depending on the argument of args[22] and return"""
        if args[23] == True: #if args[22] = True, we have computed the sums
            p = args[6]
            full_grad = np.zeros(int(p*(p+1)*0.5) + p + 2)
            full_grad[0], full_grad[1] = a_grad, b_grad
            full_grad[2:(p+2)] = beta_grad
            full_grad[(p+2):] = L_grad
        else:    #if args[22] = False, we have computed individual gradients
            p = args[6]
            num_obs = args[5]
            full_grad = np.zeros((num_obs, int(p*(p+1)*0.5) + p + 2))
            full_grad[:,0] = a_grad
            full_grad[:,1] = b_grad
            full_grad[:,2:(p+2)] = beta_grad
            full_grad[:,(p+2):] = L_grad

        return full_grad


    @staticmethod
    def precompute_VB(params, *args):
        """Precompute everything from the data that you need"""

        """STEP 1.1: Get the args"""
        alpha, num_obs, p = args[4], args[5], args[6]
        K, E3, Rinv, E2_m_ba = args[7], args[8], args[9], args[10]
        digamma_diff_1, digamma_diff_2 = args[11], args[12]
        gamma_ratio_2, downweight1, downweight2 = args[13],args[14],args[15]
        XX_t, XY_t, YY_t = args[16], args[17], args[18]
        d = args[20]
        specified_indices = args[22] #Not used in these precomputations!
        return_sum = args[23] #Used in all get_grads except precomputations!
        if specified_indices is None:
            specified_indices = np.linspace(0, num_obs-1, num_obs, dtype=int)
        elif specified_indices is not None:
            num_obs = np.size(specified_indices)

        """STEP 1.2: Get the params"""
        an, bn = params[0], params[1]
        betan = params[2:(p+2)]
        vech_Ln = params[(p+2):]
        lower_ind = np.tril_indices(p, 0)
        Ln = np.zeros((p,p))
        Ln[lower_ind] = vech_Ln


        """STEP 1: Compute quantities not depending on data"""

        """STEP 1.1: Auxiliary quantities"""
        gamma_ratio_2 = np.exp(
                scipy.special.gammaln(an +
                    0.5 * alpha * d)
                - scipy.special.gammaln(an)
            )
        digamma_diff_1 = (scipy.special.digamma(an + 1) -
                              scipy.special.digamma(an))
        digamma_diff_2 = (scipy.special.digamma(an +
                    0.5* alpha * d ) - scipy.special.digamma(an))

        """STEP 1.2: Composite quantities not depending on data"""
        #RUNTIME OVERFLOW due to pow(b,a) since a becomes huge (absorb in fraction)
        E2_m_ba = -(
                - abs(np.prod(np.diag(Ln)))
                  * (gamma_ratio_2)
                  * pow(2 * np.pi, -0.5*d * alpha)
                  * (1.0/alpha)
            )
        #DEBUG: This is where the runtime overflow occurs! If b is too large
        #       then this quantity becomes 0
        downweight1 = pow(bn, -0.5*d*alpha)
        downweight2 = pow(bn, -0.5*d*alpha-1)

        """STEP 2: Data-dependent quantities"""

        """STEP 2.1: Compute E_3i terms. Use the cholesky decomposition, as
        we will also want to store the R^-1 terms, and cholesky means we
        can compute both determinant and inverse easily"""
        E3 = np.zeros(num_obs)
        Rinv = np.zeros((num_obs, p,p))

        """NOTE: Think about updating cholesky factorization instead. I.e.,
        we could update the L_rt cholesky with S1*S2 rank-1 updates of O(p^2),
        but we typically expect the direct re-computation of O(p^3) to be
        faster"""

#        print("indices",specified_indices)
#        print("indices shape",specified_indices.shape)
#        print("XX_t", XX_t.shape)
        for i in range(0, num_obs): #specified_indices:
            #DEBUG: Check if more efficient when done using rank-1 updates
            dind = specified_indices[i]
            try:
                if p > 1:
                    Li = np.linalg.cholesky(np.matmul(Ln,
                            np.transpose(Ln)) +
                            alpha * XX_t[dind,:,:])
                else:
                    Li = np.sqrt(Ln*Ln + alpha*XX_t[dind,:,:])
            except np.linalg.LinAlgError as e:
                #if 'not positive definite' in str(e):
                #scale = np.mean(np.abs(Ln[lower_ind]))
                #err = np.random.normal(0, 1, p)
                try:
                    if p>1:
                        Li = np.linalg.cholesky(np.matmul(Ln, np.transpose(Ln)) +
                                #10*np.identity(p)
                                np.sum(np.abs(betan))*np.identity(p)
                                + alpha * XX_t[dind,:,:])
                    else:
                        Li = np.sqrt(Ln*Ln +  3 + alpha*XX_t[dind,:,:])
                except np.linalg.LinAlgError as e:
                    Li = np.identity(p)
                #else:
                #    raise
#            Li = np.linalg.cholesky(np.matmul(Ln,
#                    np.transpose(Ln)) +
#                    alpha * XX_t[i,:,:])
            #syntax: the 1 stands for 'lower triangular', since the np cholesky
            #        decomposition returns lower triangular matrices
#            print("i", i)
#            print("spec ind", specified_indices)
#            print(Li)
            if p>1:
                Li_inv =  scipy.linalg.lapack.clapack.dtrtri(Li, 1)[0]
            else:
                Li_inv = 1.0/Li
            Rinv[i,:,:] = np.matmul(np.transpose(Li_inv), Li_inv)
            E3[i] = abs(np.prod(np.diag(Li_inv)))


        """STEP 2.2: Compute the K-terms"""
        K = np.zeros(num_obs)

        """STEP 2.2.1: Compute the terms not depending on the observations"""
        L_x_beta = np.matmul(np.transpose(Ln), betan)
        vec_part_1 = np.matmul(Ln, L_x_beta)
        E4 = np.inner(L_x_beta, L_x_beta)

        """STEP 2.2.2: Compute the terms depending on observations and put
        the into the i-th entry of K"""
        for i in range(0, num_obs): #specified_indices:
            """get the terms E_{5,i}, E_{6,i}, E_{7,i}"""
            dind = specified_indices[i]
            vec = vec_part_1 + alpha * XY_t[dind,:]
            E5to7i = -np.inner(vec, np.matmul(Rinv[i,:,:], vec))
            K[i] = (bn
                    + 0.5 * (alpha * YY_t[dind] + E4 + E5to7i)
                )

        return (K, E3, Rinv, E2_m_ba, digamma_diff_1, digamma_diff_2, gamma_ratio_2,
                downweight1, downweight2)


    @staticmethod
    def get_grad_a(params, *args):
       #a0, b0, beta0,  Sig0inv, #Sig0inv = Sigma 0 inverse
       #alpha, num_obs, p,
       #K, E3, Rinv, E2_m_ba, digamma_diff_1, digamma_diff_2, gamma_ratio_2,
       #    downweight1, downweight2
       #            ):
        # an, bn, betan, vech_Ln, p, num_obs, alpha,  a0, b0, beta0, vechL0,
        #                         K, E3, Rinv
        """Gradient of ELBO w.r.t. IG-a based on the last r observations
        I.e., for r = 0, we would simply have the prior. For r = 3, we would
        compute the full gradient based on X_t, X_t-1, X_t-2.
        NOTE: Using notation of the appendix for the NIPS submission,
        see details for the expressions (and how to arrive there) there
        """

        """STEP 0: Get things in right form"""

        """STEP 0.1: Additional arguments that we supply"""
        a0, b0 = args[0], args[1]
        beta0, Sig0inv = args[2], args[3]
        alpha, num_obs, p = args[4], args[5], args[6]
        K, E3, E2_m_ba = args[7], args[8], args[10]
        digamma_diff_1, digamma_diff_2 = args[11], args[12]
        gamma_ratio_2, downweight1= args[13],args[14]
        d = args[20]
        prior_weight = args[21]
        specified_indices = args[22] #Only used in precomputations!
        return_sum = args[23] #Used in all but precomputations!
        if specified_indices is not None:
            num_obs = np.size(specified_indices)
        if specified_indices is None:
            specified_indices = np.linspace(0, num_obs-1, num_obs, dtype=int)

        """STEP 0.2: Main arguments that we optimize over"""
        an, bn = params[0], params[1]
        betan = params[2:(p+2)]
        vech_Ln = params[(p+2):]
        lower_ind = np.tril_indices(p, 0)
        Ln = np.zeros((p,p))
        Ln[lower_ind] = vech_Ln

        """STEP 1: Compute the prior-dependent (data-independent) quantities
        Note that the derivatives of E_9 and E_10 cancel except for the
        trigamma term. So we give E10_der with only said trigamma term."""
        #RUNTIME OVERFLOW due to E2 (which is in E2der) and self.K [b^a]
        E2der_m_ba = E2_m_ba*(digamma_diff_2 + np.log(bn))
        E8der =  (-0.5 * (np.inner(beta0 - betan,
                            np.matmul(Sig0inv,
                             beta0 - betan
                            )
                        )
                        + 2*(b0-bn)
                        )
                 * (1.0/bn)
                 *  an
                 *  digamma_diff_1)
        E9der = -np.log(bn) + scipy.special.digamma(an)
        E10der = ( np.log(bn) - scipy.special.digamma(an)
                    -(an - a0)
                      * scipy.special.polygamma(1, an))
        E11der = -(
                    (num_obs * gamma_ratio_2 * digamma_diff_2 )
                    * pow(bn * 2.0 * np.pi,
                            -0.5* d  * alpha)
                    * pow(1.0 + alpha,
                              -0.5 * d - 1)
                )

        """STEP 2: Compute the data-dependent quantitites
        Note: The heavy lifting for this is done in the pre-computation"""
        #RUNTIME OVERFLOW due to E2 (which is in E2der) and self.K [b^a]
        if num_obs > 0:

            log_sum = True

            #modify K s.t. we take it to be > 0 (for log)

            if return_sum:
                if not log_sum:
                    sum_1 = (E2der_m_ba
                             * downweight1
                             * np.sum(E3
                                * np.power(bn/K,
                                    an + 0.5*d*alpha)
                             )
                        )

                    Klarger0 = np.where(K > 0)
                    Ksmaller0 = np.where(K <= 0)
                    exact_der_part = np.sum(E3[Klarger0] *
                                np.power(bn/K[Klarger0], #np.maximum(K, pow(10,-5)),
                                         an + 0.5*d*alpha)
                                #DEBUG: Force K > 0
                                *  np.log(K[Klarger0]) #np.maximum(K, pow(10, -5)))
                            )

                    eps = pow(10,-5)
                    numerical_der_part = (np.sum(E3[Ksmaller0] *
                                (
                                np.power(bn/K[Ksmaller0], an + eps + 0.5*d*alpha)
                                -np.power(bn/K[Ksmaller0], an - eps + 0.5*d*alpha)
                                )
                            )
                            *(1.0/(2*eps))
                        )
                    full_sum = E2_m_ba * (-1.0) * downweight1 * (
                            exact_der_part + numerical_der_part)

                    sum_2 = (
                            E2_m_ba * (-1.0)
                            * downweight1
                            * np.sum(E3 *
                                np.power(bn/K, #np.maximum(K, pow(10,-5)),
                                         an + 0.5*d*alpha)
                                #DEBUG: Force K > 0
                                *  np.log(K) #np.maximum(K, pow(10, -5)))
                            )
                        )


                else:
                    signs = np.sign(E3) * np.sign(K)
                    sum_1_log, sign =   ( #E2der_m_ba * (
                        scipy.misc.logsumexp(
                            a = (
                                np.log(np.abs(E3))
                                #(self.a_rt[r] + 0.5*self.num_regressors*alpha)*
                                #    np.abs(np.log(b)),
                                + (np.log(bn) - np.log(np.abs(K))) *
                                    (an + 0.5*d*alpha)
                               # - np.log(bn) * an
                            ),
                            axis=0,
                            b = signs,
                            return_sign = True
                            )
                    )
                    sum_1 = (sign * np.exp(sum_1_log - np.log(bn) *
                                           0.5*d*alpha) * E2der_m_ba)


                    Klarger0 = np.where(K > 0)
                    Ksmaller0 = np.where(K <= 0)
                    """For all K>0, get the derivative exactly in log form for
                    numerical stability."""

                    if np.size(Klarger0)>0:
                        sign_ = np.sign(np.log(K[Klarger0]))*np.sign(E3[Klarger0])
                        exact_der_part_log, sign_edpl = scipy.misc.logsumexp(
                                a = ( np.log(np.abs(E3[Klarger0]))
                                      + (an + 0.5*d*alpha)*
                                         (np.log(bn) - np.log(K[Klarger0]))
                                      + np.log(np.abs(np.log(K[Klarger0])))),
                                b = sign_,
                                return_sign = True,
                                axis=0
                            )
                    else:
                        exact_der_part_log = -np.inf
                        sign_edpl = 1
                    """Get the numerical difference to approx. derivative
                    numerically for all K<0. Do this in log format in order to be
                    numerically stable for large values of an. I.e., you get the
                    log of the numerical derivative of (bn/K)^(an + 0.5d*alpha).
                    To this end, first compute the log of
                        (bn/K)^([an+ eps] + 0.5d*alpha) -
                            (bn/K)^([an - eps] + 0.5d*alpha )."""
                    if np.size(Ksmaller0):
                        eps = pow(10,-5)
                        diff_log, diff_log_sign = scipy.misc.logsumexp(
                                a = np.array([
                                        (an + eps + 0.5*d*alpha) * (
                                            np.log(bn) - np.log(np.abs(K[Ksmaller0]))),
                                        (an - eps + 0.5*d*alpha) * (
                                            np.log(bn) - np.log(np.abs(K[Ksmaller0])))
                                        ]),
                                b = np.array([
                                            np.sign(K[Ksmaller0]),
                                            (-1) * np.sign(K[Ksmaller0])
                                        ]),
                                return_sign = True,
                                axis=0
                            )
                        """Using the above, get the sum of the numerical gradient
                        parts for which K was smaller than 0."""
                        sig = diff_log_sign * np.sign(E3[Ksmaller0])
                        numerical_der_part_log, sign_ndpl = scipy.misc.logsumexp(
                                a = (np.log(np.abs(E3[Ksmaller0])) + diff_log +
                                     -np.log((2*eps))),
                                b = sig,
                                return_sign = True,
                                axis=0
                            )
                    else:
                        numerical_der_part_log = -np.inf
                        sign_ndpl = 1

                    """Finally, get the full sum of derivatives for both the K>0
                    and K<0 terms"""
                    sum_part, sum_sgn = scipy.misc.logsumexp(
                            a = np.array([exact_der_part_log,
                                         numerical_der_part_log]),
                            b = np.array([sign_edpl, sign_ndpl]),
                            return_sign = True,
                            axis=0)

                    sum_2 = E2_m_ba * (-1.0) * sum_sgn * np.exp(sum_part -
                                    (0.5*d*alpha)*np.log(bn))
            if not return_sum:
                """If we do not return the sum, we return the individual
                gradients instead"""
                #log_form = False #we return the gradients in their 'true form'
                                #rather than in log form
                #max_1 = np.log(np.finfo("float").max)
                #too_larg = 0.5*d*alpha*(np.log(bn) - np.log(K)) +

#                print("E2dermba", E2der_m_ba)
#                print("E3", E3)
#                print("bn/K", bn/K)
#                print("an", an)
#                print("downweight", downweight1)

                sum_part_1 = (E2der_m_ba
                             * downweight1
                             * (E3 * np.power(bn/K, an + 0.5*d*alpha))
                        )

                Klarger0 = np.where(K > 0)
                Ksmaller0 = np.where(K <= 0)
                exact_der_part = (E3[Klarger0] * np.power(bn/K[Klarger0],
                                     an + 0.5*d*alpha)
                            *  np.log(K[Klarger0])
                        )

                eps = pow(10,-5)
                numerical_der_part = (E3[Ksmaller0] *
                            (
                            np.power(bn/K[Ksmaller0], an + eps + 0.5*d*alpha)
                            -np.power(bn/K[Ksmaller0], an - eps + 0.5*d*alpha)
                            )
                        *(1.0/(2*eps))
                    )

                both_parts = np.zeros(np.size(K))
                if np.size(Klarger0) > 0:
                    both_parts[Klarger0] = exact_der_part
                if np.size(Ksmaller0)>0:
                    both_parts[Ksmaller0] = numerical_der_part
                sum_part_2 = E2_m_ba * (-1.0) * downweight1 * both_parts

                individual_grads_a = (
                        (sum_part_1 + sum_part_2) +
                        prior_weight*(E8der + E9der + E10der + E11der)
                    )

#                #make sure these are correct by comparing their sum to elbo der a
#                print("a grad with individual grads",
#                      np.sum(individual_grads_a))
#                print("a grad with logs",
#                      (prior_weight*(E8der + E9der + E10der + E11der)
#                        + sum_1 + sum_2))

        elif num_obs <= 0:
            """else (if num_obs  0)"""
            sum_1, sum_2 = 0.0, 0.0
            individual_grads_a = prior_weight*(E8der + E9der + E10der + E11der)
        #DEBUG: Presupposes that self.K is positive in every entry!
        #       not sure this holds, need some precaution!
        #DEBUG: Give a warning whenever some entries of K are negative!
        #if np.sum(K < 0 ) > 0 and num_obs > 0:
        #    print("number of negative K entries: ", np.sum(self.K < 0 ))

        """STEP 3: Sum up everything and return it"""
        #We only call this version of the gradient within the global opt
        #function, so we set prior_weight = 1
        if return_sum:
            ELBOder_a = (prior_weight*(E8der + E9der + E10der + E11der)
                        + sum_1 + sum_2)
            return ELBOder_a #  ELBOder_a #E8der +E9der + E10der + E11der #ELBOder_a
        else:
            return individual_grads_a


    @staticmethod
    def get_grad_b(params, *args):
       #a0, b0, beta0,  Sig0inv, #Sig0inv = Sigma 0 inverse
       #alpha, num_obs, p,
       #K, E3, Rinv, E2_m_ba, digamma_diff_1, digamma_diff_2, gamma_ratio_2,
       #    downweight1, downweight2):
        """Gradient of ELBO w.r.t. IG-a based on the last r observations
        I.e., for r = 0, we would simply have the prior. For r = 3, we would
        compute the full gradient based on X_t, X_t-1, X_t-2.
        NOTE: Using notation of the appendix for the NIPS submission,
        see details for the expressions (and how to arrive there) there
        """

        """STEP 0.1: Additional arguments that we supply"""
        a0, b0 = args[0], args[1]
        beta0, Sig0inv = args[2], args[3]
        alpha, num_obs, p = args[4], args[5], args[6]
        K, E3, E2_m_ba = args[7], args[8], args[10]
        gamma_ratio_2, downweight1, downweight2 = args[13],args[14],args[15]
        d = args[20]
        prior_weight = args[21]
        specified_indices = args[22] #Only used in precomputations!
        return_sum = args[23] #Used in all but precomputations!
        if specified_indices is not None:
            num_obs = np.size(specified_indices)
        if specified_indices is None:
            specified_indices = np.linspace(0, num_obs-1, num_obs, dtype=int)


        """STEP 0.2: Get things in right form"""
        an, bn = params[0], params[1]
        betan = params[2:(p+2)]
        vech_Ln = params[(p+2):]
        lower_ind = np.tril_indices(p, 0)
        Ln = np.zeros((p,p))
        Ln[lower_ind] = vech_Ln

        """STEP 1: Compute the prior-dependent (data-independent) quantities,
        i.e. the derivatives of E_2, E_8,E_9,E_10,E_11. Note that the sum of
        the derivatives of E_9 and E_10 is -a0/bn, so we only give E10der"""
        E2der_m_ba = E2_m_ba * (an/bn)
        E8der = ( 0.5  * (np.inner(beta0-betan,
                                    np.matmul(Sig0inv,
                                      beta0-betan
                                    )
                                )
                            #- 2*self.b
                          )
                        * (1.0/pow(bn,2))
                        * an
                 )
        E8der = E8der + b0 * (1.0/pow(bn,2)) * an
        #print("additional part:", (1.0/pow(self.b_rt[r],2)) * self.gamma_ratio_1)
        E10der = -(a0/bn)
        E9der = -(an/bn)
        E10der = (an-a0)/bn
        E10der = E9der + E10der
        E11der =  (
                (
                    (num_obs * 0.5 * d * alpha)
                    * gamma_ratio_2
                )
                *
                (
                    pow(2 * np.pi, -0.5 * d * alpha)
                    * pow(1 + alpha, -0.5 * d - 1)
                    * downweight2
                )
            )


        """STEP 2: Compute the data-dependent quantitites
        Note: The heavy lifting for this is done in the pre-computation"""
        #RUNTIME OVERFLOW due to E2 (which is in E2der) and self.K [b^a]
        if num_obs > 0:

            if return_sum:
                log_sum = True
                if not log_sum:

                    sum_1 = (
                        E2der_m_ba
                            * downweight1
                            * np.sum(
                                E3
                                * np.power(bn/K,
                                       an + 0.5*d*alpha)
                            )
                        )

                    sum_2 = (
                        E2_m_ba
                        * (-an - 0.5*d*alpha)
                        * downweight2
                        * np.sum(E3
                            * np.power(bn/K,
                                an + 0.5*d*alpha + 1)
                          )
                        )
                else:

                    sig = (np.sign(E3) * np.sign(K) )#np.maximum(K, pow(10,-5)))))
                    sum_1_log, sign_ = scipy.misc.logsumexp(
                            a = (
                                np.log(np.abs(E3))
                                + (np.log(bn) - np.log(np.abs(K)))#np.maximum(K, pow(10,-5)))))
                                    * (an + 0.5*d*alpha)
                                #- np.log(bn) * an
                                #+ np.log(np.abs(np.log(K))) #np.maximum(K, pow(10, -5)))
                            ),
                            b = sig,
                            axis=0,
                            return_sign=True
                        )
                    sum_1  = (sign_ * np.exp(sum_1_log - np.log(bn) *
                                                0.5*d*alpha) * E2der_m_ba)



                #RUNTIME OVERFLOW due to E2 and self.K [b^a]
                    sign = np.sign(E3) * np.sign(K)
                    sum_2_log, sign_ = scipy.misc.logsumexp(
                            a = (np.log(np.abs(E3)) + (an + 0.5*alpha+1)*(
                                    np.log(bn) - np.log(np.abs(K)))),
                            b = sign,
                            axis=0,
                            return_sign=True
                        )
                    sum_2 = (sign_ * np.exp(sum_2_log - np.log(bn) *
                                (0.5*d*alpha+1))* (-an - 0.5*d*alpha) * E2_m_ba )

            if not return_sum:
                """If we want to return the individual gradients, do the
                relevant computations here"""
                sum_1_part = ( (E2der_m_ba * downweight1) * (
                                E3 * np.power(bn/K, an + 0.5*d*alpha) )
                        )
                sum_2_part = ( (E2_m_ba * (-an - 0.5*d*alpha) * downweight2)
                        * (E3 * np.power(bn/K, an + 0.5*d*alpha + 1) ))

                individual_grads_b = (prior_weight*(E8der + E10der + E11der)
                    + sum_1_part + sum_2_part)

#                print("ind grads b summed", np.sum(individual_grads_b))
#                print("ELBO der b", prior_weight*(E8der + E10der + E11der)
#                            + sum_1 + sum_2)

        elif num_obs <= 0:
            sum_1, sum_2 = 0.0, 0.0
            individual_grads_b = prior_weight*(E8der + E10der + E11der)

        """STEP 3: Sum up everything and return it"""
        #prior weight = 1 because we only use this gradient function in full opt.
        if return_sum:
            ELBOder_b = prior_weight*(E8der + E10der + E11der) + sum_1 + sum_2
            return ELBOder_b #ELBOder_b
        elif not return_sum:
            return individual_grads_b


    @staticmethod
    def get_grad_beta( params, *args):
    #   a0, b0, beta0, Sig0inv, #Sig0inv = Sigma 0 inverse
    #   alpha, num_obs, p,
    #   K, E3, Rinv, E2_m_ba, digamma_diff_1, digamma_diff_2, gamma_ratio_2,
    #       downweight1, downweight2, XX_t, XY_t, YY_t):
        """Gradient of ELBO w.r.t. beta based on the last r observations
        I.e., for r = 0, we would simply have the prior. For r = 3, we would
        compute the full gradient based on X_t, X_t-1, X_t-2.
        NOTE: Using notation of the appendix for the NIPS submission,
        see details for the expressions (and how to arrive there) there
        """

        """STEP 0.1: Additional arguments that we supply"""
        beta0, Sig0inv = args[2], args[3]
        alpha, num_obs, p = args[4], args[5], args[6]
        K, E3, Rinv, E2_m_ba = args[7], args[8], args[9], args[10]
        downweight2 = args[15]
        XY_t =  args[17]
        d = args[20]
        prior_weight = args[21]
        specified_indices = args[22] #Only used in precomputations!
        return_sum = args[23] #Used in all but precomputations!
        if specified_indices is not None:
            num_obs = np.size(specified_indices)
        if specified_indices is None:
            specified_indices = np.linspace(0, num_obs-1, num_obs, dtype=int)

        """STEP 0.2: Get vech_Ln into Ln form"""
        an, bn = params[0], params[1]
        betan = params[2:(p+2)]
        vech_Ln = params[(p+2):]
        lower_ind = np.tril_indices(p, 0)
        Ln = np.zeros((p,p))
        Ln[lower_ind] = vech_Ln

        """STEP 1: Compute the prior-dependent (data-independent) quantities,
        i.e. the derivatives of E_4, E_8"""
        E4der =( 2 * np.matmul(
                        np.matmul(
                               np.transpose(betan),
                               Ln
                              ),
                        np.transpose(Ln)
                    )
                )
        E8der = (
                    (1.0/bn) * an
                     * np.matmul(
                        np.transpose(beta0 - betan),
                        Sig0inv,
                    )
                 )
        """STEP 2: Compute the data-dependent quantitites
        Note: The heavy lifting for this is done in the pre-computation"""

        """STEP 2.1: Get the quantities that are functionals of the precomputed
        quantities R, the derivatives of E_{5,i} and E_{7,i}"""

        if num_obs > 0:

            #DEBUG: INEFFICIENT: Since both XY and R in
            #       an array, s.t. I can do matrix mult along the first (0) axis?
            E7der = np.zeros((num_obs,p))
            E5der = np.zeros((num_obs, p))
            for i in range(0, num_obs):
                dind = specified_indices[i]
                E5der[i,:] = np.matmul(E4der, Rinv[i,:,:])
                E5der[i,:] = np.matmul(E5der[i,:], (Ln))
                E5der[i,:] = -np.matmul(E5der[i,:],
                                     np.transpose(Ln))
                #INEFFICIENT: R^-1 * Sigma^-1 mult. twice! once for E5der before
                E7der[i,:] = np.matmul(np.transpose(XY_t[dind,:]),
                                  Rinv[i,:,:])
                E7der[i,:] = np.matmul(E7der[i,:], Ln)
                E7der[i,:] = np.matmul(E7der[i,:], np.transpose(Ln))

            E7der = -2 * alpha * E7der

            """STEP 2.2: Get the summed quantities that are themselves functionals
            of the previously derived E5der, E7der entries."""
            #RUNTIME OVERFLOW due to E2  and self.K [b^a]

            if return_sum:
                """Return the sum (rather than individual grads)"""
                log_sum = True
                if not log_sum:
                    sum_1 = (
                                (-an -0.5*d*alpha)
                                * 0.5
                                * downweight2
                                * E2_m_ba * (
                                np.sum(
                                    E3[:,np.newaxis]
                                    * (E4der + E5der +E7der)
                                    * np.power(bn/K,
                                        an + 0.5*d*alpha + 1)[:,np.newaxis]
                                    , axis=0
                                )
                            )
                        )
                else:
                    sign = (np.sign(E3)[:,np.newaxis]  *
                            np.sign((E4der + E5der +E7der)) *
                            np.sign(K)[:,np.newaxis] )
                    #DEBUG: Somehow, an error occurs here 'divide by zero in log'
                    #       which means we get log(0) somewhere. Internally that is
                    #       solved by assigning -np.inf, so there is no issue here
                    sum_1_log, sign_ = scipy.misc.logsumexp(
                            a = (np.log(np.abs(E3))[:,np.newaxis]
                                 + np.log(np.abs(E4der + E5der +E7der))
                                 + (an + 0.5*alpha+1)*(
                                    np.log(bn) - np.log(np.abs(K)))[:,np.newaxis]
                                 ),
                            b = sign,
                            axis=0,
                            return_sign=True
                        )
                    sum_1 = (sign_ * np.exp(sum_1_log - np.log(bn) * (0.5*d*alpha+1))
                            * (-an - 0.5*d*alpha) * 0.5 * E2_m_ba )
            if not return_sum:
                """Return individual grads (rather than their sum)"""
                sum_1_part = ((-an -0.5*d*alpha)
                                * 0.5
                                * downweight2
                                * E2_m_ba * (
                                    E3[:,np.newaxis]
                                    * (E4der + E5der +E7der)
                                    * np.power(bn/K,
                                        an + 0.5*d*alpha + 1)[:,np.newaxis]
                                )
                            )
                individual_grads_beta = prior_weight*E8der + sum_1_part
#                print("individual grads sum", np.sum(individual_grads_beta, axis=0))
#                print("ELBOder beta", prior_weight*E8der + sum_1)

        elif num_obs <= 0:
            sum_1 = 0.0
            individual_grads_beta = prior_weight * E8der

        """STEP 3: Sum up everything and return it"""
        if return_sum:
            ELBOder_beta = ELBOder_beta = prior_weight*E8der + sum_1
            return ELBOder_beta
        elif not return_sum:
            return individual_grads_beta #ELBOder_beta #E7der[-1,:] #ELBOder_beta #  E5der[0,:] #(ELBOder_beta).reshape(self.num_regressors)


    @staticmethod
    def get_grad_L( params, *args):
       #a0, b0, beta0, Sig0inv, #Sig0inv = Sigma 0 inverse
       #alpha, num_obs, p,
       #K, E3, Rinv, E2_m_ba, digamma_diff_1, digamma_diff_2, gamma_ratio_2,
       #    downweight1, downweight2, XX_t, XY_t, YY_t):

        """STEP 0.1: Additional arguments that we supply"""
        Sig0inv = args[3]
        alpha, num_obs, p = args[4], args[5], args[6]
        K, E3, Rinv, E2_m_ba = args[7], args[8], args[9], args[10]
        downweight1, downweight2 = args[14],args[15]
        XY_t =  args[17]
        d=args[20]
        prior_weight = args[21]
        specified_indices = args[22] #Only used in precomputations!
        return_sum = args[23] #Used in all but precomputations!

        if specified_indices is not None:
            num_obs = np.size(specified_indices)
        if specified_indices is None:
            specified_indices = np.linspace(0, num_obs-1, num_obs, dtype=int)

        """STEP 0.2: Get params & vech_Ln into Ln form"""
        an, bn = params[0], params[1]
        betan = params[2:(p+2)]
        vech_Ln = params[(p+2):]
        lower_ind = np.tril_indices(p, 0)
        Ln = np.zeros((p,p))
        Ln[lower_ind] = vech_Ln

        """STEP 1: Compute the prior-dependent (data-independent) quantities,
        i.e. the derivatives of E1, E2, E_4"""
        B = np.outer(betan, betan)
        L_inv = scipy.linalg.lapack.clapack.dtrtri(Ln, 1)[0]
        #det = np.exp(2 * np.trace(self.L_rt[r,:,:]))
        #Note: Of the extracted indices, only the diagonal is nonzero
        E1der = (np.matmul(
                         (np.matmul(
                                np.matmul(np.transpose(L_inv), L_inv),
                                Sig0inv
                            )
                         -  np.identity(p)),
                            np.transpose(L_inv)
                        )[lower_ind]     )
        E2der_m_ba = E2_m_ba * np.transpose(L_inv)[lower_ind]
        E4der = 2.0 * np.matmul(B, Ln)[lower_ind]

        """STEP 2: Compute the data-dependent quantitites.
        For efficiency, compute the entire derivative term inside the sum
        observation by observation. I.e., we directly compute the two sums
        depending on derivatives of E_{3,i}, E_4, E_{5,i}, E_{6,i}, E_{7,i}"""

        #sum_der_terms = np.zeros((r, self.num_regressors, self.num_regressors))
        if num_obs > 0:
            entries = int(p*(p+1)*0.5)
            #sum_ders = np.zeros(entries)
            #sum_ders_log = -np.inf * np.ones(entries)
            #sum_sign = np.ones(entries)
            E3der = np.zeros((num_obs, entries))
            E4to7der = np.zeros((num_obs, entries))
            for i in range(0,num_obs):
                dind = specified_indices[i]
                """STEP 2.1: Compute auxiliary quantities needed for derivatives of
                E_{3,i}, E_4, E_{5,i}, E_{6,i}, E_{7,i}, namely A1 - A6"""
                A1 = np.matmul(Rinv[i,:,:], Ln)
                A2 = np.matmul(A1, np.transpose(Ln))
                A3 = np.matmul(A2, B)
                A4 = np.matmul(Rinv[i,:,:], XY_t[dind,:]) #vector!
                A5 = np.matmul(np.transpose(XY_t[dind,:]), A1)  #vector!
                A6 = np.matmul(np.identity(p) - A2,
                               betan)  #vector!

                """STEP 2.2: Compute  E_{3,i}, E_4, E_{5,i}, E_{6,i}, E_{7,i} for
                the current value of i"""
                E3ider = -E3[i] * A1[lower_ind]
                E5ider = (2 *
                            np.matmul(
                                (
                                  np.matmul(A2 - np.identity(p),
                                    np.transpose(A3))
                                  - A3
                                ),
                                  Ln
                            )[lower_ind]
                        )
                E6ider = 2 * pow(alpha, 2) * np.outer(A4, A5)[lower_ind]
                E7ider = -2 * alpha * (
                            np.outer(A6, A5)
                            + np.matmul(
                                np.outer(A4, A6),
                                Ln
                            )
                        )[lower_ind]

                """STEP 2.3: Put into the correct position"""
                E3der[i,:] = E3ider
                E4to7der[i,:] = (E4der + E5ider + E6ider + E7ider)

                """STEP 2.4: add the next term to the sum derivative"""
                #NEED LOG TRANSFORM!
#                sum_ders = sum_ders + (
#                            E3ider
#                            * pow(bn/K[i],
#                                an + 0.5*d*alpha)
#                            * downweight1
#                            )
                #NEED LOG TRANSFORM!
#                sum_ders = sum_ders + (
#                            E3[i]
#                            * (-an - 0.5*d*alpha) * 0.5
#                            * (E4der + E5ider + E6ider + E7ider)
#                            * pow(bn/K[i],
#                                an + 0.5*d*alpha + 1)
#                            * downweight2
#                        )
            #sum_ders = sum_ders * E2_m_ba
            #NEED LOG TRANSFORM!
#            sum_ders = sum_ders + E2der_m_ba * np.sum(
#                            E3
#                            * np.power(bn/K,
#                                an + 0.5 * d * alpha)
#                            * downweight1
#                        )

            if return_sum:
                """If we want the gradient to return the sum"""

                """STEP 3: Log-transformed summation"""
                sum_1_signs = np.sign(E3der) * np.sign(K)[:,np.newaxis]
                #needs to be multiplied by E2_m_ba later
                sum_1, sum_1_sign = scipy.misc.logsumexp(
                        a = (np.log(np.abs(E3der))
                            + (np.log(bn) - np.log(np.abs(K)))[:,np.newaxis]*(an + 0.5*d*alpha)
                            - np.log(bn) * (0.5*d*alpha)),
                        b = sum_1_signs,
                        return_sign = True,
                        axis=0
                    )

                sum_2_signs = np.sign(E4to7der) * np.sign(K)[:,np.newaxis] * (-1)
                #needs to be multiplied by E2_m_ba later
                sum_2, sum_2_sign = scipy.misc.logsumexp(
                        a = (np.log(E3)[:,np.newaxis]
                            + np.log(0.5)
                            + np.log(np.abs(E4to7der))
                            + np.log(an+0.5*d*alpha)
                            + (np.log(bn) - np.log(np.abs(K)))[:,np.newaxis] * (an + 0.5*d*alpha + 1)
                            - np.log(bn) * (0.5*d*alpha + 1)
                        ),
                        b = sum_2_signs,
                        return_sign = True,
                        axis=0
                    )

                sum_3_signs = np.sign(K)
                #needs to be multiplied by E2der_m_ba later
                sum_3, sum_3_sign = scipy.misc.logsumexp(
                        a = (np.log(E3)
                            + (np.log(bn) - np.log(np.abs(K))) * (an + 0.5 * d * alpha)
                            - np.log(bn) * (0.5 * d * alpha)),
                        b = sum_3_signs,
                        return_sign= True,
                        axis=0
                    )

                sum_sum, sum_sum_sign = scipy.misc.logsumexp(
                        a=np.array([sum_1, sum_2]),
                        b=np.array([sum_1_sign, sum_2_sign]),
                        return_sign = True,
                        axis=0
                    )


                full_sum = (E2_m_ba * np.exp(sum_sum) * sum_sum_sign +
                            E2der_m_ba * np.exp(sum_3) * sum_3_sign)
                sum_ders = full_sum

                """Get the ELBO derivatie"""
                ELBOder_L = prior_weight*E1der + sum_ders

            if not return_sum:
                """If we want the individual gradient components"""
                sum_part_1 = (E2der_m_ba * ( pow(bn, -0.5*d*alpha)
                    *(E3 * np.power(bn/K, an + 0.5*d*alpha)))[:,np.newaxis])
                sum_part_2 = ( (E2_m_ba * pow(bn, -0.5*d*alpha))
                    * (E3der * np.power(bn/K, an + 0.5*d*alpha)[:,np.newaxis]))
                sum_part_3 = (
                    (E2_m_ba*pow(bn, -0.5*d*alpha-1)*(-an - 0.5*d*alpha)*0.5)
                    * (E3 * np.power(bn/K, an + 0.5*d*alpha + 1))[:,np.newaxis]
                    * E4to7der)

                individual_grads_L = (sum_part_1 + sum_part_2 + sum_part_3 +
                                      prior_weight*E1der )


#            #Test if full_sum = sum_ders
#            if np.any(full_sum != sum_ders):
##                print("full sum:", full_sum)
##                print("sum ders", sum_ders)
##                print("full sum 2:", full_sum2)
#                #print("sum 1 no logs", sum_1_)
#                print("sum 1 logs", E2_m_ba * np.exp(sum_1) * sum_1_sign)
#                #print("sum 2 no logs", sum_2_)
#                print("sum 2 logs", E2_m_ba * np.exp(sum_2) * sum_2_sign)
#                #print("sum 3 no logs", sum_3_)
#                print("sum 3 logs", E2der_m_ba * np.exp(sum_3) * sum_3_sign)

            """STEP 3: Add the last remaining term and return everything"""
             #full_sum #sum_ders
        elif num_obs <= 0:
            ELBOder_L = prior_weight*E1der
            individual_grads_L = prior_weight * E1der
        if return_sum:
            return ELBOder_L #ELBOder_L
        elif not return_sum:
            return individual_grads_L



#***************DEPRECATED GRADIENT FUNCTIONS********************
#
#DEPRECATED! OLD MANUAL OPTIMIZATION
#    def take_step_VB(self, r, num_obs, count, alpha = None):
#        """This function (1) calls the gradient evaluations for all VB params,
#        (2) computes the step size from old iterations, and (3) takes the steps
#        to update a, b, beta, L accordingly"""
#
#        """Unless otherwise specified, work with self.alpha_param"""
#        if alpha is None:
#            alpha = self.alpha_param
#
#        """STEP 1: Compute gradients of a_rt, b_rt, beta_rt, L_rt"""
#        gradient_a = self.get_gradient_a(r=r, num_obs=num_obs,  alpha=alpha)
#        gradient_b = self.get_gradient_b(r=r, num_obs=num_obs, alpha=alpha)
#        gradient_beta = self.get_gradient_beta(r=r,num_obs=num_obs,alpha=alpha)
#        gradient_L = self.get_gradient_L(r=r, num_obs=num_obs, alpha=alpha)
#
#        """If true, then we compute the numerical gradients here and compare
#        them to the direct gradients as derived in the appendix"""
#        DEBUG_gradients_SGD = False
#        if DEBUG_gradients_SGD:
#
#            #Build stuff we need to evaluate the gradient numerically
#            num_grads = (self.num_regressors +
#                int(self.num_regressors * (self.num_regressors+1) * 0.5) + 2)
#            turb = pow(10,-3) #beta, L, a, b is order of perturb
#
#            perturbation = np.zeros(num_grads)
#            grad_beta = np.zeros(self.num_regressors)
#            grad_L = np.zeros(int(self.num_regressors *
#                                  (self.num_regressors+1) * 0.5))
#
#            turb = pow(10,-3)
#            perturbation[-1] = turb
#            elbobp = self.get_ELBO( perturbation, r, num_obs)
#            elbobm = self.get_ELBO( -perturbation, r, num_obs)
#            grad_b = (elbobp - elbobm)/(2 * turb)
#            #log_grad_b = (np.log(-elbobp) - np.log(-elbobm))/(2 * turb)
#
#            #a
#            turb = pow(10,-3)
#            perturbation[-1] = 0
#            perturbation[-2] = turb
#            elboap = self.get_ELBO( perturbation, r, num_obs)
#            elboam = self.get_ELBO( -perturbation, r, num_obs)
#            grad_a = (elboap - elboam)/(2 * turb)
#            #log_grad_a = (np.log(-elboap) - np.log(-elboam))/(2 * turb)
#
#            #beta
#            turb = pow(10,-3)
#            perturbation[-2] = 0
#            for i in range(0, self.num_regressors):
#                perturbation[i] = turb
#                elbobetap = self.get_ELBO( perturbation, r, num_obs)
#                elbobetam = self.get_ELBO( -perturbation, r, num_obs)
#                perturbation[i] = 0
#                grad_beta[i] = (elbobetap - elbobetam)/(2 * turb)
#                #log_grad_beta[i] = (np.log(-elbobetap) - np.log(-elbobetam))/(2 * turb)
#
#            #L
#            for i in range(0,int(self.num_regressors * (self.num_regressors+1) * 0.5)):
#                perturbation[i + self.num_regressors] = turb
#                elboLp = self.get_ELBO( perturbation, r, num_obs)
#                elboLm = self.get_ELBO( -perturbation, r, num_obs)
#                perturbation[i+ self.num_regressors] = 0
#                grad_L[i] = (elboLp - elboLm)/(2 * turb)
#                #log_grad_L[i] = (np.log(-elboLp) - np.log(-elboLm))/(2 * turb)
#
#            #print gradients for numerical approx if deviation significant
#            if (abs(gradient_a - grad_a) > 0.25 or abs(gradient_b - grad_b) > 0.25 or
#                np.max(np.abs(gradient_beta - grad_beta))>0.25 or
#                np.max(np.abs(gradient_L - grad_L))>0.25):
#                print("NUMERICAL GRADIENTS ELBO SIGNIFICANTLY DIFFERENT!")
#                print("a diff:", abs(gradient_a - grad_a))
#                print("b diff:", abs(gradient_b - grad_b))
#                print("beta diff:", np.max(np.abs(gradient_beta - grad_beta)))
#                print("L diff:", np.max(np.abs(gradient_L - grad_L)))
#                # print("NUMERICAL GRADIENTS LOG ELBO")
#                #print(log_grad_a)
#                #print(log_grad_b)
#                #print(log_grad_beta)
#                #print(log_grad_L)
#                print("t - lag length = ", self.retained_run_lengths.shape[0] - self.lag_length)
#                print("current values")
#                print("a", self.a_rt[r])
#                print("b", self.b_rt[r])
#                print("beta", self.beta_rt[r,:])
#                print("L", self.L_rt[r,:,:])
#                print("DIRECT GRADIENTS")
#                print("a",gradient_a)
#                print("b",gradient_b)
#                print("beta",gradient_beta)
#                print("L",gradient_L)
#                print("NUMERICAL GRADIENTS ELBO")
#                print("a", grad_a)
#                print("b", grad_b)
#                print("beta", grad_beta)
#                print("L", grad_L)
#
#        """Similar to what we do on top, but we compare numerical gradients to
#        the direct gradients we use inside the python optimization"""
#        DEBUG_gradients_optimizer = False
#        if DEBUG_gradients_optimizer:
#            #compare the static methods with object-internatl gradients
#
#            alpha, p = self.alpha_param, self.num_regressors
#            old_params = np.zeros(int(p*(p+1)*0.5) + p + 2)
#            old_params[0], old_params[1] = self.a_rt[r], self.b_rt[r]
#            old_params[2:(p+2)] = self.beta_rt[r,:]
#            old_params[(p+2):] = self.L_rt[r,:,:][np.tril_indices(p, 0)]
#
#            #priors
#            a0,b0 = self.a, self.b
#            beta0, L0 = self.prior_mean_beta, self.L_0 #don't think we ever need it
#            Sig0inv = self.prior_var_beta_inv
#
#            #precomputed values
#            K, E3, Rinv, E2_m_ba = self.K, self.E3, self.Rinv, self.E2_m_ba
#            digamma_diff_1, digamma_diff_2 = self.digamma_diff_1, self.digamma_diff_2
#            gamma_ratio_2 = self.gamma_ratio_2
#            downweight1, downweight2 = self.downweight1, self.downweight2
#            XX_t, XY_t, YY_t = self.XX_t, self.XY_t, self.YY_t
#            d = self.S1 *  self.S2
#
#            args = (a0, b0, beta0, Sig0inv,
#                alpha, num_obs, p,
#                K, E3, Rinv, E2_m_ba,
#                digamma_diff_1, digamma_diff_2, gamma_ratio_2,
#                downweight1, downweight2,
#                XX_t,XY_t,YY_t,
#                L0, d)
#            #compute grads and the elbo
#            grad2_a = BVARNIGDPD.get_grad_a(old_params,
#                *args)
#            grad2_b = BVARNIGDPD.get_grad_b( old_params,
#                *args)
#            grad2_beta = BVARNIGDPD.get_grad_beta( old_params,
#                *args)
#            grad2_L = BVARNIGDPD.get_grad_L( old_params,
#                *args)
#            elbo = BVARNIGDPD.ELBO_fun(old_params,
#                *args)
#            elbo_obj = self.get_ELBO(perturbation = np.zeros(
#                    int(p*(p+1)*0.5) + p + 2), r=r, num_obs=num_obs)
#
#            #comparison
#            print("STATIC VS OBJECT")
#            print("a diff:", abs(gradient_a - grad2_a))
#            print("b diff:", abs(gradient_b - grad2_b))
#            print("beta diff:", np.max(np.abs(gradient_beta - grad2_beta)))
#            print("L diff:", np.max(np.abs(gradient_L - grad2_L)))
#            print("elbo diff:", abs(elbo-elbo_obj))
#
#
#        """STEP 2: Compute step sizes and increments"""
#
#        """STEP 2.1: Compute step sizes"""
#        if count == 0:
#            """If first step, cannot use knowledge of previous step sizes"""
#            step_size_a = 1.0
#            step_size_b = 1.0
#            step_size_beta = 1.0
#            step_size_L = 1.0
#        else:
#
#            BB_steps= False
#            if BB_steps:
#                """If true, use BB approach, appropriate step size via
#                Barzilai and Borwein (BB), see also
#                ftp://lsec.cc.ac.cn/pub/yyx/papers/p0504.pdf"""
#
#                """STEP 2.1.1: For a"""
#                gradient_a_increment = gradient_a - self.old_gradient_a
#                a_increment = self.new_a - self.old_a
#                step_size_a = ((a_increment * gradient_a_increment)/
#                               (gradient_a_increment * gradient_a_increment))
#
#                """STEP 2.1.2: For b"""
#                gradient_b_increment = gradient_b - self.old_gradient_b
#                b_increment = self.new_b - self.old_b
#                step_size_b = ((b_increment * gradient_b_increment)/
#                               (gradient_b_increment * gradient_b_increment))
#
#                """STEP 2.1.3: For beta"""
#                gradient_beta_increment = gradient_beta - self.old_gradient_beta
#                beta_increment = self.new_beta - self.old_beta
#                step_size_beta = (np.inner(beta_increment,
#                    gradient_beta_increment)/
#                    np.inner(gradient_beta_increment, gradient_beta_increment))
#
#                """STEP 2.1.4: For L"""
#                gradient_L_increment = gradient_L - self.old_gradient_L
#                L_increment = ( (self.new_L - self.old_L )[
#                        np.tril_indices(self.num_regressors, 0)])
#                step_size_L = (np.inner(L_increment,
#                        gradient_L_increment)/
#                        np.inner(gradient_L_increment, gradient_L_increment))
#            else:
#                step_size_a, step_size_b = 1.0/count, 1.0/count
#                step_size_beta, step_size_L = 1.0/count, 1.0/count
#
#
#        """STEP 2.2: store gradient values for next iteration's step sizes"""
#        self.old_gradient_a = gradient_a
#        self.old_gradient_b = gradient_b
#        self.old_gradient_beta = gradient_beta
#        self.old_gradient_L = gradient_L
#
#
#        """STEP 3: Take the step into the gradient direction"""
#
#        """STEP 3.1: Get the increments for each VB parameter"""
#        increment_a = gradient_a * step_size_a
#        increment_b = gradient_b * step_size_b
#        increment_beta = gradient_beta * step_size_beta
#        increment_L = gradient_L * step_size_L
#
#
#        """STEP 3.2: Update the values of the VB parameters"""
#
#        """STEP 3.2.1: Do it for a"""
#        self.a_rt[r] = min(max(self.a_rt[r] + increment_a, 1.0001), pow(10,6)) #we want a>1
#
#        if count == 0:
#            self.new_a = self.a_rt[r] #i.e., we store only the updated value
#        else:
#            self.old_a = self.new_a
#            self.new_a = self.a_rt[r]
#
#        """STEP 3.2.2: Do it for b"""
#        self.b_rt[r] = min(max(self.b_rt[r] + increment_b, 1.0001), pow(10,6)) #we want b>0
#
#        if count == 0:
#            self.new_b = self.b_rt[r] #i.e., we store only the updated value
#        else:
#            self.old_b = self.new_b
#            self.new_b = self.b_rt[r]
#
#        """STEP 3.2.3: Do it for beta"""
#        self.beta_rt[r,:] = self.beta_rt[r,:] + increment_beta
#
#        if count == 0:
#            self.new_beta = self.beta_rt[r,:] #i.e., we store only the updated value
#        else:
#            self.old_beta = self.new_beta.copy()
#            self.new_beta = self.beta_rt[r,:]
#
#        """STEP 3.2.4: Do it for L"""
#        lower_ind = np.tril_indices(self.num_regressors, 0)
#        self.L_rt[r,:,:][lower_ind] = self.L_rt[r,:,:][lower_ind] + increment_L
#
#        if count == 0:
#            self.new_L = self.L_rt[r,:,:] #i.e., we store only the updated value
#        else:
#            self.old_L = self.new_L.copy()
#            self.new_L = self.L_rt[r,:,:]
#
#    #DEPRECATED!
#    def get_gradient_a(self, r, num_obs, alpha = None, prior_weight = 1):
#        # an, bn, betan, vech_Ln, p, num_obs, alpha,  a0, b0, beta0, vechL0,
#        #                         K, E3, Rinv
#        """Gradient of ELBO w.r.t. IG-a based on the last r observations
#        I.e., for r = 0, we would simply have the prior. For r = 3, we would
#        compute the full gradient based on X_t, X_t-1, X_t-2.
#        NOTE: Using notation of the appendix for the NIPS submission,
#        see details for the expressions (and how to arrive there) there
#        """
#
#        """Unless otherwise specified, work with value of alpha_param"""
#        if alpha is None:
#            alpha = self.alpha_param
#
#        """STEP 1: Compute the prior-dependent (data-independent) quantities
#        Note that the derivatives of E_9 and E_10 cancel except for the
#        trigamma term. So we give E10_der with only said trigamma term."""
#        #RUNTIME OVERFLOW due to E2 (which is in E2der) and self.K [b^a]
#        d = self.S1*self.S2
#        E2der_m_ba = self.E2_m_ba*(self.digamma_diff_2 + np.log(self.b_rt[r]))
#        E8der =  (-0.5 * (np.inner(self.prior_mean_beta - self.beta_rt[r,:],
#                            np.matmul(self.prior_var_beta_inv,
#                             self.prior_mean_beta - self.beta_rt[r,:]
#                            )
#                        )
#                        + 2*(self.b - self.b_rt[r])
#                        )
#                 * (1.0/self.b_rt[r])
#                 *  self.gamma_ratio_1
#                 *  self.digamma_diff_1)
#        E9der = -np.log(self.b_rt[r]) + scipy.special.digamma(self.a_rt[r])
#        E10der = ( np.log(self.b_rt[r]) - scipy.special.digamma(self.a_rt[r])
#                    -(self.a_rt[r] - self.a)
#                      * scipy.special.polygamma(1, self.a_rt[r]))
#        E11der = -(
#                    (num_obs * self.gamma_ratio_2 * self.digamma_diff_2 )
#                    * pow(self.b_rt[r] * 2.0 * np.pi,
#                            -0.5*d * alpha)
#                    * pow(1.0 + alpha,
#                              -0.5*d - 1)
#                )
#
#        """STEP 2: Compute the data-dependent quantitites
#        Note: The heavy lifting for this is done in the pre-computation"""
#        #RUNTIME OVERFLOW due to E2 (which is in E2der) and self.K [b^a]
#        if num_obs > 0:
#
#            sum_1 = (E2der_m_ba
#                         * self.downweight1
#                         * np.sum(self.E3
#                            * np.power(self.b_rt[r]/self.K,
#                                self.a_rt[r] + 0.5*d*alpha)
#                         )
#                    )
#
#
#
#            signs = np.sign(self.E3) * np.sign(self.K)
#            sum_1_log, sign =   ( #E2der_m_ba * (
#                scipy.misc.logsumexp(
#                    a = (
#                        np.log(np.abs(self.E3)) +
#                        #(self.a_rt[r] + 0.5*self.num_regressors*alpha)*
#                        #    np.abs(np.log(b)),
#                        (np.log(self.b_rt[r]) - np.log(np.abs(self.K))) *
#                            (self.a_rt[r] + 0.5*d*alpha)
#                        #
#                    ),
#                    axis=0,
#                    b = signs,
#                    return_sign = True
#
#                    )
#                )
#
#            sum_1_ = sign * np.exp(sum_1_log - (0.5*d*alpha) * np.log(self.b_rt[r])) * E2der_m_ba
#
#
#            sum_2 = (
#                    self.E2_m_ba * (-1.0)
#                    * self.downweight1
#                    * np.sum(self.E3 *
#                        np.power(self.b_rt[r]/self.K,
#                                 self.a_rt[r] + 0.5*d*alpha)
#                        #DEBUG: Force K > 0
#                        *  np.log(self.K)
#                    )
#                )
#
#            signs_ = np.sign(self.E3) * np.sign(self.K) * np.sign(np.log(self.K))
#            sum_2_log, sign_ = (
#                    scipy.misc.logsumexp(
#                        a = (
#                            np.log(np.abs(self.E3))
#                            + (np.log(self.b_rt[r]) - np.log(np.abs(self.K)))
#                                *(self.a_rt[r] + 0.5*d*alpha)
#                            + np.log(np.abs(np.log(self.K)))
#                            ),
#                        b = signs_,
#                        axis=0,
#                        return_sign=True
#                        )
#                )
#
#
#        else:
#            sum_1, sum_2 = 0.0, 0.0
#        #DEBUG: Presupposes that self.K is positive in every entry!
#        #       not sure this holds, need some precaution!
#        #DEBUG: Give a warning whenever some entries of K are negative!
#        if np.sum(self.K < 0 ) > 0 and num_obs > 0:
#            print("number of negative K entries: ", np.sum(self.K < 0 ))
#
#        """STEP 3: Sum up everything and return it"""
#        ELBOder_a = prior_weight*(E8der + E9der + E10der + E11der) + sum_1 + sum_2
#        return ELBOder_a #  ELBOder_a #E8der +E9der + E10der + E11der #ELBOder_a
#
#
#    #DEPRECATED!
#    def get_gradient_b(self, r, num_obs, alpha, prior_weight = 1):
#        """Gradient of ELBO w.r.t. IG-a based on the last r observations
#        I.e., for r = 0, we would simply have the prior. For r = 3, we would
#        compute the full gradient based on X_t, X_t-1, X_t-2.
#        NOTE: Using notation of the appendix for the NIPS submission,
#        see details for the expressions (and how to arrive there) there
#        """
#
#        """Unless otherwise specified, work with value of alpha_param"""
#        if alpha is None:
#            alpha = self.alpha_param
#
#        """STEP 1: Compute the prior-dependent (data-independent) quantities,
#        i.e. the derivatives of E_2, E_8,E_9,E_10,E_11. Note that the sum of
#        the derivatives of E_9 and E_10 is -a0/bn, so we only give E10der"""
#        d = self.S1*self.S2
#        E2der_m_ba = self.E2_m_ba * (self.a_rt[r]/self.b_rt[r])
#        E8der = ( 0.5  * (np.inner(self.prior_mean_beta - self.beta_rt[r,:],
#                                    np.matmul(self.prior_var_beta_inv,
#                                      self.prior_mean_beta - self.beta_rt[r,:]
#                                    )
#                                )
#                            #- 2*self.b
#                          )
#                        * (1.0/pow(self.b_rt[r],2))
#                        * self.gamma_ratio_1
#                 )
#        E8der = E8der + self.b * (1.0/pow(self.b_rt[r],2)) * self.gamma_ratio_1
#        E10der = -(self.a/self.b_rt[r])
#        E9der = -(self.a_rt[r]/self.b_rt[r])
#        E10der = (self.a_rt[r] - self.a)/self.b_rt[r]
#        E10der = E9der + E10der
#        E11der =  (
#                (
#                    (num_obs * 0.5 * d * alpha)
#                    * self.gamma_ratio_2
#                )
#                *
#                (
#                    pow(2 * np.pi, -0.5 * d * alpha)
#                    * pow(1 + alpha, -0.5 * d - 1)
#                    * self.downweight2
#                )
#            )
#
#        """STEP 2: Compute the data-dependent quantitites
#        Note: The heavy lifting for this is done in the pre-computation"""
#        #RUNTIME OVERFLOW due to E2 (which is in E2der) and self.K [b^a]
#        if num_obs > 0:
#            sum_1 = (
#                E2der_m_ba
#                    * self.downweight1
#                    * np.sum(
#                        self.E3
#                        * np.power(self.b_rt[r]/self.K,
#                               self.a_rt[r] + 0.5*d*alpha)
#                    )
#                )
#
#        #RUNTIME OVERFLOW due to E2 and self.K [b^a]
#            sum_2 = (
#                self.E2_m_ba
#                * (-self.a_rt[r] - 0.5*d*alpha)
#                * self.downweight2
#                * np.sum(self.E3
#                    * np.power(self.b_rt[r]/self.K,
#                        self.a_rt[r] + 0.5*d*alpha + 1)
#                  )
#                )
#        else:
#            sum_1 = sum_2 = 0.0
#
#        """STEP 3: Sum up everything and return it"""
#        ELBOder_b = prior_weight*(E8der + E10der + E11der) + sum_1 + sum_2
#        return ELBOder_b #ELBOder_b
#
#    #DEPRECATED
#    def get_gradient_beta(self, r, num_obs, alpha, prior_weight=1):
#        """Gradient of ELBO w.r.t. beta based on the last r observations
#        I.e., for r = 0, we would simply have the prior. For r = 3, we would
#        compute the full gradient based on X_t, X_t-1, X_t-2.
#        NOTE: Using notation of the appendix for the NIPS submission,
#        see details for the expressions (and how to arrive there) there
#        """
#
#        """Unless otherwise specified, work with value of alpha_param"""
#        if alpha is None:
#            alpha = self.alpha_param
#
#        """STEP 1: Compute the prior-dependent (data-independent) quantities,
#        i.e. the derivatives of E_4, E_8"""
#        d = self.S1*self.S2
#        E4der =( 2 * np.matmul(
#                        np.matmul(
#                               np.transpose(self.beta_rt[r,:]),
#                               self.L_rt[r,:,:]
#                              ),
#                        np.transpose(self.L_rt[r,:,:])
#                    )
#                )
#        E8der = (
#                    (1.0/self.b_rt[r]) * self.gamma_ratio_1
#                     * np.matmul(
#                        np.transpose(self.prior_mean_beta - self.beta_rt[r,:]),
#                        self.prior_var_beta_inv,
#                    )
#                 )
#        """STEP 2: Compute the data-dependent quantitites
#        Note: The heavy lifting for this is done in the pre-computation"""
#
#        """STEP 2.1: Get the quantities that are functionals of the precomputed
#        quantities R, the derivatives of E_{5,i} and E_{7,i}"""
#        if num_obs > 0:
#            E7der = np.zeros((num_obs,self.num_regressors))
#            E5der = np.zeros((num_obs, self.num_regressors))
#            for i in range(0, num_obs):
#                """COMPUTATIONAL NOTE: Probably:INEFFICIENT: Since both XY and
#                R in an array, s.t. I can do matrix mult along first axis?"""
#                E5der[i,:] = np.matmul(E4der, self.Rinv[i,:,:])
#                E5der[i,:] = np.matmul(E5der[i,:], (self.L_rt[r,:,:]))
#                E5der[i,:] = -np.matmul(E5der[i,:],
#                                     np.transpose(self.L_rt[r,:,:]))
#                E7der[i,:] = np.matmul(np.transpose(self.XY_t[i,:]),
#                                  self.Rinv[i,:,:])
#                E7der[i,:] = np.matmul(E7der[i,:], self.L_rt[r,:,:])
#                E7der[i,:] = np.matmul(E7der[i,:], np.transpose(self.L_rt[r,:,:]))
#
#            E7der = -2 * alpha * E7der
#
#            """STEP 2.2: Get the summed quantities that are themselves functionals
#            of the previously derived E5der, E7der entries."""
#            #RUNTIME OVERFLOW due to E2  and self.K [b^a]
#            sum_1 = (
#                        (-self.a_rt[r] -0.5*d*alpha)
#                        * 0.5
#                        * self.downweight2
#                        * self.E2_m_ba * (
#                        np.sum(
#                            self.E3[:,np.newaxis]
#                            * (E4der + E5der +E7der)
#                            * np.power(self.b_rt[r]/self.K,
#                                self.a_rt[r] + 0.5*d*alpha + 1)[:,np.newaxis]
#                            , axis=0
#                        )
#                    )
#                )
#
#            """STEP 3: Sum up everything and return it"""
#            ELBOder_beta = prior_weight*E8der + sum_1
#        elif num_obs <= 0:
#            ELBOder_beta = prior_weight * E8der
#
#        return ELBOder_beta
#
#    #DEPRECATED
#    def get_gradient_L(self, r, num_obs,  alpha, prior_weight = 1):
#        """Gradient of ELBO w.r.t. L based on the last r observations
#        I.e., for r = 0, we would simply have the prior. For r = 3, we would
#        compute the full gradient based on X_t, X_t-1, X_t-2.
#        NOTE: Using notation of the appendix for the NIPS submission,
#        """
#
#        """Unless otherwise specified, work with value of alpha_param"""
#        if alpha is None:
#            alpha = self.alpha_param
#
#        """STEP 1: Compute the prior-dependent (data-independent) quantities,
#        i.e. the derivatives of E1, E2, E_4"""
#        d = self.S1*self.S2
#        lower_ind = np.tril_indices(self.num_regressors, 0)
#        B = np.outer(self.beta_rt[r,:], self.beta_rt[r,:])
#        L_inv = scipy.linalg.lapack.clapack.dtrtri(self.L_rt[r,:,:], 1)[0]
#        E1der = (np.matmul(
#                         (np.matmul(
#                                np.matmul(np.transpose(L_inv), L_inv),
#                                self.prior_var_beta_inv
#                            )
#                         -  np.identity(self.num_regressors)),
#                            np.transpose(L_inv)
#                        )[lower_ind]     )
#        E2der_m_ba = self.E2_m_ba * np.transpose(L_inv)[lower_ind]
#        E4der = 2.0 * np.matmul(B, self.L_rt[r,:,:])[lower_ind]
#
#        """STEP 2: Compute the data-dependent quantitites.
#        For efficiency, compute the entire derivative term inside the sum
#        observation by observation. I.e., we directly compute the two sums
#        depending on derivatives of E_{3,i}, E_4, E_{5,i}, E_{6,i}, E_{7,i}"""
#
#        """COMPUTATIONAL NOTE: INEFFICIENT,
#        maybe one can vectorize these operations? I.e., perform all of this in
#        one line (using np.einsum, e.g.) instead of doing it in a loop"""
#        if num_obs > 0:
#            sum_ders = np.zeros(int(self.num_regressors*
#                                    (self.num_regressors+1)*0.5))
#            for i in range(0,num_obs):
#                """STEP 2.1: Compute auxiliary quantities needed for derivatives of
#                E_{3,i}, E_4, E_{5,i}, E_{6,i}, E_{7,i}, namely A1 - A6"""
#                A1 = np.matmul(self.Rinv[i,:,:], self.L_rt[r,:,:])
#                A2 = np.matmul(A1, np.transpose(self.L_rt[r,:,:]))
#                A3 = np.matmul(A2, B)
#                A4 = np.matmul(self.Rinv[i,:,:], self.XY_t[i,:]) #vector!
#                A5 = np.matmul(np.transpose(self.XY_t[i,:]), A1)  #vector!
#                A6 = np.matmul(np.identity(self.num_regressors) - A2,
#                               self.beta_rt[r,:])  #vector!
#
#                """STEP 2.2: Compute  E_{3,i}, E_4, E_{5,i}, E_{6,i}, E_{7,i} for
#                the current value of i"""
#                E3ider = -self.E3[i] * A1[lower_ind]
#                E5ider = (2 *
#                            np.matmul(
#                                (
#                                  np.matmul(A2 - np.identity(self.num_regressors),
#                                    np.transpose(A3))
#                                  - A3
#                                ),
#                                  self.L_rt[r,:,:]
#                            )[lower_ind]
#                        )
#                E6ider = 2 * pow(alpha, 2) * np.outer(A4, A5)[lower_ind]
#                E7ider = -2 * alpha * (
#                            np.outer(A6, A5)
#                            + np.matmul(
#                                np.outer(A4, A6),
#                                self.L_rt[r,:,:]
#                            )
#                        )[lower_ind]
#
#                """STEP 2.3: add the next term to the sum derivative"""
#
#                """STEP 2.3.1: Add the sum-term involving E3ider"""
#                sum_ders = sum_ders + (
#                            E3ider
#                            * pow(self.b_rt[r]/self.K[i],
#                                self.a_rt[r] + 0.5*d*alpha)
#                            * self.downweight1
#                            )
#
#                """STEP 2.3.2: Add the sum-term involving E4der - E7ider"""
#                #RUNTIME OVERFLOW due to  self.K [b^a]
#                sum_ders = sum_ders + (
#                            self.E3[i]
#                            * (-self.a_rt[r] - 0.5*d*alpha) * 0.5
#                            * (E4der + E5ider + E6ider + E7ider)
#                            * pow(self.b_rt[r]/self.K[i],
#                                self.a_rt[r] + 0.5*d*alpha + 1)
#                            * self.downweight2
#                        )
#
#            """STEP 2.4: Lastly, multiply the sum by E2 (so you have 1
#            multiplication instead of 2 * r multiplications)"""
#            #RUNTIME OVERFLOW due to E2  [b^a]
#            sum_ders = sum_ders * self.E2_m_ba
#
#            """STEP 2.5: Compute the quantities we can get directly from the
#            pre-computations made earlier in this optimization cycle"""
#            sum_ders = sum_ders + E2der_m_ba * np.sum(
#                            self.E3
#                            * np.power(self.b_rt[r]/self.K,
#                                self.a_rt[r] + 0.5 * d * alpha)
#                            * self.downweight1
#                        )
#
#            """STEP 3: Add the last remaining term and return everything"""
#            ELBOder_L = prior_weight*E1der + sum_ders
#        elif num_obs <= 0:
#            ELBOder_L = prior_weight*E1der
#        return ELBOder_L
#
#    #DEPRECATED
#    def get_ELBO(self, perturbation, r, num_obs):
#        """Let perturbation be (p + p*(p+1)*0.5 + 1 + 1) vector of small
#        changes for beta, L, a, b"""
#
#        """STEP 1: Perturb the quantities"""
#        d = self.S1 * self.S2
#        p = self.num_regressors
#        lower_ind = np.tril_indices(self.num_regressors, 0)
#        beta = (self.beta_rt[r,:] + perturbation[:p]).copy()
#        L = self.L_rt[r,:,:].copy()
#        L[lower_ind] = L[lower_ind] + perturbation[p:-2]
#        Linv = scipy.linalg.lapack.clapack.dtrtri(L, 1)[0]
#        a = self.a_rt[r] + perturbation[-2]
#        b = self.b_rt[r] + perturbation[-1]
#
#
#        """STEP 2: Compute the E-quantities not depending on observations"""
##a        print("trace", np.trace(self.L_0_inv), np.trace(L) )
##        print("direct", np.linalg.slogdet(self.L_0_inv), np.linalg.slogdet(L))
#
#        E1 = (- np.log(abs(np.prod(np.diag(self.L_0)))) - np.log(abs(np.prod(np.diag(L))))
#              + 0.5 * (p - np.trace(
#                              np.matmul(self.prior_var_beta_inv,
#                                        np.matmul(Linv,
#                                                  np.transpose(Linv))
#                                        )
#                            )
#                       )
#            )
#
#        E2_m_ba = (np.exp(scipy.special.gammaln(a + d*0.5*self.alpha_param) -
#                          scipy.special.gammaln(a))
#                    * (1.0/self.alpha_param)
#                    * pow(2 * np.pi, -0.5*d*self.alpha_param)
#                    * abs(np.prod(np.diag(L)))
#                    ) #np.exp(np.trace(L)))
#
##        print("local", E2_m_ba)
##        print("global", self.E2_m_ba)
#
#        Ltbeta = np.matmul(np.transpose(L), beta)
#        E4 = np.inner(np.transpose(Ltbeta), Ltbeta)
#
#        #have checked them. These should be correct!
#        #{{{{
#        E8 = -(0.5
#              * (1.0/b)
#              * a #np.exp(scipy.special.gammaln(a + 1) - scipy.special.gammaln(a))
#              * (np.inner(self.prior_mean_beta - beta ,
#                          np.matmul(self.prior_var_beta_inv,
#                                    self.prior_mean_beta - beta))
#                 + 2*(self.b - b)
#                )
#             )
#        #E8 = E8 + ((self.b_rt[r] - self.b)* (1.0/b)
#        #     * np.exp(scipy.special.gammaln(a + 1) - scipy.special.gammaln(a)))
##        E8old = (0.5
##              * (1.0/b)
##              * np.exp(scipy.special.gammaln(a + 1) - scipy.special.gammaln(a))
##              * (np.inner(self.prior_mean_beta - beta ,
##                          np.matmul(self.prior_var_beta_inv,
##                                    self.prior_mean_beta - beta))
##                )
##             )
#        E9 = -(a*np.log(b) + scipy.special.gammaln(self.a)
#                - self.a*np.log(self.b) - scipy.special.gammaln(a) )
#        E10 = -(a - self.a) * (scipy.special.digamma(a) - np.log(b))
#        E11 =-(  num_obs
#                * np.exp(scipy.special.gammaln(a + 0.5*d*self.alpha_param) -
#                      scipy.special.gammaln(a))
#                * pow(b*2*np.pi, -0.5*d*self.alpha_param)
#                * pow(1 + self.alpha_param, -0.5*d - 1)
#            )
#        #}}}
#
#        #print("gamma ratio:", np.exp(scipy.special.gammaln(a + 1) - scipy.special.gammaln(a)))
#        #print("E8", E8) #-1.0000001, -0.99999999
##        if perturbation[-2]!= 0:
##            print("E9", E9) #-1
##            print("E10", E10) # 0
##            print("E11", E11) #-4
#
#
#        """STEP 3: Compute data-dependent quantities"""
#        sum_part = 0.0
#        if num_obs > 0:
#            for i in range(0, num_obs):
#                """QR decomposition of R-matrix (alpha * XX + Sigma_n)"""
#                Q, R = np.linalg.qr(
#                            self.alpha_param * self.XX_t[i,:,:]
#                            + np.matmul(L, np.transpose(L))
#                        )
#                Rinv = scipy.linalg.lapack.clapack.dtrtri(R, 0)[0]
#
#                """inverse of R-matrix"""
#                #DEBUG: different from R_inv as stored in object? NO!
#                R_i_inv = np.matmul(Rinv, np.transpose(Q))
##                print("local", R_i_inv)
##                print("object", self.Rinv[i,:,:])
##                print("R dir", np.matmul(Q, R))
##                print("R via inv", np.linalg.inv(R_i_inv))
#
#                """Get E3i, i.e. |R|^-0.5. Note that det(Q) = 1 (orthogonality)"""
#                E3i = pow( abs(np.prod((np.diag(R)))), -0.5)
##                print("using R", pow( abs(np.prod((np.diag(R)))), -0.5))
##                print("directly", np.linalg.det(self.alpha_param * self.XX_t[i]
##                            + np.matmul(L, np.transpose(L))))
##                print("stored version", self.E3[i])
#
#                """Get (E4 + E5 + E6 + E7 + alpha * y'y)*0.5 + bn"""
#                vec = (np.matmul(L, Ltbeta) + self.alpha_param * self.XY_t[i,:])
#                E4 = np.inner(np.transpose(Ltbeta), Ltbeta)
#                E5_ = np.matmul(np.transpose(Ltbeta), np.transpose(L))
#                E5 = -np.matmul(E5_, np.matmul(R_i_inv, np.transpose(E5_)))
#                #DEBUG: Ensure K > 0 to make comparable to direct gradients
#                E6 = -pow(self.alpha_param, 2) * np.inner(
#                        np.transpose(self.XY_t[i,:]),
#                        np.matmul(R_i_inv, self.XY_t[i,:])
#                    )
#                E7 = -2 * self.alpha_param * np.matmul(
#                        np.matmul(
#                                    np.matmul(np.transpose(Ltbeta),
#                                              np.transpose(L)),
#                                    R_i_inv
#                                ),
#                        self.XY_t[i,:]
#                        )
#
#                term = (b + 0.5 * (
#                                self.alpha_param * self.YY_t[i]
#                                + E4
#                                - np.inner(vec, np.matmul(R_i_inv, vec))
#                        ))
#                if perturbation[-2] > 0:
#                    term = max(term, pow(10,-5))
#
#                #check if term = K[i]!
##                print("term", term)
##                print("Ki", self.K[i])
#
#                """integrate the b^a term into it"""
#                term_p_ba = (
#                                pow(b/term, a + 0.5 * d * self.alpha_param)
#                                * pow(b, -0.5*d*self.alpha_param)
#                            )
#
#    #            if self.K[i] != term:
#    #                print("K and term not same!")
#    #                print("K[i]", pow(b/self.K[i], a + 0.5 * p * self.alpha_param)
#    #                            * pow(b, -0.5*p*self.alpha_param))
#    #                print("term", term_p_ba)
#
#
#    #            print("local", E3i)
#    #            print("global", self.E3[i])
#
#                """put the term together and add to sum part"""
#                sum_part = sum_part + term_p_ba * E3i
#
#        return (E1 + E8 + E9 + E10 + E11 + E2_m_ba*sum_part)
#
