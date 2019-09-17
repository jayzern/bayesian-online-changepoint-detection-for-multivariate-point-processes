# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:30:37 2017
@author: Jeremias Knoblauch (J.Knoblauch@warwick.ac.uk)

Description: Implements Bayesian Linear Autoregression with the NIG model
(i.e., spatial locations have iid errors with common variance)
"""

import numpy as np
from scipy import special
from scipy import linalg
from scipy import stats
import scipy
from probability_model import ProbabilityModel
#from nearestPD import NPD


class BVARNIG(ProbabilityModel):
    """The Bayesian Vector Autoregression model using past observations as
    regressors in a specified neighbourhood. E.g., if the 4-neighbourhood is
    selected with lag length 1, then the mean of y_{t,i} is modelled as linear
    combination of observations y_{t-1, j} \in nb(i). Around the boundary,
    the neighbourhoods are 0-padded.

    ###****************************************************************###
    ###                            MODEL PRIORS                        ###
    ###****************************************************************###
    Inputs always needed. They correspond to priors in the following model:
            Y ~ N(X*beta, sigma2 * I),
            beta ~ N(beta_0, sigma2 * V_0),
            sigma2 ~ IG(a,b)

    prior_a: float >0:
        a parameter of the Inverse Gamma, a>0

    prior_b: float >0:
        b parameter of the Inverse Gamma, b>0

    prior_mean_beta: 1D-numpy array of size k, k = num regressors:
        corresponds to beta_0, i.e. the mean prior of coefficients.
        Takes precedence over prior_mean_scale if both specified.

    prior_var_beta: 2D-numpy array of size kxk, k = num regressors:
        corresponds to V_0, i.e. the covariance prior of coefs
        Takes precedence over prior_var_scale if both specified.

    prior_mean_scale: float:
        If prior_mean_beta is None, prior_mean_scale supplied, the number
        of regressors k is calculated automatically and
        beta_0 = prior_mean_scale * np.ones(k)

    prior_var_scale: float >0:
        If prior_var_beta is None, prior_var_scale supplied, the number
        of regressors k is calculated automatically and
        beta_0 = prior_var_scale * np.identity(k)


    ###****************************************************************###
    ###                 REGULAR GRID + STRONG PARAM BINDING            ###
    ###****************************************************************###
    Inputs needed when assuming regular grid with strong parameter binding:
        nbh_sequence, restriction_sequence, padding

    nbh_sequence: array with integer entries, only needed if data on
                  regular grid, with strong coupling between effects:
        0, 4, 8   -> specify the sequence ofVAR-nbhs.
                     corresponds to strong parameter coupling on regular
                     grid with no neighbourhood (0), 4-neighbourhood (4),
                     and 8-neighbourhood (8). I.e. all locations are on
                     a grid defining their nbhs, and share params.
                     (See restriction_sequence for param sharing)

    restriction_sequence: array with integer entries, only needed if data
                   on regular grid, with strong coupling between effects:
        0, 4, 8 -> specify the restriction of nbh_sequence on regular
                    spatial grid with parameter coupling.
                    Regardless of 0,4,8, we always couple across all
                    LOCATIONS! I.e., params the same across the grid.
                    However, we can vary how much we couple params within
                    each location's nbh: Not at all, i.e. one parameter
                    for each nbh location relative to each location (0),
                    the 4 inner and the 4 outer (4), and in the case of
                    a 8-nbh, all 8 together (8). See Fig. 2 in the paper
                    for illustration of 4-nbh (red), 8 nbh (red + blue),
                    0 nbh (orange).

                    NOTE: The parameter bindings for the intercepts are
                    again specified via intercept_grouping (see below).
                    They are NOT strongly coupled unless the argument
                    is not specified or is supplied as None.

    padding: string:
        ONLY needed if we specify nbh_sequence and restriction_sequence,
        implying that we are on a regular grid. Then, we need to pad the
        outside of the grid using one of the below options:
        'overall_mean'  -> compute mean across space and fill in
        'row_col_mean'  -> compute row and col means and fill in
        'zero'          -> insert zeros (bias estimation towards 0)
        'leave-out'     -> don't pad at all, and estimate only using
                            locations with full neighbourhood


    ###****************************************************************###
    ###                 GENERAL NBHS + ANY PARAM BINDING               ###
    ###****************************************************************###
    Inputs needed when assuming general nbh structures with arbitrary
    parameter bindings:
        intercept_grouping, general_nbh_sequence,
        general_nbh_restriction_sequence , general_nbh_coupling

    intercept_grouping: GxS1xS2 numpy array of ones or zeros grouping the
        locations into G groups so that each group shares the intercept.
        Notice that summing over the G-dimension, we would get an S1xS2
        array of only ones. I.e., each location has to be in one of the G
        groups. Extreme cases: G=1 with a single slice of ones => all
        locations have one shared intercept. G=S1*S2 with each of the G
        slicescontaining exactly a single 1-entry and only zeros otherwise
        => each location has individual intercept.

    general_nbh_sequence: list of list of lists:
        Gives an nparray of nparrays of nparrays of
        coordinates/identifiers, i.e. an object like
        [[[2,3,4],[5,6],[7]], [[5,6],[8],[9,10]], ...].
        Here, [2,3,4] would be the 'closest' nbh to the
        point with spatial coordinate 0, [5,6] the second-
        closest, [7] the third-closest. how far away from
        the closest nbh you consider the data is implied
        by the general_nbh_restriction_sequence that
        will give you the indices of the nbhs to be
        considered for each lag length.

        In the notation of the PAPER, this gives you the nbh. system as
        [[N_1(1), N_2(1), N_3(1)], [N_1(2), N_2(2), N_2(3)], ...], i.e.
        list entry s belongs to location with index s and contains n neigh-
        bourhoods N_1(s), ... N_n(s) s.t. the indices describe spatial
        closeness, with smaller indices indicating that we are closer to s.
        Note that we assume n to be the same for all locations. If there
        is a case where you assume that some locations s have less nbhs
        than others, simply add some empty nbhs, i.e. N_i(s) = [].

    general_nbh_restriction_sequence: list of list:
        Gives you a list of lists of indices, i.e.
        [[0,1,2,3], [0,1,2], [0],[]], where it must hold that
        later entries are strict subsets of previous ones
        s.t. the largest value at position l is at most as
        big as the largest value at position l-1. Also, if
        k is the largest value at position l, it must hold
        that all k' s.t. 0<=k'<=k must be in that list entry
        NOTE: If you want to have only auto-regressive
        terms at some nbh (I.e., only its own past influen-
        ces the present), then simply add an empty list [].

        In the notation of the PAPER, this is the function p(.) assigning
        temporal meaning to the neighbourhood structure. p is given in
        list form so that the l-th entry of p gives all indices that
        are going to be used at lag length l. I.e., assuming p(l) = 3
        (using p's definition from the paper), it will hold that the
        respective entry in general_nbh_restriction_sequence is going to
        be [0,1,2]. More generally, for p(l) = k, [0,1,...,k-1].

    general_nbh_coupling: string:
        ["no coupling", "weak coupling",
        "strong coupling"], tells you how neighbourhoods
        are tied together. "no coupling" means that each
        linear effect of s' \in N_i(s) is modelled sepa-
        rately. "weak coupling" means that the linear
        effect for all s' \in N_i(s) are modelled together,
        and "strong coupling" means that the linear effects
        are also modelled together across space, i.e.
        s' \in N_i(s) and g \in N_i(k) have the same effect
        (but s' in N_j(s) and g in N_i(k) do not)
        NOTE: no coupling is not implemented, because you
        can obtain the same effect by weak coupling and
        treating each station as its own nbh.

        In the PAPER, notes on this are right after SSBVAR definition.
        "weak coupling" is the standard modelling framework that assumes
        that for all locations in a given nbh, we have a single linear
        effect. "strong coupling" means that in addition, we have the same
        linear neighbourhood effect for each location.


    ###****************************************************************###
    ###                 HYPERPARAMETER LEARNING                        ###
    ###****************************************************************###
    Inputs needed when doing hyperparameter learning:
        hyperparameter_optimization [don't use auto_prior_update!]

    hyperparameter_optimization (ProbabilityModel level): string or None:
        -> [True, False, None, "caron", "online", "turner"]
        by default, this is True, which amounts to updating
        the gradients but not performing on-line/caron's
        hyperpar. learning. If False or None, the gradients
        are not updated. "caron" and "online" both employ
        the on-line hyperparameter learning proposed by
        Caron, Doucet, Gottardo (2012). If you don't want
        this, but only want to do Turner's routine, you
        have to do so via an enclosing HyperparameterOptimization
        object. For this, put hyperparameter_optimization
        to True (default) or "turner".

        I.e., "turner", "True" mean that gradients are updated recursively,
        but not used (unless an enclosing HyperparameterOptimization
        object uses them), "caron" and "online" mean that we perform
        gradient descent updates as in the PAPER. "False" and None mean
        that we don't update the gradients. (barely any computational
        benefit in so doing)

    auto_prior_update: boolean.
        Basically, DON'T set to True. It updates the priors by setting them
        to the posterior expectations at time t. For instance, the beta_0
        prior at time t will be set to
            sum_r{ beta_rt[r,:] * P(r|y_1:t) }.


    ###****************************************************************###
    ###                EXOGENEOUS/ADDITIONAL PREDICTORS                ###
    ###****************************************************************###
    NOT IMPLEMENTED!!!!
    Inputs needed when using additional variables:
        exo_selection, nbh_sequence_exo


    NOTE: Intercepts, EXO, and ENDO vars can always ge grouped by the
          following simple procedure: Suppose you have two groups G1, G2.
          Let's assume you assume the same model in G1, G2 but with diff-
          erent parameterizations. Lets say the params you want are
          a(G1), a(G2), b(G1), b(G2). Then you can just estimate all four
          coefs jointly by having G1 have a column of zeros for the var
          corresponding to a(G2), b(G2) and vice versa.

    NOTE: At some point, it may be good to replace the strings indicating
          our neighbourhood structures using integers instead, since
          string-computations are more expensive than integer-computations

    exo_selection:
       0,1,2,..     -> gives you a selection vector of length
                            num_exo_vars allowing you to select which exos
                            you want to regress on Y. The integers are
                            the row index in vector [exo1, exo2, ...] of
                            regressors available at each location.

    nbh_sequence_exo: #not in the input
        0,4,8           -> gives you the nbh of the lagged exos that are
                            regressors for your problem. Starts at time t
                            (rather than t-1, as for endo sequence)

    ###****************************************************************###
    ###                        OTHER INPUTS                            ###
    ###****************************************************************###
    None of these inputs are needed, they provide additional functionality

    non_spd_alerts: boolean:
        Gives an alert whenever the covariance matrix was not semi-positive
        definite and needed to be converted into an spd-matrix by forcing
        it via 'nearestPD' or adding a disturbance.

        NOTE: If you experience this a lot, try to rescale your data, i.e.
        normalize it on-line or do something along the same lines.
    """



    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    """              OBJECT INITIALIZATION FUNCTIONS                    """
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    def __init__(self,
                 prior_a,
                 prior_b,
                 S1,
                 S2,
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
                 #deprecated argument, should go
                 auto_prior_update=False,
                 hyperparameter_optimization = "online",
                 general_nbh_coupling = "strong coupling",
                 non_spd_alerts =False
                ):


        """STEP 1: Store priors"""
        self.a, self.b = prior_a, prior_b
        """if beta_0 or beta's covariance matrix are specified, that takes
        precedence over a supplied scaling of a vector/matrix of ones"""
        if not prior_mean_beta is None:
            self.prior_mean_beta = prior_mean_beta.flatten()
        else:
            self.prior_mean_beta= prior_mean_beta
        self.prior_var_beta= prior_var_beta


        """STEP 2: Store execution parameters"""
        self.auto_prior_update = auto_prior_update #Don't use
        if (hyperparameter_optimization is not None or
            hyperparameter_optimization is not False):
            self.a_old = prior_a + 0.0000001 #Used for gradient computation
            self.b_old = prior_b+ 0.0000001 #Used for gradient computation
            self.gradient_old = 0.0         #Used for gradient computation
            self.a_list, self.b_list = [],[]
        self.hyperparameter_optimization = hyperparameter_optimization
        self.non_spd_alerts = non_spd_alerts #if cov mat not spd and forced
                                             #to be, this alerts you.

        """STEP 3: Get informations about the model we set up"""
        self.has_lags = True #needed inside detector
        self.generalized_bayes_rld = "kullback_leibler" #changed from inside detector init
        self.alpha_rld_learning = False
        self.alpha_rld = None #changed from inside detector init
        self.S1, self.S2 = S1, S2

        """STEP 3.1: If we are on a regular grid with strong param binding"""
        self.restriction_sequence = restriction_sequence
        self.nbh_sequence = nbh_sequence
        self.padding = padding

        """STEP 3.2: If we are on general neighbourhood structures"""
        self.general_nbh_sequence = general_nbh_sequence
        self.general_nbh_restriction_sequence = general_nbh_restriction_sequence
        self.general_nbh_coupling = general_nbh_coupling
        self.intercept_grouping = intercept_grouping

        """STEP 3.3: Check if we use regular grid + strong param binding or
        the more general framework"""
        if ((not self.restriction_sequence is None) and
            (not self.nbh_sequence is None) and
            (not self.padding is None)):
            self.regular_grid = True
        elif ((not self.general_nbh_sequence is None) and
              (not self.general_nbh_restriction_sequence is None) and
              (not self.general_nbh_coupling is None)):
            self.regular_grid = False
        elif (( self.restriction_sequence is None) and
             ( self.nbh_sequence is None) and
             ( self.general_nbh_sequence is None) and
             ( self.general_nbh_restriction_sequence is None)):
            #In this case, we have only constant terms
            self.regular_grid = False
            self.has_lags = False
            self.lag_length = 0 #unclear if it is arrived at automatically
            self.general_nbh_coupling = None
        else:
            """Neither specification is complete, so end the execution here"""
            raise SystemExit("Your neighbourhood specifications " +
                "are incomplete: At least one of " +
                "restriction_sequence, nbh_sequence, padding is None; " +
                "or at least one of " +
                "general_nbh_sequence, general_nbh_restriction_sequence ,"+
                " general_nbh_coupling is None")

        """STEP 3.4: If we have any exogeneous/additional variables"""
        if exo_selection is None or exo_selection == []:
            self.exo_bool = False
            exo_selection = []
            self.exo_selection = []
        else:
            self.exo_bool = True
            self.exo_selection = exo_selection

        """STEP 4: Convert the neighbourhood into a sequence of strings
                    for the endogeneous variables"""

        """STEP 4.1: Get the codes for the intercept design"""
        self.get_intercept_codes()

        """STEP 4.2: Get endogeneous regressor codes (self.endo_vars), lag
        length (self.lag_length), and information about empty nbhs
        (self.empty_nbhs, self.sum_empty_nbhs_per_lag)"""
        #DEBUG: Not needed under constant fct. Simply set self.endo_var=[].
        #       do this inside fct.
        self.get_endo_vars()

        """STEP 4.3: Get exogeneous regressor codes (self.exo_vars)"""
        self.exo_vars = [self.intercept_codes + exo_selection]

        """STEP 4.4: Get all regressor codes"""
        self.all_vars = list(self.exo_vars) + list(self.endo_vars)
        self.all_vars = sum(self.all_vars, [])


        """STEP 5: Define quantities relating to the regressors:
                    the sequences of variables, the counts of variables,
                    the lag-structure, extraction list for updating"""

        """STEP 5.1: Get the number of each type of variable"""
        self.num_exo_regressors = len(sum(self.exo_vars, []))
        self.num_endo_regressors = len(sum(self.endo_vars, []))
        self.num_regressors = (self.num_endo_regressors +
                               self.num_exo_regressors)

        """STEP 5.2: Get the lag structure such that lag_counts stores the
                     #exo_vars at position 0,and stores at position l the count
                     {#exo_vars + sum(#endo_vars: lag <= l) inside
                     self.lag_counts"""
        #DEBUG: For constant function, this should be only the first line of
        #       the function
        self.get_lag_counts()

        """STEP 6: Get the extraction vector and the insertion position. Note
        that the result will be a list of form [1,1,1,0,0,1,1,1,1,0,0,0], which
        means that the first 3 endogeneous variables will be kept, the next
        two will be discarded, the next 4 will be kept, and the next 3 disc."""

        """STEP 6.1: Store in each entry l the number of endogeneous regressors
        for lag l"""
        #For constant fct, this should just return an empty list (if se set lag_length = 0)
        endo_regressors_per_lag = self.get_endo_regressors_per_lag()

        """STEP 6.2: You can now get a list that tells you for given X_t-1
        which columns need copying to X_t. You never copy exogeneous variables.
        Also, the first lag for X_t will be new, so one can copy at most
        lag_length -1 neighbourhoods from X_t-1 to X_t. Store this list as
        self.extraction_list, and the position where you start extracting
        as self.insertion_position with the function below"""
        #DEBUG: This should still work and return an empty extraction list as
        #       well as an insertion position = p
        self.get_extraction_list(endo_regressors_per_lag)

        """STEP 7: create the objects we need to trace through time"""
        self.XX, self.YX, self.model_log_evidence = None, None, -np.inf
        """NOTE: The quantities below will be re-initialized in the
        initialization function, but have to be instantated here due to how
        the enclosing Detector object calls model_and_run_length_distr"""
        self.retained_run_lengths = np.array([0,0])
        self.joint_log_probabilities = 1

        #DEBUG: Should not really be here (but insted in initialization)
        self.log_alpha_derivatives_joint_probabilities = None #np.ones(3)
        self.log_alpha_derivatives_joint_probabilities_sign = None #np.ones(3)


        """STEP 8: Rectify prior_beta_mean and prior_beta_var if needed.
                    Give a warning about this, too!"""

        """STEP 8.1: prior mean beta is not supplied or does not correspond
        to the right dimensions: Check if a scale is
        supplied. If not, automatically set the scale to 0.0, ensuring
        that beta_0 = 0."""
        if (self.prior_mean_beta is None or
            self.num_regressors != np.size(self.prior_mean_beta)):
            if prior_mean_scale is None:
                prior_mean_scale = 0.0
            self.prior_mean_beta = (prior_mean_scale*
                                    np.ones(self.num_regressors))

        """STEP 8.2: prior var beta is not supplied or does not correspond
        to the right dimensions: Check if a scale is
        supplied. If not, automatically set the scale to 100.0, ensuring
        that V_0 = 100*I."""
        if (self.prior_var_beta is None or
            self.num_regressors != prior_var_beta.shape[0] or
            self.num_regressors != prior_var_beta.shape[1]):
            if prior_var_scale is None:
                prior_var_scale = 100.0
            self.prior_var_beta = (prior_var_scale*
                                    np.identity(self.num_regressors))


    def get_intercept_codes(self):
        """Only called in __init__: Gets the intercept regressor codes"""
        if (self.intercept_grouping is None or
            self.intercept_grouping == np.array([])):
            self.intercept_codes = ["intercept"]
        else:
            self.num_intercept_groups = self.intercept_grouping.shape[0]
            self.intercept_codes = []
            for g in range(0, self.num_intercept_groups):
                self.intercept_codes.append(("intercept_group_" + str(g)))

    def get_endo_vars(self):
        """Only called in __init__: Gets self.endo_vars, self.lag_length,
        self.empty_nbhs, self.sum_empty_nbhs_per_lag in different ways,
        depending on how your nbh structure is set up."""

        endo_vars = []

        """"STEP A: If you are on regular grid with strong parameter binding"""
        if self.regular_grid:
            self.lag_length = np.size(self.nbh_sequence)
            for lag in range(0,int(self.lag_length)):
                restriction = self.restriction_sequence[lag]
                nbh = self.nbh_sequence[lag]
                if restriction == 0:
                    if nbh == 0:
                        endo_vars.append(["center"])
                    elif nbh == 4:
                        endo_vars.append([ "center","top", "left", "right",
                                "bottom"])
                    elif nbh == 8:
                        endo_vars.append(["center",
                                "top", "left", "right", "bottom",
                                "topleft", "topright","bottomleft", "bottomright"])
                elif restriction == 4:
                    if nbh == 0:
                        endo_vars.append(["center"])
                        print("Warning: Restriction sequence")
                        print("contained 4, nbh sequence a 1-nbh")
                        print("at the same position.\n")
                    elif nbh == 4:
                        endo_vars.append(["center", "4_inner_nbh_res"])
                    elif nbh == 8:
                        endo_vars.append(["center", "4_outer_nbh_res",
                                 "4_inner_nbh_res"])
                elif restriction == 8:
                    if nbh == 0:
                        endo_vars.append(["center"])
                        print("Warning: Restriction sequence")
                        print("contained 8, nbh sequence a 1-nbh")
                        print("at the same position.\n")
                    elif nbh == 4:
                        endo_vars.append(["center", "4_inner_nbh_res"])
                        print("Warning: Restriction sequence")
                        print("contained 8, nbh sequence a 4-nbh")
                        print("at the same position.\n")
                    elif nbh == 8:
                        endo_vars.append(["center", "8_nbh_res"])
                        print("Warning: Restriction = 8, which is not fully implemented")
        elif self.general_nbh_coupling == "weak coupling":
            """STEP B: If we use the general nbh sequence formulation with
            weak coupling (i.e. nbh-specific, but not across space).
            Recall that the structure is as follows:
            general_nbh_sequence = [[[4,5,6],[7,8],[9]], [[2,3,4],[5],[7]],...]
            general_nbh_restriction_sequence = [[0,1,2],[0,1,2],[0,1],[2]].
            Here, lag_length = 4, general_nbh_restriction_sequence[lag] = g(l),
            where g(l) gives you the index of the nbh generating the regressors
            at lag length l for s, i.e. N_p(l)(s)

            We want to get strings of form

            general_nbh_<lag>_<nbh_index>_<loc>,

            where <lag> gives you the index in general_nbh_restriction_seq that
            you need, say <lag> = 0, i.e. we care about [0,1,2]. Given this
            index list, <nbh_index> then tells us which of the indices (and
            thus neighbourhoods) we care about, i.e. nbh_index = 0 would mean
            we care about [0,1,2][0] = [0]. Lastly, the <loc> tells us which
            index on the lattice we care about, allowing us to retrieve
            general_nbh_sequence[<loc>][general_nbh_restriction_seq[<lag>][<nbh_index>]]
            as the indices of the nbh with <nbh_index> corresponding to
            <loc>'s neighbourhood at lag <lag>+1
            """
            self.lag_length = int(len(self.general_nbh_restriction_sequence))
            self.empty_nbhs = [] #helps us to sort out the extraction list later
            self.sum_empty_nbhs_per_lag = np.zeros(self.lag_length)

            """loop I: Go over all lag lengths, since the nbhs and their
            restrictions will differ between lag lengths"""
            for lag in range(0, int(self.lag_length)):
                new_endo_vars_entry = []

                """Loop II: over all locations to fill self.endo_vars with the
                correct endogeneous variables for each location and lag"""
                for location in range(0, self.S1*self.S2):
                    #DEBUG: This marks the center for each location separately
                    #       make sure that this does not cause problems for how
                    #       we find the lag (e.g., by counting # of "center"s)
                    new_endo_vars_entry.append("general_nbh_" +
                          str(lag) + "_" + "center" + "_" +
                          str(location))
                    self.empty_nbhs.append(False)
                    relevant_nbh_indices = self.general_nbh_restriction_sequence[lag]

                    """Loop III: Over all relevant nbh indices for this
                    location at the current lag. This makes sure that our
                    endo var codes are specific to lag, location, and the
                    neighbour whose values are used."""
                    for nbh_index in relevant_nbh_indices:

                        """Only add the nbh if it is non-empty. If it is
                        empty, nbh_index will have boolean value False."""
                        if nbh_index:
                            """Again, we only want to create the regressor code
                            if the list is non-empty. If it is empty, we
                            instead note so inside self.empty_nbhs and
                            self.sum_empty_nbhs_per_lag in the 'else' cond."""
                            if self.general_nbh_sequence[location][nbh_index]:
                                new_endo_vars_entry.append("general_nbh_" +
                                  str(lag) + "_" + str(nbh_index) + "_" +
                                  str(location))
                                self.empty_nbhs.append(False)
                            else:
                                """Mark which neighbourhoods were left out because
                                they were empty. Needed for extraction_list and
                                lag_counts"""
                                self.empty_nbhs.append(True)
                                self.sum_empty_nbhs_per_lag[lag] += 1
                    """Inside Loop II: For this location and lag, add the
                    required endogeneous variables into the collection of all
                    of them"""
                    endo_vars.append(new_endo_vars_entry)
                    new_endo_vars_entry = []

        elif self.general_nbh_coupling == "strong coupling":
            """STEP C: In this case, we have the same input as for weak
            coupling, but a different interpretation. In particular, we couple
            the effects over different spatial locations. Accordingly, we save

            general_nbh_<lag>_<nbh_index> only.

            Then, in the extractors, we loop over <loc> to retrieve the
            regressors in a single column as

            regressor(<lag>, <nbh_index>)[<loc>] = sum over all measurements
                at time t - <lag> for nbh given by
                gen_nbh_seq[<loc>][gen_nbh_res_seq[<lag>][<nbh]].
            """
            self.lag_length = int(len(self.general_nbh_restriction_sequence))

            """Loop I: Over the lags"""
            for lag in range(0, int(self.lag_length)):
                new_endo_vars_entry = ["general_nbh_" + str(lag) + "_center"]
                relevant_nbh_indices = self.general_nbh_restriction_sequence[lag]
                """Loop II: Over the relevant nbhs. Notice that unlike for the
                weak coupling, we only have 2 (rather than 3) loops, as the
                locations do not require a separate loop for strong coupling"""
                for nbh_index in relevant_nbh_indices:
                    new_endo_vars_entry.append("general_nbh_" +
                              str(lag) + "_" + str(nbh_index))
                endo_vars.append(new_endo_vars_entry)

        elif (self.general_nbh_coupling is None) and (not self.regular_grid):
            """In this case, we only fit constants!|"""
            endo_vars = []
            self.lag_length = 0

        """Last step: Declare endo_vars as the new attribute of the object"""
        self.endo_vars = endo_vars


    def get_lag_counts(self):
        """Only called in __init__: Gets self.lag_counts"""
        self.lag_counts = [self.num_exo_regressors]
        last_count = self.num_exo_regressors
        if self.regular_grid:
            """STEP 1.A: If 0/4/8 nbhs used: Can be done via endo vars"""
            for entry in self.endo_vars:
                self.lag_counts.append(last_count + len(entry) + 1)
                last_count = last_count + len(entry) + 1 #update
        elif self.general_nbh_coupling == "strong coupling":
            """STEP 1.B: Similar to weak coupling, except you don't need to
            multiply by the numbers of locations"""
            for lag in range(0, self.lag_length):
                self.lag_counts.append(last_count + (
                    (len(self.general_nbh_restriction_sequence[lag]) + 1)))
                last_count = last_count + (
                    (len(self.general_nbh_restriction_sequence[lag]) + 1))
        elif self.general_nbh_coupling == "weak coupling":
            """STEP 1.C: If general nbhs, we need more care"""
            """each gen_res corresponds to a lag and gives you a set at
            position l, e.g. [0,1,2] at position 0, telling you that at the
            first lag, the neighbourhoods used are 0,1,2. Thus, at the first
            lag, each location has 3 regressors corresponding to the first
            three nbhs for that location in general_nbh_sequence PLUS the
            autoregressive term, which is always incorporated but not repre-
            sented in any regressor code.
            I.e., we need [len([0,1,2]) + 1]*S1*S2 to get the #endogeneous
            variables at lag 1. Generally, we thus need
            [len(gen_nbh_res_seq[l]) + 1]*S1*S2"""
            for lag in range(0, self.lag_length):
                self.lag_counts.append(last_count + (
                    (len(self.general_nbh_restriction_sequence[lag]) + 1)
                    *self.S1*self.S2) - self.sum_empty_nbhs_per_lag[lag])
                last_count = last_count + ( - self.sum_empty_nbhs_per_lag[lag] +
                    (len(self.general_nbh_restriction_sequence[lag]) + 1)
                    *self.S1*self.S2)
        elif (not self.regular_grid) and self.general_nbh_coupling is None:
            """STEP 1.D: We only fit a constant, so self.lag_counts remains
            unchanged. self.lag_counts will be None"""



    def get_endo_regressors_per_lag(self):
        """Returns as output the endogeneous regressors per lag"""
        if self.regular_grid:
            """STEP 1A: If we have the 4-nbh structure"""
            endo_regressors_per_lag = []
            for l in range(0, self.lag_length):
                res = self.restriction_sequence[l]
                nbh = self.nbh_sequence[l]
                if res == 0:
                    endo_regressors_per_lag.append(int(nbh) + 1)
                elif res == 4:
                    endo_regressors_per_lag.append(int(nbh*0.25) + 1)
        elif self.general_nbh_coupling is not None:
            """STEP 1B: If we have a general nbh structure, we get
            endo_regressors_per_lag differently. In particular, just look at
            the self.endo_vars object."""
            endo_regressors_per_lag = []
            for l in range(0, self.lag_length):
                endo_regressors_per_lag.append(int(len(self.endo_vars[l])))
        else:
            """STEP 1C: If we only fit a constant"""
            endo_regressors_per_lag = []

        """STEP 2: Return the result"""
        return endo_regressors_per_lag

    def get_extraction_list(self, endo_regressors_per_lag):
        """Gets self.extraction_list and self.insertion position"""

        """"STEP 1: Make sure we don't want to copy exogeneous regressors"""
        self.extraction_list = [False]*(self.num_exo_regressors)

        if self.regular_grid:
            """STEP 1A: IF we have 0/4/8 nbhs """
            for i in range(0,self.lag_length-1):
                self.extraction_list = (self.extraction_list
                    + [True]*endo_regressors_per_lag[i+1]
                    + [False]*int(endo_regressors_per_lag[i] -
                                          endo_regressors_per_lag[i+1]))
            """STEP 2A: The last lag of X_t-1 will 'slide out' of sight, so it
                       definitely is not needed for X_t anymore."""
            self.extraction_list += ([False]*
                                endo_regressors_per_lag[self.lag_length-1])

        elif self.general_nbh_coupling == "weak coupling":
            """STEP 1B: IF we have general nbhs"""
            per_location = []
            for lag in range(0, self.lag_length-1):
                num_retained = (1 + len(np.intersect1d(
                            self.general_nbh_restriction_sequence[lag],
                            self.general_nbh_restriction_sequence[lag+1])))
                num_discarded = ( -num_retained + 1 +
                        len(self.general_nbh_restriction_sequence[lag]))
                per_location += ([True]* num_retained +
                                 [False] * num_discarded)
            """STEP 2B: The last lag of X_t-1 will 'slide out' of sight, so it
                       definitely is not needed for X_t anymore."""
            total_num_last_lag = 1+ len(
                    self.general_nbh_restriction_sequence[self.lag_length-1])
            per_location += ([False]* total_num_last_lag)

            """STEP 3B: Use that we have the same structure all across the
            lattice, and simply multiply each entry of 'per_location' by the
            number of lattice elements"""
            self.extraction_list += sum(
                [self.S1*self.S2*[e] for e in per_location],[])
            self.extraction_list[self.num_exo_regressors:] = np.array(
                self.extraction_list)[np.where(np.array(
                        self.empty_nbhs) == False)].tolist()

        elif self.general_nbh_coupling == "strong coupling":
            """STEP 1C: IF we have general nbhs"""
            per_location = []
            for lag in range(0, self.lag_length-1):
                num_retained = (1 + len(np.intersect1d(
                            self.general_nbh_restriction_sequence[lag],
                            self.general_nbh_restriction_sequence[lag+1])))
                num_discarded = ( -num_retained + 1 +
                        len(self.general_nbh_restriction_sequence[lag]))
                per_location += ([True]* num_retained +
                                 [False] * num_discarded)
            """STEP 2C: The last lag of X_t-1 will 'slide out' of sight, so it
                       definitely is not needed for X_t anymore."""
            total_num_last_lag = 1+ len(
                    self.general_nbh_restriction_sequence[self.lag_length-1])
            per_location += ([False]* total_num_last_lag)

            """STEP 3C: Use that we have the same structure all across the
            lattice, and simply multiply each entry of 'per_location' by the
            number of lattice elements"""
            self.extraction_list += per_location

        elif self.general_nbh_coupling is None and not self.regular_grid:
            """We have constant function and don't need to change anything"""

        """STEP 4: In order to copy entries of X_t-1 to X_t, you need to know
                     the position of X_t at which you should insert. (This does
                     only affect the endogeneous part of the regressors)"""
        self.insertion_position = - sum(self.extraction_list)


    def reinstantiate(self, a = None, b = None):
        """Return a new BVARNIG-model that contains all the same attributes as
        this BVARNIG model. In some sense, it is an 'emptied' version of the
        same model. Used inside HyperparameterOptimization, if BVARNIGs
        Detector is run for hyperparameter optimization"""

        """STEP 1: Get all the characteristics of this model"""
        prior_a, prior_b, S1, S2 = self.a, self.b, self.S1, self.S2
        prior_mean_beta,prior_var_beta=self.prior_mean_beta,self.prior_var_beta
        nbh_sequence = self.nbh_sequence
        restriction_sequence = self.restriction_sequence
        intercept_grouping = self.intercept_grouping
        general_nbh_sequence = self.general_nbh_sequence
        general_nbh_restriction_sequence = self.general_nbh_restriction_sequence
        nbh_sequence_exo = self.nbh_sequence_exo
        exo_selection = self.exo_selection
        padding = self.padding
        auto_prior_update = self.auto_prior_update
        hyperparameter_optimization = self.hyperparameter_optimization
        general_nbh_coupling = self.general_nbh_coupling
        non_spd_alerts = self.non_spd_alerts

        """STEP 2: Check whether you have a new prior already"""
        if a is None:
            a = prior_a
        if b is None:
            b = prior_b

        """STEP 2: Use the characteristics to clone the model"""
        clone_model = BVARNIG(prior_a = a, prior_b = b, S1=S1, S2=S2,
              prior_mean_beta=prior_mean_beta,
              prior_var_beta =prior_var_beta,
              prior_mean_scale=None, prior_var_scale=None,
              nbh_sequence=nbh_sequence,
              restriction_sequence=restriction_sequence,
              intercept_grouping=intercept_grouping,
              general_nbh_sequence=general_nbh_sequence,
              general_nbh_restriction_sequence=general_nbh_restriction_sequence,
              nbh_sequence_exo=nbh_sequence_exo, exo_selection=exo_selection,
              padding=padding, auto_prior_update=auto_prior_update,
              hyperparameter_optimization=hyperparameter_optimization,
              general_nbh_coupling=general_nbh_coupling,
              non_spd_alerts=non_spd_alerts)

        """STEP 3: Return the cloned model"""
        return clone_model


    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    """           FIRST OBSERVATION INITIALIZATION FUNCTIONS            """
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    #NOTE: We need to pass X_endo with one more entry into this function,
    #       namely for y_2!
    def initialization(self, X_endo, X_exo, Y_2, X_exo_2, cp_model, model_prior,
            padding_columns_computeXX = None, padding_column_get_x_new = None):
        """Initialize the model (i.e. t=1) with some inputs from the
        containing Detector object. The padding_column arguments are only
        needed for the demo Csurf object. This is different from object
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


        print("Initializing BVAR object")

        """STEP 1: Take the data-stream that was partitioned appropriately
        inside the Detector object and reshape/rename it for further processing
        Y1 = Y_t, Y2 = Y_{t+1}, X1_endo = Y_1:t-1, with t = L-1."""
        Y1 = X_endo[-1,:].flatten()
        Y2 = Y_2.flatten()
        if self.has_lags:
            X1_endo = X_endo[:self.lag_length,:].reshape(self.lag_length,
                        self.S1, self.S2)
        else:
            X1_endo = None

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

        """STEP 2.2: Cross-product quantities for time point t and run-length
        r, i.e. dimension does depend on how many run-lengths we retain. Unlike
        quantities only stored for the current time, the quantities below
        incorporate the prior beliefs.

        Quantities will hold;
        XX_rt
            At time t, r-th entry holds the cross-product of all regressors
            corresponding to  run-length r_t, i.e. you sum over the last r_t
            cross-products XX. Additionally, XX_rt also holds the prior
            belief inside, so
            XX_rt[r,:,:] = prior_var_beta^-1 + sum_{i = t-r}^t XX(i)
        XY_rt
            At time t, r-th entry holds the cross-product of all regressors
            and observationscorresponding to  run-length r_t, i.e. you sum
            over the last r_t cross-products XY. Additionally, XY_rt also holds
            the prior belief inside, so
            XY_rt[r,:] = prior_var_beta^-1 * prior_beta + sum_{i = t-r}^t XY(i)
        YY_rt
            As the other two, but with YY, and no prior belief occurs, so
            YY_rt[r] = sum_{i = t-r}^t YY(i)
        Q_rt, R_rt
            Unuseable in current version, would hold the QR-decomposition of
            inverse of XX_rt
        """
        self.XX_rt = np.zeros(shape=(2,self.num_regressors, self.num_regressors)) #2 for r=-1 and r=0
        self.XY_rt = np.zeros(shape=(2,self.num_regressors))  #2 for r=-1 and r=0
        self.YY_rt = np.zeros(2)

        #QR ADAPTION
        self.Q_rt = np.zeros(shape=(2,self.num_regressors, self.num_regressors))
        self.R_rt = np.zeros(shape=(2,self.num_regressors, self.num_regressors))


        """STEP 2.3: Inverse-related quantities for time point t and run-length
        r, i.e. dimension again depends on how many run-lengths on retains.
        These are direct functionals of the cross-produts stored above, but
        computed/updated in an efficient rather than brute-force way

        Quantities will hold:
        M_inv_1_rt
            Inverse of XX_rt, updated via Woodbury formula at each time point,
            but at a later time point than M_inv_2_rt. This means within a
            certain time window inside an iteration, we have access to both,
            XX_rt^-1 at t and XX_rt^-1 at time t-1, which is needed for
            efficient updates.
        M_inv_2_rt
            Inverse or XX_rt, updated via Woodbury formula at each time point.
            See above for the relative timing.
        log_det_1_rt
            log determinants of all entries in M_inv_1_rt, computed efficiently
        log_det_2_rt
            log dets of all entries in M_inv_2_rt, computed efficiently
        """
        self.M_inv_1_rt = np.zeros(shape=(2,self.num_regressors,
                                          self.num_regressors))
        self.M_inv_2_rt = np.zeros(shape=(2,self.num_regressors,
                                          self.num_regressors))
        self.log_det_1_rt = np.zeros(2)
        self.log_det_2_rt = np.zeros(2)


        """STEP 2.4: beta-coef related quantities for time point t and run-
        length r, i.e. dimension depends on how many run-lengths one retains

        Quantities will hold:
        beta_rt
            beta_rt[r,:] stores the coefficients beta corresponding to the
            MAP-estimate at time t if one assumes run-length r
        beta_XX_beta_rt
            what it says: beta_rt[r,:] * XX_rt[r,:,:] * beta_rt[r,:] at pos r
            each time point t.
        """
        self.beta_XX_beta_rt = np.zeros(2)
        self.beta_rt = np.zeros(shape=(2,self.num_regressors))


        """STEP 2.5: Retained run lengths, storing which run-lengths you retain
        at time t. Careful with this, as retained_run_lengths[i] = j means that
        the i-th longest run-length you retain is j"""
        self.retained_run_lengths = np.array([0,0])

        """STEP 3: Compute prior- and data-dependent quantities:
        Computation of X_t, X_tp1, X'X,  X'Y, and Y'Y from scratch."""

        """STEP 3.1: Gives X_t, X'X, X'Y, Y'Y"""
        #DEBUG: Unclear if this does what I want for constant case!
        self.compute_X_XX_XY_YY( Y1, X1_endo, X1_exo,
                                padding_columns_computeXX,
                                compute_XY = True)
        """STEP 3.2: Gives X_{t+1}"""
        #DEBUG: Unclear if this does what I want for constant case!
        self.X_tp1 = self.get_x_new(Y2, X_exo_2 ,1,padding_column_get_x_new)


        """STEP 4: Using the results of STEP 3, compute some computationally
        burdensome results, like XX_rt's inverses and prior inv + det"""

        """STEP 4.1: Computation of the prior inverse, which will be needed
        at each iteration to inform the chaingepoint probabilities"""
        self.D_inv = np.linalg.inv(self.prior_var_beta) #not efficient if D diagonal
        _, self.D_inv_log_det = np.linalg.slogdet(self.D_inv)

        #QR ADAPTION
        self.D_inv_Q, self.D_inv_R = np.linalg.qr(self.D_inv)
        self.D_inv_log_det =  np.sum(np.log(np.abs(np.diagonal(self.D_inv_R))))

        """STEP 4.2: Use the prior inverse from STEP 4.1 to get the first
        inverse computation of XX_rt underway"""
        M_inv_1 = np.linalg.inv(self.D_inv + self.XX)
        self.M_inv_1_rt[0,:,:] =  self.M_inv_1_rt[1,:,:] = M_inv_1

        #QR ADAPTION
        Q0, R0 = self.QR_loop(self.D_inv_Q, self.D_inv_R, self.X_t)
        self.Q_rt[0,:,:] = self.Q_rt[1,:,:] = Q0
        self.R_rt[0,:,:] = self.R_rt[1,:,:] = R0


        """STEP 5: Compute the prior contributions/quantities and use them to
        get XX_rt, YY_rt, XY_rt with prior influences for r_t = 0"""

        """STEP 5.1: Get D^-1*beta_prior and beta_prior * D^-1 * beta_prior
        which are needed later in the estimation as the prior contributions"""
        self.D_inv_b0 = np.matmul(self.D_inv, self.prior_mean_beta)
        self.b0_D_inv_b0 = np.inner(self.prior_mean_beta, self.D_inv_b0)

        """STEP 5.2: Get the first two values of X'X_rt and X'Y_rt using
        the result of STEP 6.1.
        NOTE: Since we will only need X'Y for computing beta(r,t),
        we need to work with (D^-1 * beta_0 + X'Y), which is why
        we add D^-1 * beta_0 to X'Y whenever we are at r=0."""
        self.XX_rt[0,:,:] = self.XX_rt[1,:,:] = self.XX + self.D_inv
        self.XY_rt[0,:] = self.XY_rt[1,:] = (self.XY + self.D_inv_b0)
        self.YY_rt[0] = self.YY_rt[1] = self.YY


        """STEP 6: Get the log-determinants by brute force or QR
        NOTE: If using QR, use trace for determinants of Q(r,t)R(r,t)
        for all run-lengths. These are needed in posterior of Y
        They can be obtained as trace of R[r,:,:] because Q is an
        orthogonal matrix, so det(Q) = 1 and as
        det(QR) = det(Q)det(R), it follows det(QR) = det(R)"""

        sign, value = np.linalg.slogdet(self.M_inv_1_rt[0,:,:])
        self.log_det_1_rt[0] = self.log_det_1_rt[1] = (value) #s.p.d. matrices have pos dets

        #QR ADAPTION
        #diag = np.abs(np.diagonal(self.R_rt, axis1=1, axis2=2))
        #self.log_det_1_rt = np.sum(np.log(diag), axis=1)


        """STEP 7: Compute the MAP of beta = MX'Y from scratch, using triangular
        solvers for speedy computation! Also compute beta^T X'X(r,t) beta.
        If QR is used, you also calculate the inverses here."""
        beta = np.matmul(self.M_inv_1_rt[0,:,:],self.XY_rt[0,:])
        self.beta_rt[0,:] = self.beta_rt[1,:] = beta

        #QR ADAPTION
        #beta = linalg.solve_triangular(a = self.R_rt[0,:,:],
        #        b = np.matmul(np.transpose(self.Q_rt[0,:,:]),self.XY_rt[0,:]),
        #        check_finite=False)
        #self.M_inv_1_rt[0,:,:] = self.M_inv_1_rt[1,:,:] = (
        #   linalg.solve_triangular(a=R0, b = np.transpose(Q0),
        #       check_finite=False))

        self.beta_XX_beta_rt[0] = self.beta_XX_beta_rt[1] = (np.inner(np.matmul(
                self.beta_rt[0,:], self.XX_rt[0,:]), self.beta_rt[0,:]))

        """STEP 8: Lastly, update the inverses for one-step-ahead of time, i.e.
        get M_inv_2_rt as well as its log determinant."""

        """STEP 8.1: If we do Woodbury, this is a brute force step involving
        inversion of the small matrix that re-appears later on
        inside 'mvt_log_density' as C_t_inv.

        If we do QR-updates, perform QR update w.r.t. X_tp1 and
        get M_inv + log_det_2. Do NOT update X'X, X'Y, X_t, X_tp1, Y'Y since
        they will be already updated"""

        small_matrix_inv = (
                np.linalg.inv(
                np.identity(self.S1*self.S2) +
                np.matmul((self.X_tp1), np.matmul(
                        self.M_inv_1_rt[0,:,:], np.transpose(self.X_tp1)))) )


        """Brute force determinant calc for small matrix + recursive update for
        determinant of M(r,t). We take -value because log(det(M^-1)) =
        -log(det(M))"""
        sign2, value2 = np.linalg.slogdet(small_matrix_inv)
        self.log_det_2_rt[0] = self.log_det_2_rt[1] = (
                value2 + self.log_det_1_rt[0])

        """Woodbury Update-Inversion formula for M_inv_2, see handwritten notes
        for derivation"""
        M_inv_1_x_X_tp1 = np.matmul(self.M_inv_1_rt[0,:,:],
                                    np.transpose(self.X_tp1))
        self.M_inv_2_rt[0,:,:] = self.M_inv_2_rt[1,:,:] = (
                self.M_inv_1_rt[0,:,:] - np.matmul((M_inv_1_x_X_tp1),
                np.matmul( small_matrix_inv,
                          np.transpose(M_inv_1_x_X_tp1))))

        #QR ADAPTION
        #Q1, R1 = self.QR_loop(self.Q_rt[0,:,:], self.R_rt[0,:,:], self.X_tp1)
        #self.Q_rt[0,:,:] = self.Q_rt[1,:,:] = Q1
        #self.R_rt[0,:,:] = self.R_rt[1,:,:] = R1
        #self.M_inv_2_rt[0,:,:] = self.M_inv_2_rt[1,:,:] = linalg.solve_triangular(
        #        a=R1, b = np.transpose(Q1), check_finite=False)
        #diag = np.abs(np.diagonal(self.R_rt, axis1=1, axis2=2))
        #self.log_det_2_rt = np.sum(np.log(diag), axis=1)

        """STEP 9: Compute the joint log probabilities under your prior by
        computing the predictive and multiplying it with the model prior as
        well as the probability that we have a CP at time 1 vs at a time
        before the first observation was made. Also compute their gradients
        for efficient updating."""

        """STEP 9.1: Get the posterior parameter estimates from your model,
        use them to get the value of your predictive distribution."""
        a_ = self.a + 0.5
        b_ = self.b + 0.5*(self.b0_D_inv_b0 + self.YY - self.beta_XX_beta_rt[0])
        C_0_inv = (a_/b_)*(np.identity(self.S1*self.S2) -
            np.matmul(self.X_t, np.matmul(self.M_inv_1_rt[0,:,:],
            np.transpose(self.X_t))))
        if b_<0:
            log_det = np.nan
        else:
            log_det = ((self.S1*self.S2) * (np.log(b_) - np.log(a_)) +
                       self.D_inv_log_det - self.log_det_1_rt[0])

        """This step ensures that we center the MVT at zero, which makes
        the computations inside mvt_log_density easier"""
        resid = Y1 - np.matmul(self.X_t, self.beta_rt[0,:])

        """For the first observation, the predictive probability and the
        model evidence are equivalent, as the model evidence is computed under
        prior beliefs (captured by a_, b_, C_0_inv) only."""
        self.model_log_evidence = ( np.log(model_prior) +
                BVARNIG.mvt_log_density(resid, C_0_inv, log_det, 2*a_,
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


    def compute_X_XX_XY_YY(self, Y0, X0_endo, X0_exo, padding_columns = None,
                           compute_XY = True):
        """Compute X'X, X'Y, Y'Y, X_t from scratch. Called at initialization.
        Uses the nbh-strings to concatenate the raw data of X0_endo, Y0 (and
        potentially at a later stage X0_exo) into the regressors that we want
        for our model.
        NOTE: compute_XY = False only for BVARNIGDPD models, where there is
              no need to know XY
        """

        """Computation: Loop over both exogeneous and endogeneous variables,
        retrieve their cross-products element-wise. If you have already it from
        product before, just copy the relevant entry in X'X and paste it."""

        #DEBUG: Reshape X0_endo into (lag_length,S1, S2)
        if self.has_lags:
            X0_endo = X0_endo.reshape(self.lag_length, self.S1, self.S2)
        else:
            X0_endo = None

        lag_count1, lag_count2 = 0,0

        """OUTER LOOP: Over all regressors"""
        for i in range(0, self.num_regressors):

            """Since exo vars are stored first in all_vars, this condition
            allows us to see if we need to access exo or endo vars"""
            if (i <= (self.num_exo_regressors - 1)):
                """EXOGENEOUS"""
                #DEBUG: Do I get the intercept from here? I should, since
                #       self.all_vars will still be containing the intercept_codes
                data_vector1 = self.get_exo_regressors(self.all_vars[i], i,
                                                       X0_exo)
            elif self.has_lags:
                """If we need endo vars, make sure that we advance the lag
                length appropriately afterwards"""
                if (i >= self.lag_counts[lag_count1]):
                    lag_count1 = lag_count1 + 1
                """ENDOGENEOUS"""
                """I.e., if we do not pass padding columns, we cannot access
                the None-type object and thus skip the argument"""
                if padding_columns is None:
                    data_vector1 = self.get_endo_regressors(self.all_vars[i],
                                                     lag_count1, X0_endo)
                else:
                    data_vector1 = self.get_endo_regressors(self.all_vars[i],
                                    lag_count1, X0_endo, padding_columns[i,:])

            lag_count2 = 0 #reset lag count

            """INNER LOOP: Over all regressors"""
            for j in range(0, self.num_regressors):

                """This condition ensures that we do not re-compute cross-
                products after having done so before"""
                if (i <= j):
                    if (j <= (self.num_exo_regressors - 1)):
                        """EXOGENEOUS"""
                        data_vector2 = self.get_exo_regressors(
                                self.all_vars[j], j, X0_exo)
                    elif self.has_lags:
                        """If we need endo vars, make sure that we advance the lag
                        length appropriately afterwards"""
                        if (j >= self.lag_counts[lag_count2]):
                            lag_count2 = lag_count2 + 1
                        """ENDOGENEOUS"""
                        if padding_columns is None:
                            data_vector2 = self.get_endo_regressors(
                                self.all_vars[j], lag_count2, X0_endo)
                        else:
                            data_vector2 = self.get_endo_regressors(
                                self.all_vars[j], lag_count2, X0_endo,
                                padding_columns[i,:])


                    """if i == 0, we loop over all j. Use this to compute X'Y
                    as well as X"""
                    if(i == 0):
                        self.X_t[:,j] = data_vector2
                        if compute_XY:
                            self.XY[j] = np.inner(data_vector2, Y0)

                    """Computation: Fill in X'X with dot products!"""
                    prod = np.inner(data_vector1, data_vector2)
                    self.XX[i,j] = prod
                    self.XX[j,i] = prod

        """Lastly, compute Y'Y"""
        self.YY = np.inner(Y0, Y0)


    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    """                  EVALUATE PROBABILITIES/BELIEFS                 """
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    def evaluate_predictive_log_distribution(self, y, t):
        """Returns the log densities of *y* using the predictive posteriors
        for all possible run-lengths r=0,1,...,t-1,>t-1 as currently stored by
        virtue of the sufficient statistics.
        The corresponding density is computed for all run-lengths and
        returned in a np array

        Note: This is called BEFORE update_log_distr, so at time t, the
                quantities one tracks through time (like Q_rt, R_rt, ...) will
                still only hold Q(r,t), R(r, t), ... and so on (rather than
                Q(r+1,t+1), R(r+1,t+1) ... ). Similarly, the regressors
                X_t will actually correspond to time point t-1, so we instead
                use the regressors stored inside X_tp1 = X_t+1 for evaluating
                the pred. density of y.
        """

        """STEP 1: Preliminaries.
            - Get y into vector format,
            - get log_densities as container of log predictive densities
            - get C_t_inv[r,:,:] as the posterior precision for run-length r+1
        """
        y = y.flatten()
        run_length_num = self.retained_run_lengths.shape[0]
        log_densities = -np.inf * np.ones(shape=run_length_num)

        """Note that we store the r0-C_t_inv too, so this quantity is one
        entry longer than all other quantities"""
        self.C_t_inv = np.zeros((run_length_num+1, self.S1*self.S2,
                                 self.S1*self.S2))
        self.predictive_variance_log_det = np.zeros(run_length_num+1)
        self.C_t_inv[0,:,:] = self.C_t_inv_r0
        self.predictive_variance_log_det[0] = (
                self.predictive_variance_r0_log_det)

        """STEP 2: Loop over all retained run-lengths and fill log_densities[r]
        with the predictive log density for run-length r.
        NOTE: We cannot use retained_run_lengths to loop directly, since
                r=t-1 and r>t-1 both have a 0 in there."""
        for r in range(0,run_length_num):

            """STEP 2.1: Get inverse of posterior variance ( = posterior
            precision) using stored quantities & woodbury (see notes)"""
            a_ = self.a + (self.retained_run_lengths[r]+1.0)*0.5
            b_ = (self.b + 0.5*(self.b0_D_inv_b0 + self.YY_rt[r] -
                            self.beta_XX_beta_rt[r]))
            self.C_t_inv[r+1,:,:] = (np.identity(self.S1*self.S2) -
                       np.matmul(self.X_tp1, np.matmul(self.M_inv_2_rt[r,:,:],
                       np.transpose(self.X_tp1))))

            """STEP 2.2: Get the log determinant using the Woodbury Formula and
            applying the determinant lemma afterwards (see notes)
            NOTE: We take the minus in front because we compute the
                   log det of the INVERSE matrix C(r,t)^-1 here, but
                   need that of C(r,t) for call of 'mvt_log_density'"""
            if b_ < 0:
                log_det = np.nan
            else:
                log_det = ((self.S1 * self.S2)*(np.log(b_) - np.log(a_)) +
                           self.log_det_1_rt[r] - self.log_det_2_rt[r])
            self.predictive_variance_log_det[r+1] = log_det

            """STEP 2.3: Evaluate the predictive probability for r_t = r"""
            resid = y - np.matmul(self.X_tp1, self.beta_rt[r,:])
            log_densities[r] = (
                    BVARNIG.mvt_log_density(resid,
                        (a_/b_)*self.C_t_inv[r+1,:,:],
                        log_det, 2*a_, self.non_spd_alerts))

        """STEP 3: return the full log density vector"""
        return log_densities


    def get_log_integrals_power_divergence(self):
        """get integrals for power div in log-form"""
        p = self.S1*self.S2
        run_length_with_0 = np.insert(self.retained_run_lengths.copy() + 1, 0, 0)

        nu_1 = 2* (self.a + (run_length_with_0+1.0)*0.5)
        nu_2 = nu_1 * self.alpha_rld + p* self.alpha_rld + nu_1

        C1 = (1.0 + self.alpha_rld) * (special.gammaln(0.5*(nu_1+p)) -
              special.gammaln(0.5*nu_1))
        C2 = (special.gammaln(0.5*(nu_2+p)) - special.gammaln(0.5*nu_2))

        #DEBUG: Inefficient as brute force, will only be here for preliminary
        #       test version
        #DEBUG: Incorrect! Posterior variance needs a/b factor!
        #_, dets = np.linalg.slogdet(self.C_t_inv[:,:,:])
        #dets = dets * self.alpha


        return (C1 - C2 - nu_1*0.5*p*self.alpha_rld
                - np.pi*0.5*p*self.alpha_rld #dets)
                 -  self.alpha_rld * self.predictive_variance_log_det )

#    def get_prior_integral_power_divergence(self):
#        """get integral for r = 0"""
#        p = self.S1*self.S2
#        nu_1 = 2*(self.a)
#        nu_2 = nu_1 + self.alpha + p*self.alpha + nu_1
#        C1 = (1.0 + self.alpha) * (special.gammaln(0.5*(nu_1+p)) -
#              special.gammaln(0.5*nu_1))
#        C2 = (special.gammaln(0.5*(nu_2+p)) - special.gammaln(0.5*nu_2))
#
#        #DEBUG: Inefficient as brute force, will only be here for preliminary
#        #       test version
#        _, det = np.linalg.slogdet(self.C_t_inv[0,:,:])
#        det = det * self.alpha
#
#        return (C1 - C2 - nu_1*0.5*p*self.alpha - np.pi*0.5*p*self.alpha - det)

    def evaluate_log_prior_predictive(self, y, t):
        """Basically, this does is as 'evaluate_predictive_log_distribution',
        but using only the prior specs of BVARNIG object to get the
        predictive prob. """
        resid = y - np.matmul(self.X_tp1, self.prior_mean_beta)
        self.C_t_inv_r0 = (
                np.identity(self.S1*self.S2) -
                np.matmul(self.X_tp1, np.matmul(self.prior_var_beta,
                       np.transpose(self.X_tp1))))
        _, log_det = np.linalg.slogdet((self.a/self.b)*self.C_t_inv_r0)
        self.predictive_variance_r0_log_det = log_det
        """Ensure that our log density is upper bounded to avoid ugly numerical
        issues. Usually, this minimum has no effect because the density is way
        below 1.0 (i.e., the log density is way below 0.0), so this prevents
        numerical issues when inverting C_t_inv_r0 to dominate the probability.
        The minimum can be removed and only causes issues in extremely
        artificial data exhibiting near-collinearity in the regressors"""
        return min(0.0, BVARNIG.mvt_log_density(resid,
                (self.a/self.b)*self.C_t_inv_r0, log_det, 2*self.a, True))


    #DEBUG: Deprecated/not needed
    def save_NLL_fixed_pars(self, y,t):
        """DEPRECATED. Similar to eval_pred_log_distr, but evaluates normal
        instead to avoid incorporating the parameter-uncertainty into the NLL
        computations. Not used for anything."""

        """STEP 0: Ensure that y has desired format"""
        y = y.flatten()
        run_length_num = self.retained_run_lengths.shape[0]
        log_densities = -np.inf * np.ones(shape=run_length_num)

        """Note: We cannot use retained_run_lengths to loop directly, since
                 r=t-1 and r>t-1 both have a 0 in there."""
        for r in range(0,run_length_num):

            """STEP 1A: Get inverse using stored quantities & woodbury"""
            a_ = self.a + (self.retained_run_lengths[r]+1.0)*0.5
            b_ = (self.b + 0.5*(self.b0_D_inv_b0 + self.YY_rt[r] -
                            self.beta_XX_beta_rt[r]))
            sigma2 = max((b_/(a_+1)), 0.0000001)
            cov_mat = sigma2*self.C_t_inv[r+1,:,:]

            """STEP 1C: Evaluate the predictive probability"""
            resid = y - np.matmul(self.X_tp1, self.beta_rt[r,:])

            """normal evaluation"""
            log_densities[r] = (
                    stats.multivariate_normal.logpdf(resid, cov=cov_mat))

        """STEP 2: return the full log density vector"""
        self.one_step_ahead_predictive_log_probs_fixed_pars = log_densities



    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    """          UPDATE PROBABILITIES/SUFFICIENT STATISTICS             """
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    #DEBUG: For now assume that one KNOWS the exos, might cause trouble
    def update_predictive_distributions(self, y_t, y_tm1, x_exo_t, x_exo_tp1, t,
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
            always: XX_old, XX, X_t, X_tp1, XX_rt, XY, XY_rt, YY, YY_rt,
                    beta_rt, M_inv_1_rt,M_inv_2_rt, log_det_1_rt, log_det_2_rt,
                    retained_run_lengths, beta_XX_beta_rt
            never:  C_t_inv_rt (instead updated in evaluate_predictive...)
            QR:     Q_rt, R_rt

        """

        """STEP 1: Get observations as vectors"""
        y_t, y_tm1 =y_t.flatten(), y_tm1.flatten()


        """STEP 2: Extract the NEW regressor vectors, and do two things:
        (A) Store them in X for the S1*S2 rank-1-updates later
        (B) Immediately compute the new dot products for X'X, and
            copy the old dot products that can be re-used
        (C) Immediately compute the new dot products for X'Y
        (D) Update  X'X(r-1,t-1) => X'X(r,t),
                    X'Y(r-1,t-1) => X'Y(r,t),
                    Y'Y(r-1,t-1) => Y'Y(r,t) using addition
        NOTE: If we want to use exogeneous variables, adapt this function"""

        """STEP 2.1: Store the value of previous iteration"""
        #DEBUG: Not used anywhere anymore it seems!?
        self.XX_old, self.XY_old, self.X_t_old = (self.XX.copy(),
                        self.XY.copy(), self.X_t.copy())
        self.Y_new, self.Y_old = y_t, y_tm1

        """STEP 2.2: Updates X'X, X'Y, Y'Y, XX_rt, XY_rt, YY_rt"""
        self.regressor_cross_product_updates(y_t,  y_tm1, x_exo_t,
                                             t, padding_column_tm1)

        """STEP 2.3: Retrieves the new regressors in X_tp1 and re-assigns the
        old value of X_tp1 to X_t"""
        self.X_tp1 = self.get_x_new(y_t, x_exo_tp1, t, padding_column_t)


        """STEP 3: Update quantities that are direct functions of the data"""

        """STEP 3.1: Extend the run-lengths and add run-length 0, add QR for
        the new run-length r=0, and update beta and beta^T M beta. Also copy
        the entries of log_det_2_rt into log_det_1_rt to update log_det_1_rt.
        PRE-UPDATES, since M_1_inv_rt, log_det_1_rt are only overwritten by
        the old M_2_inv_rt, log_det_2_rt before they in turn are updated.

        UPDATES WB: run-lengths
                    M_1_inv_rt[0,:,:] added, rest of M_1_inv copied
                    beta_rt
                    beta_XX_beta_rt
                    log_det_1_rt[0,:] added, rest of log_det_2_rt copied
        UPDATES QR: not implemented
        """
        #QR ADAPTION
        #self.pre_QR_updates(t)
        self.pre_updates(t)

        """STEP 3.2: Bottleneck: Update your QR decomposition with the
        regressors in X_tp1 from QR(r,t) to QR(r+1,t+1)
        UPDATES, since this is where the heavy computational lifting occurs
        via updating M_2_inv_rt, log_det_2_rt. If in QR, this is where QR
        decomposition is updated (as this corresponds to the heavy lifting)

        UPDATES WB: M_2_inv(r+1,t+1) for all r
                    log_det_2(r+1,t+1) for all r
        UPDATES QR: not implemented
        """
        self.updates(t)

        #QR ADAPTION
        #self.QR_updates(t)
        #self.post_QR_updates(t)


    def regressor_cross_product_updates(self, y_t, y_tm1, x_exo, t,
                                        padding_column=None,
                                        rt_updates = True):
        """Get the new regressors, i.e. transform the shape of X and X'X in
        accordance with your new observations at time t. Also update X'Y
        and Y'Y, since it is not much more work once the other stuff is
        in place. Also immediately updates XX_rt, XY_rt, YY_rt.

        The argument *padding_column* is only needed for the demo
        object BVAR_NIG_CSurf for the column that is next to the C Surface."""

        """STEP 1: Copy all old regressors from  X and X'X_t into new X and
        X'X and shift them within the same data structure"""

        """STEP 1.1: Shift the ENDO regressors that are already in X, X'X,
                     provided that there is something to shift (i.e. provided
                     that the lag length is at least 2)"""
        if self.has_lags and self.lag_length > 1:
            self.X_t[:,self.insertion_position:] = (
                self.X_t[:,self.extraction_list])
            self.XX[self.insertion_position:,self.insertion_position:] = (
                self.XX[self.extraction_list,:][:,self.extraction_list])

        """STEP 2: For each variable x_i that newly arrives per time step,
                   put it into X and compute relevant cross-prods X'X & X'Y"""
        i = 0
        #DEBUG: Not correct anymore for weak coupling!
        if (not (self.restriction_sequence is None) or
            self.general_nbh_coupling == "strong coupling"):
            num_new_vars = len(self.endo_vars[0]) + self.num_exo_regressors
            new_vars = sum(self.exo_vars,[]) + self.endo_vars[0]
        elif self.general_nbh_coupling == "weak coupling":
            new_vars = (sum(self.exo_vars,[]) +
                        sum(self.endo_vars[:self.S1*self.S2], []))
            num_new_vars = int(len(new_vars))
        elif self.general_nbh_coupling is None and not self.regular_grid:
            """only constants"""
            new_vars = sum(self.exo_vars,[])
            num_new_vars = int(len(new_vars))


        """NEW VARIABLE LOOP I"""
        for regressor_code in new_vars: #sum(self.exo_vars,[]) + self.endo_vars[0]:

            """STEP 2.1: Retrieve the values of x_i"""
            if i <= self.num_exo_regressors - 1:
                x_i = self.get_exo_regressors(regressor_code, i, x_exo)
            elif self.has_lags:
                x_i = self.get_endo_regressors(regressor_code, 1,
                                            y_tm1.reshape(1, self.S1, self.S2),
                                            padding_column)

            """STEP 2.2: Store x_i inside X"""
            self.X_t[:,i] = x_i


            """STEP 2.3: Compute the cross-products x_i^Tx_j for all other
                         new variables x_j s.t. j>=i as well as for all old
                         x_j that we retain, and store in X'X_t"""

            """NEW VARIABLE LOOP II"""
            for j in range(0, num_new_vars):
                if (i <= j):
                    if (j <= self.num_exo_regressors-1):
                        x_j = self.get_exo_regressors(self.all_vars[j],
                                                      j, x_exo)
                    elif self.has_lags:
                        x_j = self.get_endo_regressors(self.all_vars[j],
                                        1, y_tm1.reshape(1,self.S1, self.S2),
                                        padding_column)
                    self.XX[i,j] = self.XX[j,i] = np.inner(x_i, x_j)


                """STEP 2.4: Since for i=0, we will retrieve all new regressors
                             into x_j, use this to directly fill in the new
                             cross-products between old regressors in X and
                             the new regressors x_j"""

                """OLD VARIABLE LOOP"""
                if i == 0 and self.has_lags:
                    for k in range(num_new_vars, self.num_regressors):
                        x_k = self.X_t[:,k]
                        self.XX[k,j] = self.XX[j,k] = np.inner(x_j, x_k)

            """STEP 2.5: Advance the counter"""
            i = i+1

        """Get XX for only constants (like in OLD VARIABLE LOOP)"""
        #DEBUG: Unclear if this works
        if not self.has_lags:
            self.XX = np.identity(self.num_regressors)


        """STEP 3: Add X'X [X'Y, Y'Y] to X'X(r-1,t-1) [X'Y(r-1,t-1),
                    Y'Y(r-1,t-1)]to update to X'X(r,t) [X'Y(r,t), Y'Y(r,t)]"""

        """STEP 3.1: Add X'X [X'Y, Y'Y] to all the t+1 possible run-lenghts"""
        self.YY = np.inner(y_t, y_t)
        self.XY = np.matmul(np.transpose(self.X_t), y_t)

        """Note: Also update the sums of previous X'X quantities, provided that
        you are in a BVARNIG model. In a BVARNIGDPD model, this will not
        happen as XX_rt, XY_rt, YY_rt are not traced through time"""
        if rt_updates:
            self.XX_rt = self.XX_rt + self.XX
            self.XY_rt = self.XY_rt + self.XY
            self.YY_rt = self.YY_rt + self.YY

            """STEP 3.2: Insert X'X [X'Y, Y'Y] at position r=0 of XX_rt
                         [XY_rt, YY_rt]. Note: For X'Y, we need to add the prior
                         influence encoded in D^-1 * beta_0. Since we always add
                         this at r=0, we can ignore it when we add X'Y to the
                         longer run-lengths in STEP 3.1"""
            self.XX_rt = np.insert(self.XX_rt, 0, (self.XX + self.D_inv), axis = 0)
            self.XY_rt = np.insert(self.XY_rt, 0, (self.XY + self.D_inv_b0), axis = 0)
            self.YY_rt = np.insert(self.YY_rt, 0, self.YY, axis = 0)


    def get_x_new(self,y_t, x_exo_tp1, t, padding_column=None):
        """STEP 1: Shift the ENDO regressors that are already in X, X'X,
                     provided that there is something to shift (i.e. provided
                     that the lag length is at least 2)"""
        if self.has_lags and self.lag_length > 1:
            x_new = np.zeros((self.S1*self.S2, self.num_regressors))
            x_new[:,self.insertion_position:] = (
                    self.X_t[:,self.extraction_list].copy())
        else:
            x_new = np.zeros((self.S1*self.S2, self.num_regressors))

        """STEP 2: For each variable x_i that newly arrives per time step,
                   put it into X and compute relevant cross-prods X'X & X'Y"""
        i = 0

        """NEW VARIABLE LOOP I: Only over exogeneous variables and first lag"""
        #DEBUG: This is a list of lists! We want a list of codes!
        if self.has_lags:
            all_codes = sum(self.exo_vars,[]) + self.endo_vars[0]
        else:
            all_codes = sum(self.exo_vars,[]) #endo_vars will be empty, so its
                                                #first entry doesn't exist!
        for regressor_code in all_codes:
            """STEP 2.1: Retrieve the values of x_i"""
            if i <= self.num_exo_regressors - 1:
                x_i = self.get_exo_regressors(regressor_code, i, x_exo_tp1)
            elif self.has_lags:
                """Note: y is treated as 3-dim array in get_endo_regressors"""
                x_i = self.get_endo_regressors(regressor_code, 1,
                                               y_t.reshape(1,self.S1, self.S2),
                                               padding_column)

            """STEP 2.2: Store x_i inside X"""
            x_new[:,i] = x_i
            i = i+1

        return x_new


    def pre_updates(self, t):
        """Updates      retained_run_lengths,
                        M_1_inv_rt, log_det_1_rt,
                        beta_rt, beta_XX_beta_rt"""

        """STEP 1: extend the retained run lengths"""
        self.retained_run_lengths =  self.retained_run_lengths + 1
        self.retained_run_lengths = np.insert(self.retained_run_lengths, 0, 0)

        """STEP 2: Add the new M inverse"""
        new_M_inv = np.linalg.inv(self.D_inv + self.XX)
        self.M_inv_1_rt = np.insert(self.M_inv_2_rt.copy(), 0, new_M_inv, axis=0)

        """STEP 3: update the beta estimates and beta * XX * beta"""
        self.compute_betas(t)

        """STEP 4: update the log determinant 1 and M_inv_1_rt. Again, take
        -new_log_det because we take the log-det of the inverse of the matrix
        whose log det we wanna store."""

        sign, new_log_det = np.linalg.slogdet(new_M_inv)
        self.log_det_1_rt = np.insert(self.log_det_2_rt.copy(), 0, new_log_det)

        #QR ADAPTION
        #"""add the QR for r=0"""
        #newQ, newR = self.QR_loop( self.D_inv_Q, self.D_inv_R, self.X_t)
        #self.Q_rt = np.insert(self.Q_rt, 0, newQ, axis=0)
        #self.R_rt = np.insert(self.R_rt, 0, newR, axis=0)
        #"""get new log det and new M_inv"""
        #new_log_det = np.sum(np.log(np.abs(np.diagonal(newR))))
        #self.log_det_1_rt = np.insert(self.log_det_2_rt.copy(), 0, -new_log_det)
        #
        #new_M_inv = linalg.solve_triangular(a=newR, b = np.transpose(newQ),
        #                                    check_finite=False)
        #self.M_inv_1_rt = np.insert(self.M_inv_2_rt.copy(), 0, new_M_inv,
        #                            axis=0)
        #DEBUG: Should be the same as computing
        #       np.sum(np.log(np.abs(np.diagonal(self.R_rt, axis1=1, axis2=2))), axis=1)


    def updates(self, t):
        """Updates      M_inv_2_rt, log_det_2_rt
        NOTE: It updates the inverses of X'X(r, t) to that ofX'X(r+1,t+1), so
              we have the inverse for t+1 at time t
        """
        run_length_num = self.retained_run_lengths.shape[0]

        self.M_inv_2_rt = np.zeros((run_length_num, self.num_regressors,
                                    self.num_regressors))
        self.log_det_2_rt = np.zeros(run_length_num)

        for r in range(0,run_length_num):
            """Compute small_matrix for r, and get inverse + det"""
            M_inv_x_X_tp1 = np.matmul(self.M_inv_1_rt[r,:,:], np.transpose(self.X_tp1))
            small_matrix_inv = np.linalg.inv(np.identity(self.S1*self.S2) +
                    np.matmul( np.transpose(M_inv_x_X_tp1),
                              np.transpose(self.X_tp1)))

            """Update M_2_inv"""
            self.M_inv_2_rt[r,:,:] = self.M_inv_1_rt[r,:,:] - np.matmul(
                    (M_inv_x_X_tp1), np.matmul(small_matrix_inv,
                    np.transpose(M_inv_x_X_tp1) ))

            """Update log_det_2"""
            sign, value = np.linalg.slogdet(small_matrix_inv)
            self.log_det_2_rt[r] = value + self.log_det_1_rt[r]

            #QR ADAPTION
            #self.Q_rt[r,:,:], self.R_rt[r,:,:] = self.QR_loop(
            #        self.Q_rt[r,:,:], self.R_rt[r,:,:], self.X_tp1)


    def post_QR_updates(self, t):
        #QR ADAPTION
        """After the QR updates have been updated to contain the X_tp1 r
        regressors, obtain the M_inv_rt matrices and update log_det_2"""
        self.M_inv_2_rt = np.insert(self.M_inv_2_rt, 0,
                        np.zeros((self.num_regressors,self.num_regressors) ),
                        axis=0)
        run_length_num = self.retained_run_lengths.shape[0]
        for r in range(0,run_length_num):
            self.M_inv_2_rt[r,:,:] = linalg.solve_triangular(a=self.R_rt[r,:,:],
                b = np.transpose(self.Q_rt[r,:,:]), check_finite=False)
        self.log_det_2_rt = np.sum(np.log(np.abs(np.diagonal(self.R_rt,
                                            axis1=1, axis2=2))), axis=1)


    def compute_betas(self,t):
        """compute beta = MX'Y for all run-lengths using triangular solver:
        Since M^-1*beta = X'Y, and since M^-1 = QR with R a triangular
        matrix and Q^TQ = I, it holds that QR*beta = X'Y, and so
        R*beta = Q^T*X'Y can be solved for beta using a triangular solver
        once Q^T*X'Y has been computed (which is O(n^2)). Thus, since the
        triangular solver itself is O(n^2), we obtain beta in O(n^2)!"""
        run_length_num = self.retained_run_lengths.shape[0]
        self.beta_rt = (
            np.insert(self.beta_rt , 0, np.zeros(self.num_regressors), axis=0))
        self.beta_XX_beta_rt = np.insert(self.beta_XX_beta_rt , 0, 0, axis=0)
        for r in range(0,run_length_num):
            #QR ADAPTION
            #self.beta_rt[r,:] = linalg.solve_triangular(a = self.R_rt[r,:,:],
            #    b = np.matmul(np.transpose(self.Q_rt[r,:,:]), self.XY_rt[r,:]),
            #    check_finite=False)
            self.beta_rt[r,:] = np.matmul(self.M_inv_1_rt[r,:,:], self.XY_rt[r,:])
            self.beta_XX_beta_rt[r] = np.inner(self.beta_rt[r,:],
                np.matmul(self.XX_rt[r,:,:],self.beta_rt[r,:]))

    def QR_loop(self,Q0, R0, X):
        #QR ADAPTION, but also used in initialization (though it need not be)
        """Taking Q0, R0 as starting decomposition, this function loops over
        the elements in X_t until all row vectors have been used for rank-1
        updates. Overwrites Q0,R0 in the process."""
        current_count = end_point = 0
        while (end_point != self.S1*self.S2):
            start_point = current_count*self.num_regressors
            end_point = min((current_count+1)*self.num_regressors,
                            self.S1*self.S2)
            current_range = range(start_point, end_point)
            Q0, R0 = linalg.qr_update( Q0, R0,
                    np.transpose( X[current_range,:]),
                    np.transpose( X[current_range,:]),
                    check_finite=False)
            current_count = current_count + 1
        return Q0, R0


    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    """              GET POSTERIOR PREDICTIVE QUANTITIES                """
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


    #DEBUG: Assumes acess to x_exo(t+1) at time t.
    def get_posterior_expectation(self, t, r_list=None):
        """get the predicted value/expectation from the current posteriors
        at time point t, for all possible run-lengths."""
        post_mean = np.matmul((self.X_tp1),
                              self.beta_rt[:,:,np.newaxis])
        return post_mean

    def get_prior_expectation(self, t):
        """Get the prior value/expectation at time t, for r_t = 0"""
        return ( np.matmul(self.X_tp1, self.prior_mean_beta))


    #DEBUG: Assumes acess to x_exo(t+1) at time t.
    def get_posterior_variance(self, t, r_list=None):
        """get the predicted variance from the current posteriors at
        time point t, for all possible run-lengths."""

        post_var = np.zeros((np.size(self.retained_run_lengths),
                                     self.S1*self.S2, self.S1*self.S2))

        """NOTE: See the derivations in my notes"""
        run_length_num = self.retained_run_lengths.shape[0]
        for r in range(0,run_length_num):

            """Get inverse using stored quantities & woodbury"""
            a_ = self.a + (r+1.0)*0.5
            b_ = (self.b + 0.5*(self.b0_D_inv_b0 + self.YY_rt[r] -
                            self.beta_XX_beta_rt[r]))

            """NOTE: Overflow encountered here when your floats are too big!
                Though this can be avoided by normalizing the data, usually"""
            post_var[r,:,:] = (b_/a_)*(np.identity(self.S1*self.S2) +
                       np.matmul(self.X_tp1, np.matmul(self.M_inv_1_rt[r,:,:],
                                            np.transpose(self.X_tp1))))
        return post_var


    @staticmethod
    def mvt_log_density(y_flat, prec, log_det, df, prior = False, alerts = False):
        """Returns the density of a multivariate t-distributed RV.
        Assumes that the mean is 0. Amounts to the predictive probability for
        a given run-length (and model).

        Here, we have y_flat being the point at which we want to evaluate the
        density, mu its mean, prec the precision matrix, and det the cov matrix'
        determinant. A very helpful reference is the formulation in
        https://www.statlect.com/probability-distributions/
        multivariate-student-t-distribution
        """
        p, nu = y_flat.shape[0], df

        """NOTE: Because it typically is the case that num_regressors > S1*S2,
                 1+(1.0/nu)*np.matmul(np.matmul(y_flat, prec),y_flat)) is
                negative sometimes because prec is not s.p.d.
                This happens for r s.t. (r+1)*S1*S2 < num_regressors, and is
                addressed by forcing prec to be s.p.d."""
        log_term = (1+(1.0/nu)*np.matmul(np.matmul(y_flat, prec),y_flat))

        if( log_term<0 or np.isnan(log_det) ):
            """If there is trouble with your log determinant (i.e. it is nan
            or negative), you will try to fix it. Usually does not happen, but
            needs care if it does. If it does occur, it will typically affect
            the prior (i.e. the predictive for r=0 without observations)"""

            if not prior and p>1:
                """If we don't evaluate the prior predictive (i.e. for r=0),
                immediately try more expensive (but more principled) routine"""
                if alerts:
                    print("covariance estimate not s.p.d. or log_det nan")
                    print("degrees of freedom: ", df)
                """NOTE: Use the (expensive) nearest pd matrix function if not
                         prior, otherwise just add an identity matrix that
                         is large enough"""
                #print(prec)
                try:
                    prec = (NPD.nearestPD(prec) +
                        np.identity(prec.shape[0])*max(df*nu, max(25, p)))
                except (ValueError, np.linalg.LinAlgError) as e: #np.linalg.LinAlgError
                    prec = prec + np.identity(p)*pow(10,5)
                log_term = (1+(1.0/nu)*
                            np.matmul(np.matmul(y_flat, prec),y_flat))

            elif prior and p>1:
                """If we do evaluate the prior predictive (i.e. for r=0),
                try computationally cheap methods first, and then the more
                expensive ones. (Influence of r=0 probability typically
                negligible)"""

                """First try the easy fix (=adding diagonal)"""
                if log_term<0 and p>1:
                    prec = prec + np.identity(prec.shape[0])*nu*df
                    log_term = (1+(1.0/nu)*np.matmul(
                            np.matmul(y_flat, prec),y_flat))
                """If this did not rectify the issue, try the computationally
                more expensive way of fixing it (=PD function)"""
                if log_term<0 and p>1:
                    prec = (NPD.nearestPD(prec) +
                            np.identity(prec.shape[0])*max(df*nu, 25))
                    log_term = (1+(1.0/nu)*
                                np.matmul(np.matmul(y_flat, prec),y_flat))

                """Last safeguard: add diagonal terms until you are spd. I note
                that I have never needed/entered this. nearestPD has worked"""
                count = 0
                while log_term<0:
                    if count == 0:
                        print("Covariance matrix injected with sphericity")
                    prec = prec + np.identity(prec.shape[0])*nu*df*10
                    log_term = (1+(1.0/nu)*np.matmul(np.matmul(y_flat, prec),y_flat))
                    count = count+1

            elif p == 1:
                """If we only fit a single constant!"""
                return -pow(10,4)

            """If you have tried to make the matrix spd but it simply has not
            worked, take drastic action and simply set it to some extremely
            small value (this corresponds to saying that whatever has happened
            is extremely unlikely. That being said, I don't recall the algo
            ever entering this contition."""
            if( log_term<0):
                print("non-s.p.d. covariance estimate:",
                      "problem persists! Set it to log(pow(10,-100))")
                print("log term is", log_term)
                print("det term is", log_det)
                #log_term = np.log(pow(10,-50))
                #log_det = np.log(pow(100, p)) # = slogdet(np.identity*100)
                return -pow(10,5)
            else:
                log_term = np.log(log_term)
                _, log_det = np.linalg.slogdet(prec)
            if np.isnan(log_det):
                print("log_det nan: problem persists!")

        else:
            log_term = np.log(log_term)

        """Note: Should not happen after we have corrected b_ to be positive.
        I have never entered this condition in any simulation."""
        if np.isnan(log_det):
            print("nan log det")
            _, log_det = np.linalg.slogdet(prec)
            log_det = 1.0/log_det #since we want the log det of cov mat
            if np.isnan(log_det):
                print("problem persists!")

        calc = (special.gammaln(0.5*(nu+p)) - special.gammaln(0.5*nu) -
                 0.5*p*( np.log(nu) + np.log(np.pi) ) - 0.5*log_det -
                 0.5*(nu+p)*log_term)

        """Note: Should not happen after we have corrected b_ to be positive.
        I have never entered this condition in any simulation."""
        if np.isnan(calc):
            print("Alert! Calc is nan")
            calc = -pow(10,5)
        return calc


    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    """              TRIMMING THE RUN-LENGTH-DISTRIBUTION                """
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


    def trimmer(self, kept_run_lengths, BAR_submodel = False):
        """Trim the relevant quantities for the BVAR NIG model"""

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
        self.beta_XX_beta_rt = self.beta_XX_beta_rt[kept_run_lengths]
        self.XX_rt = self.XX_rt[kept_run_lengths,:,:]
        self.XY_rt = self.XY_rt[kept_run_lengths,:]
        self.YY_rt = self.YY_rt[kept_run_lengths]
        #QR ADAPTION
        #self.log_det_rt = self.log_det_rt[kept_run_lengths]
        #self.R_rt = self.R_rt[kept_run_lengths,:,:]
        #self.Q_rt = self.Q_rt[kept_run_lengths,:,:]
        self.M_inv_1_rt = self.M_inv_1_rt[kept_run_lengths,:,:]
        self.M_inv_2_rt = self.M_inv_2_rt[kept_run_lengths,:,:]
        self.log_det_1_rt = self.log_det_1_rt[kept_run_lengths]
        self.log_det_2_rt = self.log_det_2_rt[kept_run_lengths]
        self.retained_run_lengths = (
                    self.retained_run_lengths[kept_run_lengths])
        self.model_log_evidence = scipy.misc.logsumexp(
                        self.joint_log_probabilities )


    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    """                  REGRESSOR EXTRACTOR FUNCTIONS                  """
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    def get_exo_regressors(self, regressor_code, i, data):
        """Extract all the exogeneous regressors as you should"""

        """If the regressor_code is intercept, you just create intercepts.
        Otherwise, you just extract the relevant row in data"""
        #DEBUG: Check that data has format (num_exos, S1,S2)!
        if regressor_code == "intercept":
            data_vector = np.ones((self.S1, self.S2)) #assumes ONE intercept
        elif self.intercept_codes != ["intercept"]:
            """retrieve the number of the group in question"""
            group_number = int(regressor_code.split("_")[-1])
            """return the intercept only at the desired locations"""
            data_vector = self.intercept_grouping[group_number,:,:].flatten()
        else:
            data_vector = data[i,:,:].flatten()
        return data_vector.flatten()


    def get_endo_regressors(self, regressor_code, lag, data,
                            padding_column=None):
        """Get the predictors in order, where we give back the *data* as it
        should be. In particular, give the *lag*-th lattice slide with the
        neighbourhood as specified by *position*.
        Around the edges, the *padding* is applied. The argument
        *padding_column* is only for the demo object BVAR_NIG_CSurf: It is the
        rhs or lhs column of the lattice.

        NOTE: The lag passed to this function is the actual lag - 1.
              The data passed to this function at time t has times t-l:t-1
        """
        padding = self.padding
        lag = -(lag - 1) #need this conversion since we have 1,2,3...,T order,
                         #but want to access the l-th lag, i.e. T-l. Also, the
                         #FIRST lag corresponds to 0, i.e. T-1 i.e. the last
                         #entry contained in the endogeneous regressors

        """STEP 1: Compute padding for rows, columns, and corners"""
        if padding == 0 or padding == "zero":
            padding_row = np.zeros(self.S2)
            padding_col = np.zeros(self.S1)
            padding_corners = 0.0
        elif padding == "overall_mean":
            mean = np.mean(data[lag,:,:])
            padding_row = mean * np.ones(self.S2) #np.mean(data[lag,:,:], axis=0)
            padding_col = mean * np.ones(self.S1)
            padding_corners = mean
        elif padding == "row_col_mean":
            padding_row = np.mean(data[lag,:,:], axis=0)
            padding_col = np.mean(data[lag,:,:], axis=1)
            weight = (np.size(padding_row)/
                      (np.size(padding_row) + np.size(padding_col)))
            padding_corners = (weight*np.sum(padding_row) +
                               (1-weight)*np.sum(padding_col))

        elif padding.split("_")[-1] == "rhs" or  padding.split("_")[-1] == "lhs":
            """I.e., if we have a CSurf object, we need some extra care at the
            boundaries of the change surface"""
            padding_row = np.mean(data[lag,:,:], axis=0)

            if padding.split("_")[-1] == "rhs":
                """get padding for cols as usual + specific one for rhs, lhs"""
                padding_rhs = padding_column
                padding_lhs = padding_col = np.mean(data[lag,:,:], axis=1)
                weight = (np.size(padding_row)/
                      (np.size(padding_row) + np.size(padding_col)))
                padding_corner_rhs = (weight*np.sum(padding_row) +
                               (1-weight)*np.sum(padding_rhs))
                padding_corner_lhs = padding_corners = (
                        weight*np.sum(padding_row) +
                               (1-weight)*np.sum(padding_lhs))
            else:
                """get padding for cols as usual + specific one for rhs, lhs"""
                padding_rhs = padding_col = np.mean(data[lag,:,:], axis=1)
                padding_lhs = padding_column
                weight = (np.size(padding_row)/
                      (np.size(padding_row) + np.size(padding_col)))
                padding_corner_rhs = padding_corners = (weight*
                        np.sum(padding_row) + (1-weight)*np.sum(padding_rhs))
                padding_corner_lhs = (weight*np.sum(padding_row) +
                               (1-weight)*np.sum(padding_lhs))

        #data_vector = np.ones((self.S1, self.S2))
        """STEP 2A: Get the data_vector for the 4-nbh case or intercept/center"""
        #DEBUG: intercept will be called in exo regr, this is redundant
        if regressor_code == "intercept":
            data_vector = np.ones((self.S1, self.S2))
        elif regressor_code == "center":
            data_vector = data[lag,:,:]
        elif regressor_code == "left":
            if padding.split("_")[-1] == "rhs":
                """Insert the padding column passed to this function"""
                data_vector = np.insert(data[lag,:,:-1], 0, padding_rhs, axis=1)
            else:
                """Take the row averages as padding"""
                data_vector = np.insert(data[lag,:,:-1], 0, padding_col, axis=1)

        elif regressor_code == "right":
            if padding.split("_")[-1] == "lhs":
                """Insert the padding column passed to this function"""
                data_vector = np.insert(data[lag,:,1:], self.S2-1, padding_lhs, axis=1)
            else:
                """Take the row averages as padding"""
                data_vector = np.insert(data[lag,:,1:], self.S2-1, padding_col, axis=1)

        elif regressor_code == "top":
            data_vector = np.insert(data[lag,:-1,:], 0, padding_row, axis=0)
        elif regressor_code == "bottom":
            data_vector = np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0)

        elif regressor_code == "topleft":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[1:, 1:] = data[lag,:-1,:-1]
            if padding.split("_")[-1] == "rhs":
                """Insert the padding column passed to this function"""
                data_vector[0,:] = np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:,0] = np.append(padding_corner_rhs, padding_rhs[:-1])
            else:
                """Take the row averages as padding"""
                data_vector[0,:] = np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] = np.append(padding_corners, padding_col[:-1])


        elif regressor_code == "topright":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[1:, :-1] = data[lag,:-1,1:]
            if padding.split("_")[-1] == "lhs":
                """Insert the padding column passed to this function"""
                data_vector[0,:] = np.append( padding_row[1:], padding_corner_lhs)
                data_vector[:,-1] = np.append(padding_corner_lhs, padding_lhs[:-1])
            else:
                """Take the row averages as padding"""
                data_vector[0,:] = np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] = np.append(padding_corners, padding_col[:-1])

        elif regressor_code == "bottomleft":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[:-1, 1:] = data[lag,1:,:-1]
            if padding.split("_")[-1] == "rhs":
                """Insert the padding column passed to this function"""
                data_vector[-1,:] = np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:,0] = np.append( padding_rhs[1:],padding_corner_rhs)
            else:
                """Take the row averages as padding"""
                data_vector[-1,:] = np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] = np.append(padding_col[1:], padding_corners)


        elif regressor_code == "bottomright":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[:-1, :-1] = data[lag,1:,1:]
            if padding.split("_")[-1] == "lhs":
                """Insert the padding column passed to this function"""
                data_vector[-1,:] = np.append( padding_row[1:], padding_corner_lhs)
                data_vector[:,-1] = np.append(padding_lhs[1:],padding_corner_lhs )
            else:
                """Take the row averages as padding"""
                data_vector[-1,:] = np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] = np.append(padding_col[1:], padding_corners)

        elif regressor_code == "4_inner_nbh_res":
            if padding.split("_")[-1] == "lhs":
                #pad with real data on the right
                data_vector = (np.insert(data[lag,:,:-1], 0, padding_col, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_lhs, axis=1) +
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) +
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))
            elif padding.split("_")[-1] == "rhs":
                data_vector = (np.insert(data[lag,:,:-1], 0, padding_rhs, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_col, axis=1) +
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) +
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))
            else:
                data_vector = (np.insert(data[lag,:,:-1], 0, padding_col, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_col, axis=1) +
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) +
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))

        elif ((regressor_code == "4_outer_nbh_res") or
              (regressor_code == "8_nbh_res")):
            if padding.split("_")[-1] == "lhs":
                #pad with real data on the right
                """initialize with topleft"""
                data_vector = np.zeros((self.S1, self.S2))
                data_vector[1:, 1:] = data[lag,:-1,:-1]
                data_vector[0,:] = np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] = np.append(padding_corners, padding_col[:-1])
                """add topright"""
                data_vector[1:, :-1] += data[lag,:-1,1:]
                data_vector[0,:] += np.append( padding_row[1:], padding_corner_lhs)
                data_vector[:,-1] += np.append(padding_corner_lhs, padding_lhs[:-1])
                """add bottomleft"""
                data_vector[:-1, 1:] += data[lag,1:,:-1]
                data_vector[-1,:] += np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] += np.append(padding_col[1:], padding_corners)
                """add bottomright"""
                data_vector[:-1, :-1] += data[lag,1:,1:]
                data_vector[-1,:] += np.append( padding_row[1:], padding_corner_lhs)
                data_vector[:,-1] += np.append(padding_lhs[1:],padding_corner_lhs )
            elif padding.split("_")[-1] == "rhs":
                #pad with real data on the left
                """initialize with topleft"""
                data_vector = np.zeros((self.S1, self.S2))
                data_vector[1:, 1:] = data[lag,:-1,:-1]
                data_vector[0,:] = np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:,0] = np.append(padding_corner_rhs, padding_rhs[:-1])
                """add topright"""
                data_vector[1:, :-1] += data[lag,:-1,1:]
                data_vector[0,:] += np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] += np.append(padding_corners, padding_col[:-1])
                """add bottomleft"""
                data_vector[:-1, 1:] += data[lag,1:,:-1]
                data_vector[-1,:] += np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:,0] += np.append( padding_rhs[1:],padding_corner_rhs)
                """add bottomright"""
                data_vector[:-1, :-1] += data[lag,1:,1:]
                data_vector[-1,:] += np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] += np.append(padding_col[1:], padding_corners)
            else:
                """initialize with topleft"""
                data_vector = np.zeros((self.S1, self.S2))
                data_vector[1:, 1:] = data[lag,:-1,:-1]
                data_vector[0,:] = np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] = np.append(padding_corners, padding_col[:-1])
                """add topright"""
                data_vector[1:, :-1] += data[lag,:-1,1:]
                data_vector[0,:] += np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] += np.append(padding_corners, padding_col[:-1])
                """add bottomleft"""
                data_vector[:-1, 1:] += data[lag,1:,:-1]
                data_vector[-1,:] += np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] += np.append(padding_col[1:], padding_corners)
                """add bottomright"""
                data_vector[:-1, :-1] += data[lag,1:,1:]
                data_vector[-1,:] += np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] += np.append(padding_col[1:], padding_corners)

        if regressor_code == "8_nbh_res":
            """Will already have the outer 4-nbh stored inside of data_vector
            due to the previous elif-bracket call!"""
            if padding.split("_")[-1] == "lhs":
                data_vector += (np.insert(data[lag,:,:-1], 0, padding_col, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_lhs, axis=1) +
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) +
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))
            elif padding.split("_")[-1] == "rhs":
                data_vector += (np.insert(data[lag,:,:-1], 0, padding_rhs, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_col, axis=1) +
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) +
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))
            else:
                data_vector += (np.insert(data[lag,:,:-1], 0, padding_col, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_col, axis=1) +
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) +
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))

        """STEP 2B: Get the data_vector for the general nbh case"""
        """Note: No need to extract the lag, since that is already passed
            to the function as *lag*. But we do need to know which group
            we need to extract the regressors from. We want the g-th group
            for each neighbourhood, so we access the g-th element in
            [[[1,2,3],[4,5],[6]], [[3,4],[5,6,7],[8,9]], ... ], i.e. if g=0
            we have relevant_nbh = [1,2,3], then relevant_nbh = [3,4], etc."""
        if regressor_code.split("_")[0] == "general":
            data_vector = np.zeros((self.S1, self.S2))

            """STEP 2B.a: If we have strong coupling, all nbhs N_i(s) for a
            specific i but pooled across space have the same effect beta_i for
            each location inside N_i(s)."""
            if self.general_nbh_coupling == "strong coupling":
                """In this case, we have the same input as for weak
                coupling, but a different interpretation. In particular, we couple
                the effects over different spatial locations. Accordingly, we save

                general_nbh_<lag>_<nbh_index> only.

                Then, in the extractors, we loop over <loc> to retrieve the
                regressors in a single column as

                regressor(<lag>, <nbh_index>)[<loc>] = sum over all measurements
                    at time t - <lag> for nbh given by
                    gen_nbh_seq[<loc>][gen_nbh_res_seq[<lag>][<nbh]].
                """
            #if self.general_nbh_coupling == "strong coupling":
                data_vector = data_vector.flatten()
                """num_group retrieves the index i of N_i(s)"""
                num_group = (regressor_code.split("_")[-1])

                if num_group == "center":
                    #We just need the old obs of same location
                    data_vector = data[lag,:,:].flatten()
                else:
                    """convert from str to int"""
                    num_group = int(num_group)
                    """relevant_nbhs retrieves the collection
                    {N_i(s): s \in S1*S2}"""
                    relevant_nbhs = [item[num_group] for item in
                             self.general_nbh_sequence]
                    """Since we assume the s' \in {N_i(s): s \in S1*S2} to have the
                    same linear effect, sum them all up s.t. we put
                    sum(s' \in N_i(s)) in position i"""

                    dat = data[lag,:,:].flatten()
#                    print(relevant_nbhs)
#                    print([sum(dat[relevant_nbh]) for relevant_nbh
#                                    in relevant_nbhs])
#                    print(np.array([sum(dat[relevant_nbh]) for relevant_nbh
#                                    in relevant_nbhs]))
#                    print(data.vector.shape)
#                    print(np.array([sum(dat[relevant_nbh]) for relevant_nbh
#                                    in relevant_nbhs]).shape)
                    data_vector = np.array([sum(dat[relevant_nbh]) for relevant_nbh
                                    in relevant_nbhs])

            elif self.general_nbh_coupling == "weak coupling":
                """STEP 2B.b: If we have weak coupling, all s' \in N_i(s) for a
                specific i and a specific space have the same effect beta_is for
                each location inside N_i(s)."""
                data_vector = data_vector.flatten()
                """num_group retrieves the index i of N_i(s) OR 'center' for
                the autoregressive term"""
                if regressor_code.split("_")[-2] == "center":
                    #retrieve AR term
                    location = int(regressor_code.split("_")[-1])
                    data_vector[location]= (data[lag,:,:].flatten()
                            [location])
                else:
                    """What we want here is retrieve the data corr. to the
                    <nbh>-th nbh of location <loc> at lag <lag> +1. This
                    is done by getting
                    general_nbh_sequence[<loc>][general_nbh_restriction_seq
                                        [<lag>][<nbh_index>]]
                    from strings of form
                    'general_nbh_sequence_<lag>_<nbh_index>_<loc>'
                    """
                    nbh_index = int(regressor_code.split("_")[-2])
                    #DEBUG: will be the same as lag (we hope)
                    lag2 = int(regressor_code.split("_")[-3])
                    #print(lag)
                    #print(nbh_index)
                    relevant_nbh_index = self.general_nbh_restriction_sequence[lag2][nbh_index]
                    """location retrieves the s of N_i(s)"""
                    location = int(regressor_code.split("_")[-1])
                    """relevant_nbh retrieves the indices s' \in N_i(s)"""
                    #print(self.general_nbh_sequence[location][relevant_nbh_index])
                    relevant_nbh = (self.general_nbh_sequence[location][relevant_nbh_index])
                    """now, the entire data vector is 0 except for the index corr.
                    to location s, where we store sum(s' \in N_i(s))"""
                    data_vector = np.zeros(self.S1*self.S2)
                    """if relevent_nbh = [], this will be FALSE and we return
                    a vector of zeros in that case."""
                    if relevant_nbh:
                        data_vector[location]= sum(data[lag,:,:].flatten()
                            [relevant_nbh])

        """STEP 3: Return the data vector"""
        return data_vector.flatten()


    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    """                  HYPERPARAMETER OPTIMIZATION                    """
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


    def prior_update(self,t, model_specific_rld):
        """Update b and a priors using a_, b_ as well as beta mean prior using
        beta_rt. DEPRECATED, DON'T USE"""

        #DEBUG: We lose a lot of information by exp-conversion
        filled_in = model_specific_rld > -np.inf
        """Update 1: Regression coefficients"""
        if True:
            self.prior_mean_beta = np.sum(self.beta_rt *
                                    np.exp(model_specific_rld[filled_in,np.newaxis]), axis=0)
            self.D_inv_b0 = np.matmul(self.D_inv, self.prior_mean_beta)
            self.b0_D_inv_b0 = np.inner(self.prior_mean_beta, self.D_inv_b0)
        """Update 2: variance parameters"""
        if True:
            a_vec = self.a + (self.retained_run_lengths+1.0)*0.5
            b_vec = (self.b + 0.5*(self.b0_D_inv_b0 +
                    self.YY_rt + self.beta_XX_beta_rt))
            self.a = np.inner(a_vec, np.exp(model_specific_rld[filled_in]))
            self.b = np.inner(b_vec, np.exp(model_specific_rld[filled_in]))


    """return a_vec and b_vec, needed in differentation"""
    def get_param_rt(self):
        a_rt = np.insert(self.a + (self.retained_run_lengths+1.0)*0.5,
                          0, self.a)
        b_rt = np.insert(
            self.b + 0.5*(self.b0_D_inv_b0 + self.YY_rt - self.beta_XX_beta_rt),
            0, self.b)
        return (a_rt, b_rt)


    def differentiate_predictive_log_distribution(self, y,t,
                                                  run_length_log_distro):
        """Take derivative of log-posterior predictive w.r.t.aa hyperpars a,b.
        The size of the output will be num_params x run_length_num"""

        """STEP 1: Compute expressions needed for further computation"""
        y = y.flatten()
        p = np.size(y)
        run_length_num = self.retained_run_lengths.shape[0]
        num_params = 2 #a,b

#        a_vec = np.insert(self.a + (self.retained_run_lengths+1.0)*0.5,
#                          0, self.a)
#        b_vec = np.insert(
#            self.b + 0.5*(self.b0_D_inv_b0 + self.YY_rt - self.beta_XX_beta_rt),
#            0, self.b)

        #ensures this function's compatability with BVARNIGDPD
        a_vec, b_vec = self.get_param_rt()
        resids = np.insert(np.matmul(self.X_t,self.beta_rt[:,:,np.newaxis]) -
                y.flatten()[:,np.newaxis],0,
                np.matmul(self.X_t, self.prior_mean_beta[:,np.newaxis]) -
                y.flatten()[:,np.newaxis], axis=0)

        products = np.sum(resids *
            np.matmul(self.C_t_inv, resids), axis=1).reshape(run_length_num+1)


        """STEP 2: Compute a-derivative in log form"""
        expr1 = scipy.special.psi(a_vec + 0.5*p)
        expr2 = -scipy.special.psi(a_vec)
        expr3 = -p/(2.0*a_vec) #expr4 = 0
        expr5 = 0.5*self.S1*self.S2*(1.0/a_vec)

        """Note: The inside of expression 6 could be negative, so we store its
        sign and take the log of its absolute value. We then pass the signs on
        so that the actual value of the gradient can be constructed as
        sign(inside6)*np.exp(gradient)"""
        inside6 = (1.0 + 0.5 * (1/b_vec) * products)
        expr6 = -np.log(np.abs(inside6))
        log_posterior_predictive_gradients_a_val = (expr1 + expr2 + expr3 +
                                                expr5 + expr6)

        """STEP 3: Comute b-derivative in log form"""
        expr5_ = -0.5*self.S1*self.S2*(1.0/b_vec)
        expr6_ = (0.5*(p + 2*a_vec))*(1/(b_vec*b_vec))*(
                    0.5*products/(1+0.5*(1/b_vec)*products))
        log_posterior_predictive_gradients_b_val = expr5_ + expr6_

        """STEP 4: Convert d/dtheta log(f(a,b)) = f'(a,b)/f(a,b) into what
        you actually want: log(f'(a,b)) = log( log(f(a,b)) * f(a,b) ) =
        log(f'/f) + log(f) = log('log_posterior_predictive_gradients_val) +
        self.evaluated_log_predictive_distr. We need to take special care
        about the signs for the a-gradient, since we already have the sign
        of inside6, and we need to multiply it with the sign of the log-form
        log_posterior_predictive_gradients_a_val."""

        """STEP 4.1: log gradients' signs"""
        log_gradients_a_sign = (np.sign(inside6)*
            np.sign(log_posterior_predictive_gradients_a_val))
        log_gradients_b_sign = np.sign(
            log_posterior_predictive_gradients_b_val)

        """STEP 4.2: log gradients' values"""
        #DEBUG: Only occurence of self.one_step_ahead_predictive_log_probs
        all_predictives = np.insert(
            self.one_step_ahead_predictive_log_loss, 0, self.r0_log_loss)
        log_gradients_a_val = (np.log(
                np.max(
                    np.array([
                        np.abs(log_posterior_predictive_gradients_a_val),
                               0.000005 * np.ones((run_length_num+1))]
                    ), axis=0)
                ) + all_predictives)
        log_gradients_b_val = (np.log(
                np.max(
                    np.array([
                        np.abs(log_posterior_predictive_gradients_b_val),
                               0.000005 * np.ones((run_length_num+1))]
                    ), axis=0)
                ) + all_predictives)


        """STEP 5: If you use a generalized Bayesian loss function, adapt the
        derivatives you return. In particular, you will want the derivative of
        exp{loss(P(y_t|y_1:t-1, r_t))} for general losses. In the KL-case,
        loss(.) = log(.) and nothing more needs to be done. In other cases, we
        need to take some more steps"""
        #DEBUG: Don't do this!

#        """STEP 5A: Power divergence, i.e. we use the loss function
#            loss(P(y_t|y_1:t-1, r_t)) =
#                P(y_t|y_1:t-1, r_t)^alpha + (1/(alpha+1)) *
#                integral(P(y_t|y_1:t-1, r_t)^(alpha + 1)dy_t,
#
#        The derivative of which is
#            d/dtheta(loss(P(y_t|y_1:t-1, r_t)) =
#                alpha * [d/dtheta P(y_t|y_1:t-1, r_t)]^(alpha-1) +
#                alpha/(alpha+1) *  integral * |b*/a* * C_t_inv|^-1  *
#                    d/dtheta |b*/a* * C_t_inv|,
#
#        which we store in log-form."""
#        if self.generalized_bayes == "power_divergence":
#            #DEBUG: predictive_variance_log_det needs adjustment for a* and b*!
#            #       they are stored inside there, so we need to subtract them
#            #       again! i.e. multiply by a*/b*!
#
#            """STEP 5A.1: Get all the parts that we need to sum over as logs"""
#            integrals = self.get_log_integrals_power_divergence()
#            modified_log_gradient_a_1 = (np.log(self.alpha) +
#                                       (self.alpha - 1) * log_gradients_a_val)
#            modified_log_gradient_b_1 = (np.log(self.alpha) +
#                                       (self.alpha - 1) * log_gradients_b_val)
#            """Note: Sign of log_det_der_a is always negative! I.e., here we
#            store the log(X*) of X s.t. X = -exp(log(X*)), where X = det_der_a
#            and log(X*) = log_det_der_a"""
#            log_det_der_a = (np.log(self.S1) + np.log(self.S2)
#                - np.log(a_vec) + (self.predictive_variance_log_det))
#            log_det_der_b = (np.log(self.S1) + np.log(self.S2)
#                - np.log(b_vec) + (self.predictive_variance_log_det))
#            """Note: sign of modified_log_gradient_a_2 is negative, as
#            log_det_der_a has negative sign in exponential form, but integrals,
#            alpha, and determinant will have positive values"""
#            modified_log_gradient_a_2 = (integrals + np.log(self.alpha) -
#                                         np.log(self.alpha + 1) -
#                                         self.predictive_variance_log_det  +
#                                         log_det_der_a)
#            modified_log_gradient_b_2 = (integrals + np.log(self.alpha) -
#                                         np.log(self.alpha + 1) -
#                                         self.predictive_variance_log_det  +
#                                         log_det_der_b)
#
#            """STEP 5A.2: Sum over the log-parts using logsumexp to retrieve
#            the final gradients. Notice that we have the signs of the
#            modified_log_gradient_a_1 (b_1) from before, since the only part
#            modifying the sign from positive is the gradient of
#            P(y_t|y_1:t-1,r_t). Similarly, modified_log_gradient_a_2 (b_2) will
#            be completely deterimined in its sign by log_det_der_a (_b), which
#            is always positive for b and always negative for a."""
#            log_gradients_a_val, log_gradients_a_sign = (
#                scipy.misc.logsumexp(
#                    a = np.array([modified_log_gradient_a_1,
#                                  modified_log_gradient_a_2]),
#                    b = np.array([log_gradients_a_sign,
#                        -np.ones(np.size(modified_log_gradient_a_2))]),
#                    axis = 0,
#                    return_sign = True
#                ))
#            log_gradients_b_val, log_gradients_b_sign = (
#                scipy.misc.logsumexp(
#                    a = np.array([modified_log_gradient_b_1,
#                                  modified_log_gradient_b_2]),
#                    b = np.array([log_gradients_b_sign,
#                        np.ones(np.size(modified_log_gradient_b_2))]),
#                    axis = 0,
#                    return_sign = True
#                ))

        """STEP 5: Package the results for each hyperparameter derivative,
        put it into one object, and return it"""
        log_gradients_sign = np.array([log_gradients_a_sign,
                            log_gradients_b_sign]).reshape(num_params, run_length_num+1)
        log_gradients_val = np.array([log_gradients_a_val,
                      log_gradients_b_val]).reshape(num_params, run_length_num+1)

        return log_gradients_val, log_gradients_sign


    def caron_hyperparameter_optimization(self, t, gradient, step_size):
        """Gradient for type-II ML for the hyperparameters a and b, as in
        Caron, Doucet, Gottardo (2012).
        Called by ProbabilityModel level at each time step.

        Note: Though step_size is passed, we compute the step size inside
                this function using a dampened version of BB. The step size
                passed into this function from the detector is used only
                if the BB-based one is smaller
        """

        """STEP 1: Get the difference a_{t-1} - a_{t-2}, b_{t-1} - b_{t-2}"""
        #scale = -10
        max_step_scale = 1#pow(10,-1)#*0.05 #pow(10,-3)*5#0.05
        min_step_scale = pow(10,-5)#0.001
        disturbance = 0.0 #np.random.normal(loc=0.0, scale=pow(10,scale)) #
        dif_old_a = self.a - self.a_old  + disturbance
        dif_old_b = self.b - self.b_old  + disturbance
        dif_old_val = np.array([dif_old_a, dif_old_b])
        dif_old_grad = gradient - self.gradient_old

        """STEP 2: Compute step size via a hybrid-BB method"""
        if True: #Might want to add different options w.r.t step size
            """These conversions are necessary for numerical stability"""
            dampener = pow((1/1.005), t*0.1)#pow(0.5, t*0.25)

            """D1, D2, D3 are needed for both types of BB-step sizes"""
            D1_sign = np.dot(np.sign(dif_old_val), np.sign(dif_old_grad))
            if D1_sign >0:
                D1 = min(np.abs(np.dot(dif_old_val, dif_old_grad)), pow(10,5))
            else:
                D1 = max(- np.abs(np.dot(dif_old_val, dif_old_grad)),
                         -pow(10, 5))
            if np.abs(D1) < pow(10,-1):
                D1 = (pow(10,-1))*D1_sign
            D2 = min(np.dot(dif_old_grad, dif_old_grad), pow(10,5))
            if D2 < pow(10,-1):
                D2 = pow(10,-1)
            D3 = min(np.dot(dif_old_val, dif_old_val), pow(10,5))
            if D3 < pow(10,-1):
                D3 = pow(10,-1)
            alpha_1 = ((D1/D2)*dampener) #/(pow(10,-scale))

            """Take either the dampened BB step size, or the step size passed
            down from the Detector object."""
            step_size_abs = max(np.abs(alpha_1),step_size)

            step_size = np.sign(alpha_1)*step_size_abs
            """If alpha_1 = 0, the sign of step_size*gradient[i] will be 0,
            so we take account of this case here:"""
            if np.sign(alpha_1) == 0.0:
                sign_a = np.sign(gradient[0])
                sign_b = np.sign(gradient[1])
            else:
                sign_a = np.sign(step_size*gradient[0])
                sign_b = np.sign(step_size*gradient[1])

            """min_step_scale and max_step_scale give you a bound on how big a
            step can be relative to the current value of a (b). Here, we set
            them to be at 5% and 0.1% of the current value. This prevents
            explosive behaviour."""
            dampener = 1.0/t
            increment_a = (sign_a*
                max(self.a*dampener*min_step_scale,
                min( self.a*dampener*max_step_scale,
                    np.abs(step_size_abs*gradient[0]))))
            increment_b = (sign_b*
                max(self.a*dampener*min_step_scale,
                min( self.b*dampener*max_step_scale,
                    np.abs(step_size_abs*gradient[1]))))
            dampener = 1.0/t
#            increment_a = sign_a * dampener * min(np.abs(gradient[0]), pow(10,5))
#            increment_b = sign_b * dampener * min(np.abs(gradient[1]), pow(10,5))

        """STEP 3: Calculate the new values of a and b, and store the previous
        values (needed for BB-method of computing step sizes)"""
        self.a_old = self.a
        self.b_old = self.b
        self.gradient_old = gradient
        self.a = min(max(self.a + increment_a,
                         1.0), pow(10,pow(10,3)))
        self.b = min(max(self.b + increment_b,
                         pow(10,-10)), pow(10,pow(10,3)))
        self.a_list.append(self.a)
        self.b_list.append(self.b)


    def get_one_step_ahead_log_loss_derivatives_power_divergence(self):
        """Obtain d/(d alpha) P(y_t|y_1:t-1, r_t-1, m_t-1) as well as the
        derivative w.r.t. the prior P(y_t|r_t=0). for the case of the power
        divergence. log them, i.e. you return log( d/dalpha ... ) with the
        signs.

        Note 1: The notation used is as in my personal notes. I.e., we use
                f_1, f_2, f_3 (f_1_der, ...) as there for the computation of
                the integral's derivative.
        Note 2: We build up from f_1, f_2, f_3 towards the final predictive.
                Inbetween, we will need the integral's actual value as well as
                the predictive probabilities (NOT losses)
        Note 3: We also compute all one-step-aheads, including when the run-
                length is 0.
        """
        #DEBUG: Output is huuuge, largely dominated by log_constant expression


        """STEP 1: Get quantities that are needed throughout the computation
        and appear in the (log) gamma functions of the MVSt. Perturb the self.a
        such that we have at least 1.001 since log(log(alpha)) is computed for
        f_3_der_log, and log(1.0) = -infinity."""
        p = self.S1*self.S2
        run_length_num = np.size(self.retained_run_lengths) + 1
        """Note: Take the element-wise maximum between the nu and 1.005 to
        avoid having np.log(np.log(1.0)) = -np.inf in f_2, f_3"""
        #basically a_vec again!
        a_vec, _ = self.get_param_rt()
        nu_1_vec = np.maximum(2 * a_vec,
                              #np.insert(
            #self.a + 0.5*(self.retained_run_lengths + 1),0, self.a),
            1.005)
        nu_2_vec = np.maximum(
            nu_1_vec*p + nu_1_vec*self.alpha_rld + nu_1_vec ,
            1.005)

        """STEP 2: Compute the derivative of the integral. To this end,
        compute log f_1, f_2, f_3 and their derivatives. Note that the
        gamma-function is positive as long as its argument is.  Since the
        arguments will be positive, this means that f_1, f_2, f_3 are positive,
        too and so their logs are well-defined. See (7) of my handwritten
        notes. For Sigma = posterior covariance, we have

            f_1 = [Gamma(0.5 * {p + nu_1})/Gamma(0.5 * nu_1)]^(1+alpha)
            f_2 = [Gamma(0.5 * nu_2)]/Gamma(0.5 * {nu_2 + p})]
            f_3 = (nu_1*pi)^(-0.5*p*alpha) * det(Sigma)^(-alpha)"""

        f_1_log = ( (1.0 + self.alpha_rld)*(special.gammaln(0.5*(nu_1_vec + p)) -
                                     special.gammaln(0.5*nu_1_vec)))
        f_2_log = (special.gammaln(0.5*(nu_2_vec)) -
                   special.gammaln(0.5*nu_2_vec + p))
        f_3_log = ( (np.log(nu_1_vec) + np.log(np.pi))*(-0.5*p*self.alpha_rld) +
                (-self.alpha_rld) * self.predictive_variance_log_det)

        """STEP 2.2: Compute logs of derivatives of f_1, f_2, f_3"""

        """STEP 2.2.1: Compute log derivative of f_1. We have
            f_1_der = [log(Gamma(0.5 * {p + nu_1})) - log(Gamma(0.5 * nu_1))] *
                      [Gamma(0.5 * {p + nu_1})/Gamma(0.5 * nu_1)]^(1+alpha)
        Since the second part is always positive, the sign of the derivative
        is determined by the first expressions only."""
        expr_1 = (scipy.special.gammaln(0.5*(nu_1_vec + p)) -
                    scipy.special.gammaln(0.5*nu_1_vec))
        f_1_der_sign= np.sign(expr_1)
        f_1_der_log = (np.log(expr_1 * f_1_der_sign) +
                       (1.0 + self.alpha_rld) * expr_1)

        """STEP 2.2.2: Compute log derivative of f_2. We have
            f_2_der = [Gamma(nu_2*0.5) * Psi(nu_1*0.5) * {0.5*(nu_1+p)} *
                      1/Gamma(0.5*{nu_1 + p})]
                      -
                      [1/Gamma(0.5*{nu_2 + p}) * Psi(0.5*{nu_2 + p}) *
                      {0.5*(p+nu_1)} * Gamma(0.5*nu_2)]
        Both parts of the expressions are sign due to the psi function
        ( =digamma function). Note thatthe second expression is subtracted
        (hence the minus sign when we instantiate expr_2_B_sign)"""
        digamma_expr_2_A = scipy.special.digamma(0.5* (nu_2_vec))
        expr_2_A_sign = np.sign(digamma_expr_2_A)
        expr_2_A = (
                scipy.special.gammaln(0.5 * (nu_2_vec )) +
                np.log( expr_2_A_sign * digamma_expr_2_A)  +
                np.log(0.5*(nu_1_vec + p)) -
                scipy.special.gammaln(0.5*(nu_2_vec + p))
            )

        digamma_expr_2_B = scipy.special.digamma(0.5* (nu_2_vec + p))
        expr_2_B_sign = np.sign(digamma_expr_2_B)
        expr_2_B = (
                -scipy.special.gammaln( 0.5 * ( p + nu_2_vec )) +
                np.log(expr_2_B_sign * digamma_expr_2_B) +
                np.log(0.5*(nu_1_vec + p)) +
                scipy.special.gammaln(0.5*(nu_2_vec))
                )
        """Note: expr_2_B is subtracted from expr_2_A, so reverse the sign"""
        expr_2_B_sign = -expr_2_B_sign

        f_2_der_log, f_2_der_sign = scipy.misc.logsumexp(
                a = np.array([
                        expr_2_A, expr_2_B
                        ]),
                b = np.array([
                        expr_2_A_sign, expr_2_B_sign
                        ]),
                return_sign = True,
                axis = 0
            )

        """STEP 2.2.3: Compute log derivative of f_3. We have
            f_3_der = {(nu_1*pi)^(0.5*p) * det(Sigma)}^(-alpha) *
                      [-0.5*p*{log(nu_1) + log(pi)} - log(det(Sigma))]
        Note that only the last expression is sign-sensitive, thereby
        determining the sign of the entrie derivative"""
        expr_3 = -(0.5*p*(np.log(nu_1_vec) + np.log(np.pi)) +
                   self.predictive_variance_log_det)
        f_3_der_sign = np.sign(expr_3)
        f_3_der_log = f_3_log  + np.log(f_3_der_sign * expr_3)


        """STEP 2.3: Put together f1, f2, f3 and their derivatives to get the
        derivative of the integral. We have
            der_integral = f_1_der * f_2 * f_3 + f_2_der * f_1 * f_3 +
                            f_3_der * f_1 * f_2 """
        f_1_full_expr = f_1_der_log + f_2_log + f_3_log
        f_2_full_expr = f_2_der_log + f_1_log + f_3_log
        f_3_full_expr = f_3_der_log + f_1_log + f_2_log
        log_integral_derivatives_val, log_integral_derivatives_sign = (
            scipy.misc.logsumexp(
                a = np.array([
                        f_1_full_expr,
                        f_2_full_expr,
                        f_3_full_expr]),
                b = np.array([f_1_der_sign,
                              f_2_der_sign,
                              f_3_der_sign]),
                return_sign = True,
                axis = 0
            ))

        """STEP 3: Combine everything into the derivative of the predictive
        loss/likelihood, using the quantities computed in STEP 2. Defining
        P(y_t|y_1:t-1, r_t, m_t) = f(y), we have
            loss(y) = exp([1/alpha] * f(y)^alpha - [1/(alpha+1)] * Int(alpha)),
        where
            Int(alpha) = Integral(f(y)^(alpha+1), y)
        We compute the derivative of the loss as
            loss_der(y) = loss(y) * {[-[1/alpha^2] * f(y)^alpha  +
                          [1/alpha]*log(f(y)) * f(y)^alpha] + [[1/(alpha+1)^2] *
                           Int(alpha) - [1/(alpha+1)] * Int_der(alpha)]}
       where we have computed Int_der(alpha) in log form in STEP 2."""

        """STEP 3.1: Get the LOG of the constant expression loss(y)"""
        integrals = self.get_log_integrals_power_divergence()

        predictive_log_probs = np.insert(
                self.one_step_ahead_predictive_log_probs.copy(), 0,
                self.r0_log_prob)

        log_constant = ((1.0/self.alpha_rld) *
            np.power(np.exp(predictive_log_probs), self.alpha_rld) -
            (1.0/(1.0+self.alpha_rld)) * np.exp(integrals))
        #DEBUG: This constant is F*** Huge! logged version is around 200, i.e.
        #       exponentiated we have something of order 10^86...
        #print(log_constant)

        """STEP 3.2: Get the first term we need for loss_der, namely
            [-[1/alpha^2] * f(y)^alpha  + [1/alpha]*log(f(y)) * f(y)^alpha]"""
        expr_1_A = -2.0*np.log(self.alpha_rld) + self.alpha_rld * predictive_log_probs
        sign_1_A = -np.ones(run_length_num)
        sign_1_B = np.sign(predictive_log_probs)
        expr_1_B = (-np.log(self.alpha_rld) +
                    np.log(sign_1_B * predictive_log_probs) +
                    self.alpha_rld * predictive_log_probs)
        expr_1_val, expr_1_sign = scipy.misc.logsumexp(
                a = np.array([
                        expr_1_A,
                        expr_1_B
                        ]),
                b = np.array([
                        sign_1_A,
                        sign_1_B
                        ]),
                return_sign = True,
                axis = 0
            )
        #DEBUG: Not quite as large as the log_constant, but logged version still
        #       around 10, i.e. exp. version is around 25000
        #print(expr_1_val)

        """STEP 3.3: Get the second term, namely
            [[1/(alpha+1)^2] * Int(alpha) - [1/(alpha+1)] * Int_der(alpha)]"""
        expr_2_A = -2.0*np.log(self.alpha_rld + 1.0) + integrals
        sign_2_A = np.ones(run_length_num)
        sign_2_B = (-1)*log_integral_derivatives_sign
        expr_2_B = -np.log(self.alpha_rld +1) + log_integral_derivatives_val
        expr_2_val, expr_2_sign = scipy.misc.logsumexp(
                a = np.array([
                        expr_2_A,
                        expr_2_B
                        ]),
                b = np.array([
                        sign_2_A,
                        sign_2_B
                        ]),
                return_sign = True,
                axis=0
            )

        """STEP 3.4: add together expr_1_val and expr_2_val, yields log of
            {[-[1/alpha^2] * f(y)^alpha  + [1/alpha]*log(f(y)) * f(y)^alpha] +
            [[1/(alpha+1)^2] * Int(alpha) - [1/(alpha+1)] * Int_der(alpha)]}"""
        expr_val, expr_sign = scipy.misc.logsumexp(
                a = np.array([
                        expr_1_val,
                        expr_2_val
                        ]),
                b = np.array([
                        expr_1_sign,
                        expr_2_sign
                        ]),
                return_sign = True,
                axis = 0
            )

        """STEP 3.5: Finally, multiply with the constant ( = add the log of
        the constant) to retrieve the final logged value of
        d/dalpha(P(y_t|y_1:t-1, r_t, m_t))."""
        final_expr = log_constant + expr_val
        final_expr_sign = expr_sign
        #DEBUG: final_expr has huge values, mostly because of the log_constant
        #       term that dominates and is of log-order 200
        #print(final_expr)

        return final_expr, final_expr_sign


    def get_hyperparameters(self):
        """Simply returns a and b, used by HyperparameterOptimization"""
        return [self.a, self.b]


    def turner_hyperparameter_optimization(self, step_size):
        """Called by HyperparameterOpimizer at each iteration of the
        optimization routine.

        NOTE: Should be checked w.r.t. step-size computation, probably one can
        do better here!"""

        """STEP 1: Get the new gradient value"""
        sign, gradient = scipy.misc.logsumexp(
                a=(self.model_specific_joint_log_probabilities_derivative),
                b=self.model_specific_joint_log_probabilities_derivative_sign,
                return_sign=True, axis=1)
        gradient = np.exp(gradient)*sign

        """STEP 2: Get the difference a_{t-1} - a_{t-2}, b_{t-1} - b_{t-2}"""
        #Debug: Initialize self.a_old as self.a!, gradient_old = 0
        dif_old_a = self.a - self.a_old
        dif_old_b = self.b - self.b_old
        dif_old_val = np.array([dif_old_a, dif_old_b])
        dif_old_grad = gradient - self.gradient_old

        """STEP 3: Compute step size via BB method"""
        if True:
            D1_sign = np.dot(np.sign(dif_old_val), np.sign(dif_old_grad))
            D1 = min(max(np.abs(np.dot(dif_old_val, dif_old_grad)), pow(10,5)),
                     pow(10, -5))*D1_sign
            D2 = min(max(np.dot(dif_old_grad, dif_old_grad), pow(10,5)),
                     pow(10,-5))
            #D3 = min(max(np.dot(dif_old_val, dif_old_val), pow(10,5)),
            #         pow(10,-5))
            alpha_1 = D1/D2
            #alpha_2 = D3/D1
            step_size = alpha_1

        """STEP 2: Calculate the new a-value, and return the difference to
        old a-value, (a_{it+1} - a_it)"""
        self.a_old, self.b_old = self.a, self.b
        self.gradient_old = gradient
        self.a = min(max(self.a + gradient[0]*step_size, pow(10,-20)), pow(10,15))
        self.b = min(max(self.b + gradient[1]*step_size, pow(10,-20)), pow(10,15))
        a_dif, b_dif = self.a - self.a_old, self.b - self.b_old
        return [a_dif, b_dif]


    @staticmethod
    def objective_optimization(x, *args):
        """Objective function for type-II ML for the hyperparameters a and b.
        Can be used inside conjugate gradient descent methods like this one:
            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
            scipy.optimize.fmin_cg.html
        in the way outlined in this paper:
            http://mlg.eng.cam.ac.uk/pub/pdf/TurSaaRas09.pdf

        args = S1, S2, y_t, X_t (would be X_tp1 if this were called before
                updating the predictive distros), retained_run_lengths,
                run_length_distro, beta_rt, log_det_1, log_det_2, C_t_inv,
                b0_D_inv_b0, YY_rt, beta_XX_beta_rt
        x = a,b
        """

        """STEP 1: get *args, which are basically a heap of quantities stored
        inside BVARNIG that you have to pass to the static method"""
        (S1, S2, y, X, retained_run_lengths, run_length_distro, beta_rt,
             log_det_1_rt, log_det_2_rt, C_t_inv, b0_D_inv_b0, YY_rt,
             beta_XX_beta_rt) = args
        a,b = x
        y = y.flatten()

        """STEP 2: From the inputs, compute posterior predictive for all run
        lengths and multiply by the run-length distro specific to this
        BVAR model (do NOT use the model-and-run-length distro). Note that
        we will have to use logsumexp, as all the MVSt-densities are in log
        format"""

        """STEP 2.1: Get a_ and b_ for each run-length"""
        a_vec = a + (retained_run_lengths+1.0)*0.5
        b_vec = (b + 0.5*(b0_D_inv_b0 + YY_rt - beta_XX_beta_rt))

        """STEP 2.2: Get the log determinants as function of a_ and b_"""
        log_dets = ((S1 * S2)*(np.log(b_vec) - np.log(a_vec)) +
                           log_det_1_rt - log_det_2_rt)

        """STEP 2.3: Get the log densities corresponding to those values for
        all run-lengths"""
        run_length_num = retained_run_lengths.shape[0]
        MVSt_log_densities = np.array([BVARNIG.mvt_log_density(
            y_flat = (np.matmul(X, beta_rt[r,:]) - y.flatten()),
            prec = (a_vec[r]/b_vec[r])*C_t_inv[r,:,:],
            log_det = log_dets[r], prior = False, alerts = False)
            for r in range(0, run_length_num)])

        """STEP 2.4: Multiply the MVSt densities with the run_length distro,
        which is equivalent to adding them on a log scale. Afterwards, use
        logsumexp to get the sum over all of them (again on a log scale)"""
        evaluation_objective = scipy.misc.logsumexp(
                MVSt_log_densities + run_length_distro)

        """STEP 3: Return the objective value"""
        return evaluation_objective
