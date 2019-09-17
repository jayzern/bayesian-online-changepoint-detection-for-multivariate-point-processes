# Python 3.5.2 |Anaconda 3.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited: 3-12-2017
Author: Yannis Zachos (i.zachos@warwick.ac.uk)
Forked by: Jeremias Knoblauch (J.Knoblauch@warwick.ac.uk)

Description: Implements the naive (PG) model assuming that all locations are
uncorrelated & independent, yet have common change points.
"""

import numpy as np
from scipy import stats
import scipy.misc
from probability_model import ProbabilityModel
from cp_probability_model import CpModel

class PGModel(ProbabilityModel):
    """The naive Poisson Gamma model.

    Statistical model:
        $y_t,\lambda \sim \mathcal{POISSON}(y_t| \lambda) *
                    \mathcal{GAMMMA}(\lambda|\alpha_0, \beta_0)$
        In the remainder, \alpha_0 is called 'prior_alpha', \beta_0 'prior_beta'.

    Inputs at creation:
        S1, S2: int;
            give the dimensions of the lattice. The one-dimensional case
            corresponds to S1=1 and S2=1.
        prior_alpha: float;
            stores the shape for the prior intensity. Default is 1.
        prior_beta: float;
            stores the rate for the prior intensity. Default is 1.
        auto_prior_update: boolean
            Indicates whether the prior_mean and prior_var should be updated
            to the posterior expectation and posterior variance at each time
            step. Default is False.

    Internal Attributes:
        sums: float numpy array; dimension S1xS2x(t+1) at time t
            gives the sums for each of the S1*S2 series for each possible
            run length r=0,1,...t-1, >t-1. at each time point t. Since this
            model corresponds to the special case where we assume that there
            is no spatial information, the mean numpy array is stored in a
            lattice again. (I.e., has two dimensions)
            This object stores at position r the posterior sums corresponding
            to the run length of r=0,1,...,t at each time t.
        joint_log_probabilities: float numpy array; dimension t+1 at time t
            stores the joint log probabilities (y_{1:t}, r_t| q) for this
            model q and all potential run lengths r_t = 0,1,...t-1
        model_log_evidence: float;
            stores in a scalar the evidence for this model, i.e. the joint
            probability (y_{1:t}| q) for this model q.
    """


    def __init__(self, prior_alpha, prior_beta, prior_mean,
                 S1, S2,
                 auto_prior_update=False):
        """Construct a naive PG probability model, providing a prior
        as ...

        NOTE: The Input must always be given as a lattices, i.e.
        we have to input even the one-dimensional model in 2-dimensional form.
        If it is given in different form, simply convert it.
        """

        """initialize all the prior quantities"""
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.prior_mean = prior_mean.reshape(S1,S2)
        self.auto_prior_update = False

        """initialize all the quantities that we compute through time"""
        self.sums = None
        self.S1, self.S2 = S1, S2
        self.joint_log_probabilities = 1
        self.model_log_evidence = -np.inf
        self.retained_run_lengths = np.array([0,0])

        self.has_lags = False
        self.hyperparameter_optimization = False
        self.generalized_bayes_rld = "kullback_leibler"


    def initialization(self, y, cp_model, model_prior):
        """function which is only called ONCE, namely when the very first
        lattice of observations y is obtained/observed. Called from the
        'update_all_joint_log_probabilities' function within the Detector
        object containing the PGModel object.
        """
        y = y.reshape(self.S1, self.S2)

        """Evaluate the negative binomial pmf corresponding to the observations, and
        multiply by the *model_prior* of the corresponding model. The model
        prior is stored by the Detector object, and passed from there to the
        ProbabilityModel object only when the joint probabilities
        are updated"""
        # Equation 3.6 in notes
        evidence = np.sum(stats.nbinom.logpmf(y, self.prior_alpha, (self.prior_beta)/(self.prior_beta+1)))

        self.model_log_evidence = (np.log(model_prior) + evidence)

        """STEP 2: get the hazard/probability of CP using *cp_model* as passed
        to the probability model from the containing Detector object. Obtain
        the *joint_probabilities*. Use the 'pmf' function of the CP model only
        at time 0. This 'pmf' function is only used this once and NOWHERE ELSE,
        since the hazard function (i.e., the conditional probability of
        r_t = r| r_{t-1} = r' can be used from here on"""

        """Ensure that we do not get np.log(0)=np.inf by perturbation"""
        if cp_model.pmf_0(1) == 0:
            epsilon = 0.005
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
        self.retained_run_lengths = np.array([0,0])

        """STEP 3: update the sufficient statistics.
        For the mean, use the  same entry as suff statistic for both,
        r=0 and r>0, because we have no observation at t=0. Similarly for the
        scaled square sums and the sufficient statistic for the variance"""
        self.sums =  np.array([y,y])

    def evaluate_predictive_log_distribution(self, y, t):
        """Returns the log densities of *y* using the predictive posteriors
        for all possible run-lengths r=0,1,...,t-1,>t-1 as currently stored by
        virtue of the sufficient statistics. In particular, it holds that
        at lattice position (s1, s2) and for run length r,

            y(s1, s2) ~ NB(y(s1,s2)| (beta(s1,s2)[r]+r)/(beta(s1,s2)+r+1), alpha(s1,s2)[r] + r*means(s1,s2)[r]).

        The corresponding density is computed for all run-lengths and
        returned in a np array"""

        y = y.reshape(self.S1, self.S2)

        """get the posterior variance & dfs"""
        factors = (self.retained_run_lengths + 1)
        prob = 1 - (1 / (self.prior_beta + factors[:,np.newaxis,np.newaxis] + 1))
        size = self.prior_alpha + (factors[:,np.newaxis,np.newaxis]) * self.sums[:,:,:]

        """evaluate *y* using *prob* as prob and *size* as size.
        Compute predictive probability  using negative binomial pmf,
        assuming independence of all locations/series.
        This will return a vector of size t+1 corresponding to the predictive
        distributions under r=0,1,...,t-1, >t-1"""
        #print('predictive', np.sum(stats.nbinom.logpmf(y, size, prob), axis=(1,2)))
        return np.sum(stats.nbinom.logpmf(y, size, prob), axis=(1,2))

    def evaluate_log_prior_predictive(self, y, t):

        """Returns the prior log density of the predictive distribution
        for all possible run-lengths r=0,1,...,t-1,>t-1.
        This is the case when no other data has been processed.
        First, reshape the data as always.
        """
        y = y.reshape(self.S1, self.S2)

        """Get the probability and size parameters"""
        prob = (self.prior_beta + 1) / (self.prior_beta + 2)
        size = self.prior_alpha + y

        """Evaluate *y* using *prob* as prob and *size* as size.
        Compute predictive probability  using negative binomial pmf,
        assuming independence of all locations/series.
        This will return a vector of size t+1 corresponding to the predictive
        distributions under r=0,1,...,t-1, >t-1"""
        return np.sum(stats.nbinom.logpmf(y, size, prob), axis=0)

    # Jay: This function has been changed, find out why it didn't work
    def update_predictive_distributions(self, y, t, r_evaluations = None):

        """Takes the next observation, *y*, at time *t* and updates the
        sufficient statistics/means corresponding to all potential
        run-lengths r=0,1,...,t-1,>t-1.

        Since the two entries corresponding to r=t-1 and r>t-1 will always
        have the same sufficient statistic in this model, we only update the
        entries r=0,1,...,t-1 and then double the last one.
        """

        """STEP 0: Get the new observation into the right shape"""
        y = y.reshape(self.S1, self.S2)

        """STEP 0.1: global helper quantity for mean computation"""
        fac = self.retained_run_lengths

        """STEP 1: MEAN
        The sufficient statistics for the means are updated.
        Note that this requires the knowledge of the true posterior
        variances."""

        """STEP 1.1: update the means from t-r to t to the means from t-r to
        t+1 and add the t+1 th observation as the mean for r=0"""
        self.sums = (self.sums * fac[:,np.newaxis, np.newaxis] + y)/(fac[:,np.newaxis, np.newaxis]+1) # for r \neq 0
        self.sums = np.insert(self.sums, 0, y, axis=0) # for r = 0

        """STEP 4: Update retained_run_lengths in two steps: First by advancing
        the run lengths, second by adding the 0-run-length from this run."""
        self.retained_run_lengths = self.retained_run_lengths + 1
        self.retained_run_lengths = np.insert(self.retained_run_lengths, 0, 0)


    def get_posterior_expectation(self, t, r_list=None):

        """get the predicted value/expectation from the current posteriors
        at time point t, for all possible run-lengths"""

        """get the weights for the prior. Last two entries of w are the same
        by the usual r=t-1 and r>t-1 logic.
        Posterior expectation = (self.prior_beta/(self.prior.beta+t-r)*(self.prior.alpha/self.prior.beta) +
                                ((t-r)/(self.prior.beta+t-r)*((sum_{i=t-r}^{t} y_i))/t)
        where (sum_{i=1}^{t} y_i))/t is the empirical mean and prior.alpha/prior.beta is the prior mean.
        """
        fac = self.retained_run_lengths[:,np.newaxis,np.newaxis]


        """compute the weighted averages/posterior expectations"""
        output = (self.sums + self.prior_alpha) / (fac + self.prior_beta)
        #print('posterior expectation',output.shape)
        #exit()
        #print(output)
        #exit()
        return (output)



    def get_posterior_variance(self, t, r_list=None):

        """get the predicted variance from the current posteriors at
        time point t, for all possible run-lengths.
        Posterior variance is (self.prior_alpha + sum_{i=t-r}^{t} y_i) / (self.prior_beta + t-r)^2.
        """

        """STEP 0: store how many variance matrices you have to return"""
        run_length_num = self.retained_run_lengths.shape[0]
        fac = self.retained_run_lengths[:,np.newaxis,np.newaxis]

        """STEP 1: Flatten each S1xS2 matrix into an S1*S2 vector"""
        var = ((self.sums + self.prior_alpha) * (fac + self.prior_beta + 1) )/ (fac + self.prior_beta)**2

        variances = np.zeros(shape = (run_length_num,
                                 self.S1*self.S2, self.S1*self.S2))

        for d in range(0, run_length_num):
            variances[d,:,:] = np.diagflat(var[d,:])
        #print(variances)
        #exit()
        return(variances)

    def prior_update(self, t, r_list=None):

        """update the prior expectation & variance to be the posterior
        expectation and variances weighted by the run-length distribution"""
        self.pred_exp = np.sum((self.get_posterior_expectation(t) *
             (np.exp(self.joint_log_probabilities - self.model_log_evidence)
             [:,np.newaxis, np.newaxis])), axis=0)
        """cannot use get_posterior_variance here, because that returns the
        covariance matrix in the global format. I need it in the local (Naive)
        format here though"""
        posterior_variance = self.get_posterior_variance(t)
        self.pred_var = np.sum(posterior_variance *
             np.exp(self.joint_log_probabilities - self.model_log_evidence)
             [:,np.newaxis, np.newaxis], axis=0)
        """finally, update the prior mean and prior var"""
        self.prior_mean = self.pred_exp
        self.prior_var = self.pred_var


    def prior_log_density(self, y):
        """Computes the log-density of *y* under the prior. Unused."""
        y = y.reshape(self.S1, self.S2)
        prob = 1 - (1 / self.prior_beta)
        size = self.prior_alpha

        return (np.sum(stats.nbinom.logpmf(y, size, prob)))


    def trimmer(self, kept_run_lengths):
        """Trim the relevant quantities for the PG model"""
        self.joint_log_probabilities = self.joint_log_probabilities[kept_run_lengths]
        self.sums = self.sums[kept_run_lengths,:,:]
        self.retained_run_lengths = self.retained_run_lengths[kept_run_lengths]
        self.model_log_evidence = scipy.misc.logsumexp(self.joint_log_probabilities)
