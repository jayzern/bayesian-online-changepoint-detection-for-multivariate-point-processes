# Python 3.5.2 |Anaconda 3.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited: 15-03-2017
Author: Yannis Zachos (i.zachos@warwick.ac.uk)
Forked by: Jeremias Knoblauch (J.Knoblauch@warwick.ac.uk)

Description: Implements the Multinomial Dirichlet (MD) model.
"""

import numpy as np
from probability_model import ProbabilityModel
from scipy.special import gammaln
import scipy.misc
import math
from cp_probability_model import CpModel


class MDGPModel(ProbabilityModel):
    """The  Multinomial Dirichlet (MD) model.

    Statistical model:
        Let Y_t be a S1xS2 matrix of data, \LAMBDA a S1xS2 matrix of
        normalised intensities, and \ALPHA a S1xS2 matrix of prior hyperparameters.
        Let S1xS2 = k.
        $Y_t,\LAMBDA \sim \mathcal{Multinomial}(\LAMBDA;n) *
                    \mathcal{Dirichlet}(\LAMBDA|\ALPHA)$
        where n = sum_{i=1}^{k} Y_{it} for a fixed time t.
        In the remainder, \ALPHA are called 'prior_alphas'.

    Inputs at creation:
        S1, S2: int;
            give the dimensions of the lattice. The one-dimensional case
            corresponds to S1=1 and S2=1.
        prior_alphas: float numpy array; dimension S1xS2
            stores the parameters of the prior. Default is a vector of 1s.
        auto_prior_update: boolean
            Indicates whether the prior_mean and prior_var should be updated
            to the posterior expectation and posterior variance at each time
            step. Default is False.

    Internal Attributes:
        sums: float numpy array; dimension (t+1)xS1xS2 at time t.
            sum_{i=1}^{t} Y_{ij}
            gives the sum for each of the S1*S2 series for each possible
            run length r=0,1,...t-1, >t-1. at each time point t. Since this
            model corresponds to the special case where we assume that there
            is no spatial information, the sums numpy array is stored in a
            lattice again. (I.e., has two dimensions)
            This object stores at position r the posterior means corresponding
            to the run length of r=0,1,...,t at each time t.
        joint_log_probabilities: float numpy array; dimension t+1 at time t
            stores the joint log probabilities (y_{1:t}, r_t| q) for this
            model q and all potential run lengths r_t = 0,1,...t-1
        model_log_evidence: float;
            stores in a scalar the evidence for this model, i.e. the joint
            probability (y_{1:t}| q) for this model q.
    """


    def __init__(self, prior_alphas,
                 S1, S2, auto_prior_update=False):
        """Construct a naive MDGP probability model, providing a prior
        as ...

        NOTE: The Input must always be given as a lattices, i.e.
        we have to input evenjoint_log_probabilities the one-dimensional model in 2-dimensional form.
        If it is given in different form, simply convert it.
        """

        """initialize all the prior quantities"""
        self.prior_alphas = prior_alphas.reshape(S1,S2)
        self.auto_prior_update = auto_prior_update

        """initialize all the quantities that we compute through time"""
        self.S1, self.S2 = S1, S2
        self.sums = None
        self.joint_log_probabilities = None
        self.model_log_evidence = -np.inf
        self.retained_run_lengths = None

        self.has_lags = False
        self.hyperparameter_optimization = False
        self.generalized_bayes_rld = "kullback_leibler"



    def initialization(self, y, cp_model, model_prior):
        #print('----------------------INITIALISATION----------------------')
        """function which is only called ONCE, namely when the very first
        lattice of observations y is obtained/observed. Called from the
        'update_all_joint_log_probabilities' function within the Detector
        object containing the MDGPModel object.
        """

        """Get the new observation into the right shape"""
        y = y.reshape(self.S1, self.S2)

        """Evaluate the pmf corresponding to the observations, and
        multiply by the *model_prior* of the corresponding model. The model
        prior is stored by the Detector object, and passed from there to the
        ProbabilityModel object only when the joint probabilities
        are updated"""


        """ sum_process is the sum_{i=1}^{k} y_{it}, where each y_{it} are counts of process i at time t"""
        sum_process = sum(y)
        updated_alphas = y + self.prior_alphas

        """ Compute an ugly term included in the expression of the log marginal pmf.
        See derivation of the marginal."""


        """ Compute log marginal pmf - i.e. the log_evidence """

        sum_updated_alphas = sum(updated_alphas)
        positive = gammaln(sum_process+1) + gammaln(sum_updated_alphas) + sum(gammaln(y+updated_alphas))
        negative = sum(gammaln(y+1)) + sum(gammaln(updated_alphas)) + gammaln(sum_process+sum_updated_alphas)

        evidence = positive - negative


        """ evidence is a singleton array and therefore use the flattened
        version by selecting its first element."""

        self.model_log_evidence = np.log(model_prior) + evidence[0]

        """Get the hazard/probability of CP using *cp_model* as passed
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

        """Get log-probs for r_1=0 or r_1>0. Typically, we assume that the
        first observation corresponds to a CP (i.e. P(r_1 = 0) = 1),
        but this need not be the case in general."""
        r_equal_0 = (self.model_log_evidence +
                    np.log(cp_model.pmf_0(0) + epsilon))
        r_larger_0 = (self.model_log_evidence +
                    np.log(cp_model.pmf_0(1)+ epsilon))
        self.joint_log_probabilities = np.array([r_equal_0, r_larger_0])
        self.retained_run_lengths = np.array([0,0])

        """Update the sufficient statistics.
        For the sum, use the  same entry as sufficient statistic for both,
        r=0 and r>0, because we have no observation at t=0."""
        self.sums = np.array([y,y])


    def evaluate_predictive_log_distribution(self, y, t):
        #print('----------------PREDICTIVE LOG DIST----------------')
        """Returns the log densities of *y* using the predictive posteriors
        for all possible run-lengths r=0,1,...,t-1,>t-1 as currently stored by
        virtue of the sufficient statistics.

        See derivations for the expression of the posterior predictive.

        The corresponding density is computed for all run-lengths and
        returned in a np array"""

        """Get the new observation into the right shape"""
        y = y.reshape(self.S1, self.S2)

        """ sum_process is the sum_{i=1}^{k} y_{t,i}, where each y_{t,i} are counts of process i at time t"""
        sum_process = sum(y)
        updated_alphas = self.prior_alphas + self.sums
        sum_updated_alphas = np.sum(updated_alphas,axis=1)

#         print('t',t)
#         print('y',y.shape)
#         print('sum_process',sum_process.shape)
#         print('updated_alphas',updated_alphas.shape)
#         print('sum_updated_alphas',sum_updated_alphas.shape)
#         print('y+updated_alphas',(y+updated_alphas).shape)
#
#         print('gammaln(sum_process+1)',gammaln(sum_process+1).shape)
#         print('gammaln(sum_updated_alphas)',gammaln(sum_updated_alphas).shape)
#         print('np.sum(gammaln(y+updated_alphas))',np.sum(gammaln(y+updated_alphas),axis=1).shape)
#         print('np.sum(gammaln(y+1))',np.sum(gammaln(y+1),axis=0).shape)
#         print('np.sum(gammaln(updated_alphas))',np.sum(gammaln(updated_alphas),axis=1).shape)
#         print('gammaln(sum_process+sum_updated_alphas)',gammaln(sum_process+sum_updated_alphas).shape)

        positive = gammaln(sum_process+1) + gammaln(sum_updated_alphas) + np.sum(gammaln(y+updated_alphas),axis=1)
        negative = np.sum(gammaln(y+1),axis=0) + np.sum(gammaln(updated_alphas),axis=1) + gammaln(sum_process+sum_updated_alphas)
        #print('positive',positive)
        #print('negative',negative.shape)
        difference = positive - negative
        #print('difference', difference)

        """" Flatten the posterior predictive to match the dimensions
        of the corresponding object in the detector"""
        #print('posterior_predictive',np.ndarray.flatten(difference).shape)

        #print(np.ndarray.flatten(difference))
        return np.ndarray.flatten(difference)

    def evaluate_log_prior_predictive(self, y, t):
        #print('----------------LOG PRIOR PREDICTIVE----------------')
        """Returns the prior log density of the predictive distribution
        for all possible run-lengths r=0,1,...,t-1,>t-1.
        This is the case when no other data has been processed.
        First, reshape the data as always.
        """
        y = y.reshape(self.S1, self.S2)

        """ sum_process is the sum_{i=1}^{k} y_{i,t}, where each y_{i,t} are counts of process i at time t"""
        sum_process = sum(y)
        updated_alphas = self.prior_alphas + y
        sum_updated_alphas = sum(updated_alphas)


        """ Compute the prior predictive in the same fashion as the posterior
        predictive but without using updated quantities, such as model parameters
        and sufficient statistics."""

        positive = gammaln(sum_process+1) + gammaln(sum_updated_alphas) + sum(gammaln(y+updated_alphas))
        negative = sum(gammaln(y+1)) + sum(gammaln(updated_alphas)) + gammaln(sum_process+sum_updated_alphas)
        difference = positive - negative

        return np.ndarray.flatten(difference)

    def update_predictive_distributions(self, y, t, r_evaluations = None):
        #print('----------------UPDATE PREDICTIVE DIST----------------')
        """Takes the next observation, *y*, at time *t* and updates the
        sufficient statistics/means corresponding to all potential
        run-lengths r=0,1,...,t-1,>t-1.

        Since the two entries corresponding to r=t-1 and r>t-1 will always
        have the same sufficient statistic in this model, we only update the
        entries r=0,1,...,t-1 and then double the last one.
        """

        """STEP 0: Get the new observation into the right shape"""
        y = y.reshape(self.S1, self.S2)

        """STEP 1: SUMS
        The sums are updated.
        Note that this requires the knowledge of the true posterior
        variances.
        Update the sums from t-r to t to the sums from t-r to
        t+1 and add the t+1 th observation as the sum for r=0."""
        fac = self.retained_run_lengths[:,np.newaxis, np.newaxis]
        self.sums += y*fac # for r > 0
        self.sums = np.insert(self.sums, 0, y, axis=0) # for r = 0

        """STEP 2: Update retained_run_lengths in two steps: First by advancing
        the run-lengths, second by adding the 0-run-length from this run."""
        self.retained_run_lengths += 1
        self.retained_run_lengths = np.insert(self.retained_run_lengths, 0, 0)
        #print('self.retained_run_lengths.shape',self.retained_run_lengths.shape)


    def get_posterior_expectation(self, t, r_list=None):
        #print('----------------POSTERIOR EXPECTATION----------------')

        """ NOTE: THE POSTERIOR EXPECTATION OF THE PARAMETERS IS BEING RETURNED"""

        """get the predicted value/expectation from the current posteriors
        at time point t, for all possible run-lengths"""

        """get the weights for the prior. Last two entries of w are the same
        by the usual r=t-1 and r>t-1 logic."""
        #print('t',t,'prior_alpha',self.prior_alpha.shape,'sums',self.sums.shape)
        num = self.prior_alphas + self.sums
        den = np.sum(num)

        if den == np.zeros(den.shape):
            den += 0.005
        output = num/den

        #print(output)
        #print('get_posterior_expectation.shape',output.shape)
        #print('get_posterior_expectation',output[:,:,0])
        return (output)


    def get_posterior_variance(self, t, r_list=None):
        #print('----------------POSTERIOR VARIANCE----------------')
        """ NOTE: Needs to be debugged"""

        """get the predicted variance from the current posteriors at
        time point t, for all possible run-lengths."""

        """STEP 0: store how many variance matrices you have to return"""
        run_length_num = self.retained_run_lengths.shape[0]

        """STEP 1: Flatten each S1xS2 matrix into an S1*S2 vector"""
        fac1 = self.prior_alphas + self.sums
        fac2 = np.sum(fac1,axis=1)
        fac3 = (fac2**2 * (fac2+1))[:,:,np.newaxis]
        fac2 = fac2[:,:,np.newaxis]

        #print('posterior_variance_fac1',fac1[t,:,:])
        #print('posterior_variance_fac2',fac2)
        #print('fac1',fac1)
        #print('fac1.shape',fac1.shape)
        #print('fac2',fac2)
        #print('fac2.shape',fac2.shape)
        #print('sum(np.ndarray.flatten(fac1[t]))',sum(np.ndarray.flatten(fac1)))
        #print('fac3',fac3)
        #print('fac3.shape',fac3.shape)

        covariance = np.zeros((run_length_num,self.S1*self.S2,self.S1*self.S2))
        #print('product',-np.transpose(fac1, (0,2,1))*fac1)
        #print('product',(-np.transpose(fac1, (0,2,1))*fac1).shape)
        #print('fac3',fac3)
        #print('fac3',fac3.shape)
        #covariance = -fac1 * np.transpose(fac1,(0,2,1)) * (1/fac3)
        #print('covariance.shape',covariance.shape)
        #print('fac1.shape',fac1.shape)
        #print('covariance.shape',covariance.shape)
        #print('covariance BEFORE',covariance)
        for i in range(0,covariance.shape[0]):
            for j in range(0,fac1.shape[1]):
                for k in range(0,fac1.shape[2]):
                    #print('diagonal (',i,j,j,')',covariance[i,j,j])
                    #print('fac1[i,j,k]',fac1[i,j,k])
                    #covariance[i,j,j] += (fac1[i,j,k]*fac2)/fac3
                    #covariance[i,j,j] += (fac1[i,j,k]*fac2)/fac3
                    a = 0
        #print('covariance',covariance)
        #print('get_posterior_variance',covariance[t,:,:])
        #print('get_posterior_variance.shape',covariance.shape)
        #print(covariance[:,:,0])
        #print(covariance)
        #exit()
        return(covariance)

    def prior_update(self, t, r_list=None):
        #print('----------------PRIOR UPDATE----------------')
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
        #print('----------------PRIOR LOG DENSITY----------------')
        """ NOTE: HAS NOT BEEN DEBUGGED YET"""

        """Computes the log-density of *y* under the prior."""

        """Get the new observation into the right shape"""
        y = y.reshape(self.S1, self.S2)

        sum_process = sum(y)
        updated_alphas = y + self.prior_alphas

        num = math.log(math.gamma(sum_process + 1))
        den = self.multiSum(gammaln(y+1))

        term = np.sum(num-den)

        """ Compute log marginal pmf - i.e. the log_evidence """
        positive = gammaln(sum(self.prior_alphas)) + term + np.sum(gammaln(updated_alphas))
        negative = sum(gammaln(self.prior_alphas)) + gammaln(self.S1*self.S2*sum_process + sum(self.prior_alphas))
        #print('positive',positive)
        #print('negative',negative)
        evidence = positive - negative

        #param = self.prior_alpha + y
        #return (np.sum(stats.dirichlet.logpdf(y, param)))
        #print('prior_log_density',evidence)
        return evidence


    def trimmer(self, kept_run_lengths):
        #print(kept_run_lengths)
        """Trim the relevant quantities for the MDGP model"""
        self.joint_log_probabilities = (self.joint_log_probabilities[kept_run_lengths,:,:])
        #print('kept_run_lengths',kept_run_lengths)
        #print('self.sums.shape',self.sums.shape)
        self.sums = self.sums[kept_run_lengths,:,:]
        self.retained_run_lengths = (self.retained_run_lengths[kept_run_lengths])
        self.model_log_evidence = scipy.misc.logsumexp(self.joint_log_probabilities)
