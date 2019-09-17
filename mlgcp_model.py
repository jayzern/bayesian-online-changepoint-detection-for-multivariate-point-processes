# Python 3.6 |Anaconda 3.0 (64-bit)|
# -*- coding: utf-8 -*-

"""
Created: 25-2-2019
Author: Jay Zern Ng (J.Ng.3@warwick.ac.uk)
Description: Implements a multivariate log-Gaussian cox process model
using the intrinsic model of coregionalization (ICM) for multi-output GPs.
"""

import numpy as np
import gpflow
import GPy

from probability_model import ProbabilityModel

class mLGCPModel(ProbabilityModel):

    """
    Statistical model:
        Intensity \lambda is an inhomogeneous PP driven by a GP.
        f(x) ~ GP(\mu(x), k(x,x'))
        y_t | f(x) ~ PP(exp(f(x)))

    Inputs at creation:
        S1, S2: int;
            give the dimensions of the lattice. The one-dimensional case
            corresponds to S1=1 and S2=1.
        auto_prior_update: boolean
            Indicates whether the prior_mean and prior_var should be updated
            to the posterior expectation and posterior variance at each time
            step. Default is False.

    Inputs for model:
        prior_signal_variance: float;
            stores prior variance hyperparameter of kernel. Default is 1.
        prior_lengthscale: float;
            stores prior lengthscale hyperparameter of kernel. Default is 1.
        custom_kernel: GPFlow/GPy kernel class;
            overwrites the inbuilt kernel.
        inference_method: string;
            specify inference type here, i.e. laplace, variational_inference
            or sparse_variational_inference.
        refresh_rate: int;
            Used for batch optimisation in VI/SVI.
        M_pseudo_input_size: int;
            size of inducing points |Z| = M

    Internal Attributes:
        sums: float numpy array; dimension S1xS2x(t+1) at time t
            stores y values for each of the S1*S2 series for each possible
            run length r=0,1,...t-1, >t-1. at each time point t.
        joint_log_probabilities: float numpy array; dimension t+1 at time t
            stores the joint log probabilities (y_{1:t}, r_t| q) for this
            model q and all potential run lengths r_t = 0,1,...t-1
        model_log_evidence: float;
            stores in a scalar the evidence for this model, i.e. the joint
            probability (y_{1:t}| q) for this model q.
    """

    def __init__(self,
                prior_signal_variance,
                prior_lengthscale,
                custom_kernel,
                inference_method,
                refresh_rate,
                S1,
                S2,
                M_pseudo_input_size=10,
                auto_prior_update=False
                ):

        """Initialize prior quantities for kernel"""
        self.prior_signal_variance = prior_signal_variance
        self.prior_lengthscale = prior_lengthscale
        self.auto_prior_update = auto_prior_update

        """Initialize variables for Gaussian Process and inference method"""
        self.m = None
        self.kernel = None
        self.custom_kernel = custom_kernel
        self.likelihood = None
        self.mean_function = None
        self.inference_method = inference_method
        self.M = M_pseudo_input_size

        """Initialize all the quantities that we compute through time"""
        self.S1, self.S2 = S1, S2
        self.sums = None
        self.joint_log_probabilities = 1
        self.model_log_evidence = -np.inf
        self.retained_run_lengths = None

        """Initalize settings for detector object"""
        self.generalized_bayes_rld = "kullback_leibler"
        self.has_lags = False
        self.hyperparameter_optimization = False

        """Initialize count and refresh rate for batch updating"""
        self.count = 0
        self.refresh_rate = refresh_rate


    def initialization(self, y, cp_model, model_prior):
        """function which is only called ONCE, namely when the very first
        lattice of observations y is obtained/observed. Called from the
        'update_all_joint_log_probabilities' function within the Detector
        object containing the LGCPModel object.
        """

        """Handle zero values for GPy error by adding a small value epsilon
        DEBUG: possibly find an alternative way to handle zero values"""
        if (y==0).any():
            epsilon = 0.01
            y = y + epsilon

        y = y.reshape(self.S1, self.S2)

        """Use y_init, x_init naming convention because we want to append
        the original sufficient statistics *y* later."""
        y_init = y.copy()

        """Reshape to [t * output_dim,2].
        Column 0 is the time.
        Columns 1 is the dimension."""
        x_init = np.hstack((
            np.zeros(self.S1)[:,np.newaxis], np.arange(self.S1)[:,np.newaxis]))

        """Instantiate the GP using one of the three methods:
        laplace, variational_inference, sparse_variational_inference"""
        if self.inference_method == 'laplace':
            """Method 1: GPy Laplace Approximation with Poisson Likelihood.
            Evaluate the posterior by applying Newton's method to find the
            mode (not always optimal!), followed by a second order Taylor
            expansion around the mode."""

            if self.custom_kernel is not None:
                self.kernel = self.custom_kernel
                print("Using custom kernel:")
                print(self.kernel)
            else:
                rbf = GPy.kern.RBF(
                    input_dim=1,
                    variance=self.prior_signal_variance,
                    lengthscale=self.prior_lengthscale
                )
                rbf = rbf ** GPy.kern.Coregionalize(1, output_dim=self.S1)
                """BUGS:
                https://github.com/tensorflow/probability/issues/195
                https://github.com/GPflow/GPflow/issues/78

                Numerical overflow sometimes when you don't specify prior
                Possible solution for kernel not positive semi definite
                hence not invertible:
                1) specify hyperparameter prior
                2) add jitter or white kernel with small variance
                3) logistic transform"""
                rbf.set_prior(GPy.priors.Exponential(1), warning=False)
                bias = GPy.kern.Bias(1)
                bias = bias ** GPy.kern.Coregionalize(1, output_dim=self.S1)
                self.kernel = rbf + bias
                print("Using standard RBF kernel")
                print(self.kernel)

            """Called once only."""
            laplace_inf = GPy.inference.latent_function_inference.Laplace()

            """Specify Poisson likelihood for cox processes.
            Note: Exponential term is already implemented here,
            i.e. exp(GP(\mu(x), k(x,x')))"""
            self.likelihood = GPy.likelihoods.Poisson()

            """Build the GP model via the above setting"""
            self.m = GPy.core.GP(
                X=x_init, Y=y_init,
                kernel=self.kernel,
                inference_method=laplace_inf,
                likelihood=self.likelihood
            )

            self.m.optimize()

            """Compute p(y), the objective function of the model being optimised"""
            evidence = self.m.log_likelihood()

        elif self.inference_method == 'variational_inference':
            """Method 2: GPFlow Variational Inference for posterior.
            Minimize the KL-divergence between the the true and approximated
            posterior. This is equivalent to maximising the Evidence Lower
            Bound (ELBO), because not all log terms in the posterior are
            tractable. This implementation is equivalent to svgp with X=Z,
            but is more efficient."""

            """First column is y. Second column is the dimension label."""
            y_init = np.hstack((y_init,np.arange(self.S1)[:,np.newaxis]))

            with gpflow.defer_build():
                """BUGS:
                https://github.com/tensorflow/probability/issues/195
                https://github.com/GPflow/GPflow/issues/78

                Numerical overflow sometimes when you don't specify prior
                Possible solution for kernel not positive semi definite
                hence not invertible:
                1) specify hyperparameter prior
                2) add jitter or white kernel with small variance
                3) logistic transform"""

                if self.custom_kernel is not None:
                    self.kernel = self.custom_kernel
                    print("Using custom kernel")
                    print(self.kernel)
                else:
                    rbf = gpflow.kernels.RBF(
                        input_dim=1,
                        variance=self.prior_signal_variance,
                        lengthscales=self.prior_lengthscale,
                        active_dims=[0]
                    )
                    bias = gpflow.kernels.Bias(
                        input_dim=1,
                        active_dims=[0]
                    )
                    coreg = gpflow.kernels.Coregion(
                        1,
                        output_dim=self.S1,
                        rank=1,
                        active_dims=[1]
                    )

                    """Optional: specifying prior avoids numerical overflow,
                    due to kernel not being PSD, hence cholesky decomp doesn't work"""
                    #rbf.variance.prior = gpflow.priors.Exponential(1)
                    #rbf.lengthscales.prior = gpflow.priors.Exponential(1)

                    rbf = rbf * coreg
                    bias = bias * coreg
                    self.kernel = rbf + bias
                    print("Using standard RBF kernel")
                    print(self.kernel)

                """Specify Poisson likelihood for cox processes.
                GPFlow: For use in a Log Gaussian Cox process (doubly stochastic
                model) where the rate function of an inhomogeneous Poisson
                process is given by a GP. The intractable likelihood can be
                approximated by gridding the space (into bins of size 'binsize')
                and using this Poisson likelihood."""
                self.likelihood = gpflow.likelihoods.Poisson()

                """Build the GP model using the above settings"""
                self.m = gpflow.models.VGP(
                    X=x_init, Y=y_init,
                    kern=self.kernel,
                    likelihood=self.likelihood
                )

            """Compile and optimise"""
            self.m.compile()
            gpflow.train.ScipyOptimizer().minimize(self.m)

            """Compute p(y), the objective function of the model being optimised"""
            evidence = self.m.compute_log_likelihood()

        elif self.inference_method == 'sparse_variational_inference':
            """Method 3: GPFlow: Sparse Variational Inference for posterior.
            Choose inducing points Z, |Z| = M s.t. M << N
            so the time complexity reduces to O(NM^2) instead of O(N^3)
            and space from O(N^2) to O(NM)"""

            """First column is y. Second column is the dimension label."""
            y_init = np.hstack((y_init,np.arange(self.S1)[:,np.newaxis]))

            with gpflow.defer_build():

                if self.custom_kernel is not None:
                    self.kernel = self.custom_kernel
                    print("Using custom kernel")
                    print(self.kernel)
                else:
                    rbf = gpflow.kernels.RBF(
                        input_dim=1,
                        variance=self.prior_signal_variance,
                        lengthscales=self.prior_lengthscale,
                        active_dims=[0]
                    )
                    bias = gpflow.kernels.Bias(
                        input_dim=1,
                        active_dims=[0]
                    )
                    coreg = gpflow.kernels.Coregion(
                        1,
                        output_dim=self.S1,
                        rank=1,
                        active_dims=[1]
                    )

                    """Optional: specifying prior avoids numerical overflow,
                    due to kernel not being PSD, hence cholesky decomp doesn't work"""
                    #rbf.variance.prior = gpflow.priors.Exponential(1)
                    #rbf.lengthscales.prior = gpflow.priors.Exponential(1)

                    rbf = rbf * coreg
                    bias = bias * coreg
                    self.kernel = rbf + bias
                    print("Using standard RBF kernel")
                    print(self.kernel)

                self.likelihood = gpflow.likelihoods.Poisson()

                """Build the GP model using the above settings"""
                self.m = gpflow.models.SVGP(
                    X=x_init, Y=y_init,
                    kern=self.kernel,
                    likelihood=self.likelihood,
                    Z=x_init
                )

            """Compile and optimise GP model"""
            self.m.compile()
            gpflow.train.ScipyOptimizer().minimize(self.m)

            """Compute p(y), the objective function of the model being optimised"""
            evidence = self.m.compute_log_likelihood()

        """Evidence is a singleton array and therefore use the flattened
        version by selecting its first element."""
        self.model_log_evidence = (np.log(model_prior) + evidence)

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

        """Update the sufficient statistics.
        Note: Following the naming convention from PG and MDGP"""
        self.retained_run_lengths = np.array([0,0])
        self.sums = np.array([y,y])

    def evaluate_predictive_log_distribution(self, y, t):
        """Returns the log densities of *y* using the predictive posteriors
        for all possible run-lengths r=0,1,...,t. Posterior is approximated
        depending on the choice of inference method.
        The corresponding density is computed for all run-lengths and
        returned in a np array"""

        """Handle zero values for GPy error by adding a small value epsilon
        DEBUG: possibly find an alternative way to handle zero values"""
        if (y==0).any():
            epsilon = 0.01
            y = y + epsilon

        y = y.reshape(self.S1, self.S2)

        """Evaluate the predictive posteriors of *y* for all possible run-lengths"""
        factors = (self.retained_run_lengths + 1)
        factors = factors[:,np.newaxis]
        factors = factors.astype(np.float64)
        sums = np.full((factors.size,self.S1,1), y)

        """Compute log predictive density
        p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})
        Following the notation of Rasmussen & Williams (2006)"""
        if self.inference_method == 'laplace':

            """Reshape x and y to include dimensions in second column"""
            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors, np.full(factors.size, i)[:,np.newaxis]))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)

            y_list = []
            for i in range(self.S1):
                y_list.append(sums[:,i])
            y = np.vstack(y_list)

            pred_log_dist = self.m.log_predictive_density(x, y)

        elif self.inference_method == 'variational_inference':

            """Reshape x and y to include dimensions in second column"""
            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors, np.full(factors.size, i)[:,np.newaxis]))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)

            y_list = []
            for i in range(self.S1):
                y_at_index_i = np.hstack((
                    sums[:,i], np.full(len(sums[:,i]), i)[:,np.newaxis]))
                y_list.append(y_at_index_i)
            y = np.vstack(y_list)

            pred_log_dist = self.m.predict_density(x, y)

            """Take the first column only. Reshape."""
            pred_log_dist = pred_log_dist[:,0]
            pred_log_dist = pred_log_dist[:,np.newaxis]

        elif self.inference_method == 'sparse_variational_inference':

            """Reshape x and y to include dimensions in second column"""
            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors, np.full(factors.size, i)[:,np.newaxis]))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)

            y_list = []
            for i in range(self.S1):
                y_at_index_i = np.hstack((
                    sums[:,i], np.full(len(sums[:,i]), i)[:,np.newaxis]))
                y_list.append(y_at_index_i)
            y = np.vstack(y_list)

            pred_log_dist = self.m.predict_density(x, y)

            """Take the first column only. Reshape."""
            pred_log_dist = pred_log_dist[:,0]
            pred_log_dist = pred_log_dist[:,np.newaxis]

        """Split by self.S1 dimensions. Add them together.
        Flip it because run-lengths are stored in the opposite direction"""
        pred_log_dist_list = np.split(pred_log_dist, self.S1)
        pred_log_dist = sum(pred_log_dist_list)
        pred_log_dist = np.flip(pred_log_dist)
        pred_log_dist = pred_log_dist[:,0]

        return pred_log_dist

    def evaluate_log_prior_predictive(self, y, t):
        """Returns the prior log density of the predictive distribution
        for all possible run-lengths.
        This is the case when no other data has been processed.
        """

        """Handle zero values for GPy error by adding a small value epsilon
        DEBUG: possibly find an alternative way to handle zero values"""
        if (y==0).any():
            epsilon = 0.01
            y = y + epsilon

        """First, reshape the data as always."""
        y = y.reshape(self.S1, self.S2)

        """Evaluate pred distr at the most recent point of
        the Gaussian Process at time t."""
        factors = (self.retained_run_lengths + 1)
        factors = factors[-1:, np.newaxis]
        factors = factors.astype(np.float64)

        """Compute log predictive density
        p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})
        Following the notation of Rasmussen & Williams (2006)"""
        if self.inference_method == 'laplace':

            """Reshape x and y to include dimensions in second column"""
            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors, np.full(factors.size, i)[:,np.newaxis]))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)

            y_list = []
            for i in range(self.S1):
                y_list.append(y[i])
            y = np.vstack(y_list)

            pred_log_prior_dist = self.m.log_predictive_density(x, y)

        elif self.inference_method == 'variational_inference':

            """Reshape x and y to include dimensions in second column"""
            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors,
                    np.full(factors.size, i)[:,np.newaxis].astype(np.float64)))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)

            y = np.hstack((y,np.arange(self.S1)[:,np.newaxis]))

            pred_log_prior_dist = self.m.predict_density(x, y)

            """Take the first column only. Reshape. """
            pred_log_prior_dist = pred_log_prior_dist[:,0]
            pred_log_prior_dist = pred_log_prior_dist[:,np.newaxis]

        elif self.inference_method == 'sparse_variational_inference':

            """Reshape x and y to include dimensions in second column"""
            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors,
                    np.full(factors.size, i)[:,np.newaxis].astype(np.float64)))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)

            y = np.hstack((y,np.arange(self.S1)[:,np.newaxis]))

            pred_log_prior_dist = self.m.predict_density(x, y)

            """Take the first column only. Reshape. """
            pred_log_prior_dist = pred_log_prior_dist[:,0]
            pred_log_prior_dist = pred_log_prior_dist[:,np.newaxis]

        """Split by self.S1 dimensions and take sum."""
        pred_log_prior_dist_list = np.split(pred_log_prior_dist, self.S1)
        pred_log_prior_dist = sum(pred_log_prior_dist_list)

        return pred_log_prior_dist

    def update_predictive_distributions(self, y, t, r_evaluations = None):
        """Takes the next observation, *y*, at time *t* and updates the
        the lgcp by refitting it and optimising it from its previous state.
        """

        """Handle zero values for GPy error by adding a small value epsilon
        DEBUG: possibly find an alternative way to handle zero values"""
        if (y==0).any():
            epsilon = 0.01
            y = y + epsilon

        y = y.reshape(self.S1, self.S2)

        """Inserts *y* at the most recent point of the GP."""
        self.sums = np.insert(self.sums, len(self.sums), y, axis=0)

        """Increment run lengths by one, then insert zero at the start"""
        self.retained_run_lengths = self.retained_run_lengths + 1
        self.retained_run_lengths = np.insert(self.retained_run_lengths, 0, 0)

        factors = self.retained_run_lengths[:,np.newaxis]

        if self.inference_method == 'laplace':

            """Reshape x and y to include dimensions in second column"""
            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors, np.full(factors.size, i)[:,np.newaxis]))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)

            y_list = []
            for i in range(self.S1):
                y_list.append(self.sums[:,i])
            y = np.vstack(y_list)

            self.m.set_XY(x, y)
            self.m.optimize(max_iters=1000)

        elif self.inference_method == 'variational_inference':

            self.count = self.count + 1

            """Optimise model for every factor of the refresh_rate"""
            if (self.count%self.refresh_rate == 0):# or (self.count <= self.refresh_rate):

                """Reshape x and y to include dimensions in second column"""
                x_list = []
                for i in range(self.S1):
                    x_at_index_i = np.hstack((
                        factors, np.full(factors.size, i)[:,np.newaxis]))
                    x_list.append(x_at_index_i)
                x = np.vstack(x_list).astype(np.float64)

                y_list = []
                for i in range(self.S1):
                    y_at_index_i = np.hstack((
                        self.sums[:,i],
                        np.full(len(self.sums[:,i]), i)[:,np.newaxis]))
                    y_list.append(y_at_index_i)
                y = np.vstack(y_list).astype(np.float64)

                with gpflow.defer_build():
                    self.m = gpflow.models.VGP(
                        X=x, Y=y,
                        kern=self.kernel,
                        likelihood=self.likelihood
                    )

                """Compile and optimise"""
                self.m.compile()
                gpflow.train.ScipyOptimizer().minimize(self.m)

        elif self.inference_method == 'sparse_variational_inference':

            self.count = self.count + 1

            """Optimise model for every factor of the refresh_rate"""
            if (self.count%self.refresh_rate == 0):

                """Method 1:Choose Z inducing points uniformly with length M."""
                M = self.M
                if int(len(np.array(self.retained_run_lengths))) < M:
                    """If population size is small, then just take Z=X"""
                    z = self.retained_run_lengths[:,np.newaxis]
                else:
                    z = np.round(np.linspace(
                        0, self.retained_run_lengths[-1], M))[:,np.newaxis]

                """Reshape x, y, z to include dimensions in second column"""
                x_list = []
                for i in range(self.S1):
                    x_at_index_i = np.hstack((
                        factors, np.full(factors.size, i)[:,np.newaxis]))
                    x_list.append(x_at_index_i)
                x = np.vstack(x_list).astype(np.float64)

                y_list = []
                for i in range(self.S1):
                    y_at_index_i = np.hstack((
                        self.sums[:,i],
                        np.full(len(self.sums[:,i]), i)[:,np.newaxis]))
                    y_list.append(y_at_index_i)
                y = np.vstack(y_list).astype(np.float64)

                z_list = []
                for i in range(self.S1):
                    z_at_index_i = np.hstack((
                        z, np.full(z.size, i)[:,np.newaxis]))
                    z_list.append(z_at_index_i)
                z = np.vstack(z_list).astype(np.float64)

                with gpflow.defer_build():
                    self.m = gpflow.models.SVGP(
                        X=x, Y=y,
                        kern=self.kernel,
                        likelihood=self.likelihood,
                        Z=z
                    )

                """Compile and optimise"""
                self.m.compile()
                gpflow.train.ScipyOptimizer().minimize(self.m)


    def get_posterior_expectation(self, t, r_list=None):
        """get the predicted value/expectation from the current posteriors
        at time point t, for all possible run-lengths"""

        factors = self.retained_run_lengths[:,np.newaxis]

        if self.inference_method == 'laplace':

            """Reshape x and y to include dimensions in second column"""
            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors, np.full(factors.size, i)[:,np.newaxis]))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)

            post_mean, post_var = self.m._raw_predict(x)


        elif self.inference_method == 'variational_inference':

            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors, np.full(factors.size, i)[:,np.newaxis]))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)

            post_mean, post_var = self.m.predict_y(x)

            """Take first column and reshape."""
            post_mean = post_mean[:,0]
            post_mean = post_mean[:,np.newaxis]

        elif self.inference_method == 'sparse_variational_inference':

            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors, np.full(factors.size, i)[:,np.newaxis]))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)

            post_mean, post_var = self.m.predict_y(x)

            """Take first column and reshape."""
            post_mean = post_mean[:,0]
            post_mean = post_mean[:,np.newaxis]


        """Split by self.S1 dimensions and add them together"""
        post_mean_list= np.split(post_mean, self.S1)
        post_mean = np.hstack(post_mean_list)
        post_mean = post_mean[:,:,np.newaxis]

        return post_mean

    def get_posterior_variance(self, t, r_list=None):
        """get the predicted variance from the current posteriors at
        time point t, for all possible run-lengths.
        """

        factors = self.retained_run_lengths[:,np.newaxis]

        if self.inference_method == 'laplace':

            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors, np.full(factors.size, i)[:,np.newaxis]))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)

            """Get full covariance matrix for all dimensions for all time"""
            post_mean, covariance = self.m._raw_predict(x,full_cov=True)
            #print(covariance.shape)

        elif self.inference_method == 'variational_inference':

            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors, np.full(factors.size, i)[:,np.newaxis]))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)
            post_mean, covariance = self.m.predict_f_full_cov(x)
            covariance = covariance[0]

        elif self.inference_method == 'sparse_variational_inference':

            x_list = []
            for i in range(self.S1):
                x_at_index_i = np.hstack((
                    factors, np.full(factors.size, i)[:,np.newaxis]))
                x_list.append(x_at_index_i)
            x = np.vstack(x_list)
            post_mean, covariance = self.m.predict_f_full_cov(x)
            covariance = covariance[0]

        """Get run length size"""
        run_length_num = self.retained_run_lengths.shape[0]

        """Loop all run-lengths. Get corresponding cov(j,k) values per time index"""
        placeholder = np.zeros((run_length_num,self.S1*self.S2,self.S1*self.S2))
        for i in range(0,placeholder.shape[0]): # for loop all run-lengths
            cov_at_time_i = np.zeros((self.S1*self.S2,self.S1*self.S2))
            for j in range(self.S1):
                for k in range(self.S1):
                    cov_at_time_i[j,k] = covariance[i + j * self.S1, i + k * self.S1]
            placeholder[i,:,:] = cov_at_time_i
        covariance = placeholder

        return covariance

    def prior_update(self, t, r_list=None):
        """Update the prior variance and lengthscale by setting the optimised
        kernel hyperparameters to be the prior for the next update."""
        if self.inference_method == 'laplace':
            """Skip. No need to update signal variance and lengthscaleself.
            This is already implemented sequentially"""
            pass
        elif self.inference_method == 'variational_inference':
            if self.custom_kernel is None:
                self.prior_signal_variance = float(
                    self.m.kern.kernels[0].kernels[0].variance.value)
                self.prior_lengthscale = float(
                    self.m.kern.kernels[0].kernels[0].lengthscales.value)
        elif self.inference_method == 'sparse_variational_inference':
            if self.custom_kernel is None:
                self.prior_signal_variance = float(
                    self.m.kern.kernels[0].kernels[0].variance.value)
                self.prior_lengthscale = float(
                    self.m.kern.kernels[0].kernels[0].lengthscales.value)

    def prior_log_density(self, y):
        """Unused. Computes the log-density of *y* under the prior."""

        if self.inference_method == 'laplace':
            """Note: Need to place a prior over hyperparameters for this to work"""
            log_prior = self.m.log_prior()
        elif self.inference_method == 'variational_inference':
            log_prior = self.m.compute_log_prior()
        elif self.inference_method == 'sparse_variational_inference':
            log_prior = self.m.compute_log_prior()

        return log_prior

    def trimmer(self, kept_run_lengths):
        """Trim the relevant quantities"""
        self.joint_log_probabilities = self.joint_log_probabilities[kept_run_lengths]

        """Flip the kept_run_lengths"""
        self.sums = self.sums[np.flip(kept_run_lengths)]
        self.retained_run_lengths = self.retained_run_lengths[np.flip(kept_run_lengths)]

        """Update p(y), the objective function of the model being optimised"""
        if self.inference_method == 'laplace':
            self.model_log_evidence = self.m.log_likelihood()
        elif self.inference_method == 'variational_inference':
            self.model_log_evidence = self.m.compute_log_likelihood()
        elif self.inference_method == 'sparse_variational_inference':
            self.model_log_evidence = self.m.compute_log_likelihood()
