# Python 3.6 |Anaconda 3.0 (64-bit)|
# -*- coding: utf-8 -*-

"""
Created: 20-1-2019
Author: Jay Zern Ng (J.Ng.3@warwick.ac.uk)
Description: Implements a univariate log-Gaussian cox process model
with Laplace approx. or Variational Inference to evaluate the posterior.
"""

import numpy as np
import gpflow
import GPy

from probability_model import ProbabilityModel

class LGCPModel(ProbabilityModel):
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
        self.refresh_rate = refresh_rate;

    def initialization(self, y, cp_model, model_prior):
        """function which is only called ONCE, namely when the very first
        lattice of observations y is obtained/observed. Called from the
        'update_all_joint_log_probabilities' function within the Detector
        object containing the LGCPModel object.
        """

        """Handle zero values for GPy error by adding a small value epsilon
        DEBUG: possibly find an alternative way to handle zero values"""
        if y == 0:
            epsilon = 0.01
            y = y + epsilon

        y = y.reshape(self.S1, self.S2)
        x = np.zeros(1).reshape(self.S1, self.S2)

        """Instantiate the GP using one of the three methods:
        laplace, variational_inference, sparse_variational_inference"""
        if self.inference_method == 'laplace':
            """Method 1: GPy Laplace Approximation with Poisson Likelihood.
            Evaluate the posterior by applying Newton's method to find the
            mode (not always optimal!), followed by a second order Taylor
            expansion around the mode."""

            """Specify kernel options here. """
            if self.custom_kernel is not None:
                """For unusual kernels, i.e. sum, products
                of periodic, linear etc, specify it within custom_kernel."""
                self.kernel = self.custom_kernel
                print("Using custom kernel:")
                print(self.kernel)
            else:
                """Standard RBF kernel and a bias.
                Use RBF for smooth latent functions.
                Bias for offsetting default mean at 1"""
                self.kernel = GPy.kern.RBF(
                    input_dim=1,
                    variance=self.prior_signal_variance,
                    lengthscale=self.prior_lengthscale
                )

                """Optional: specifying prior avoids numerical overflow,
                due to kernel not being PSD, hence cholesky decomp doesn't
                work"""
                self.kernel.set_prior(GPy.priors.Exponential(1), warning=False)
                self.kernel += GPy.kern.Bias(1)

                print("Using standard kernel")
                print(self.kernel)

            """Called once only."""
            laplace_inf = GPy.inference.latent_function_inference.Laplace()

            """Specify Poisson likelihood for cox processes."""
            self.likelihood = GPy.likelihoods.Poisson()

            """Choose mean function \mu(x;\theta)
            Note: Mean function does not work for non-Gaussian likelihood.
            Alternatively, choose a Bias kernel to offset mean at 1 or 0."""
            self.mean_function = None

            """Build the GP model via the above setting"""
            self.m = GPy.core.GP(
                x, y,
                kernel = self.kernel,
                inference_method = laplace_inf,
                likelihood = self.likelihood,
                mean_function = self.mean_function
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

            """Compile and build the model later. Order of this matters because
            certain GP settings cannot be changed in GPFlow otherwise."""
            with gpflow.defer_build():

                if self.custom_kernel is not None:
                    self.kernel = self.custom_kernel
                    print("Using custom kernel")
                    print(self.kernel)
                else:
                    self.kernel = gpflow.kernels.RBF(
                        input_dim=1,
                        variance=self.prior_signal_variance,
                        lengthscales=self.prior_lengthscale
                    )

                    """BUGS:
                    https://github.com/tensorflow/probability/issues/195
                    https://github.com/GPflow/GPflow/issues/78

                    Numerical overflow sometimes when you don't specify prior
                    Possible solution for kernel not positive semi definite
                    hence not invertible:
                    1) specify hyperparameter prior
                    2) add jitter or white kernel with small variance
                    3) logistic transform"""
                    self.kernel.variance.prior = gpflow.priors.Exponential(1)
                    self.kernel.lengthscales.prior = gpflow.priors.Exponential(1)

                    """Add Bias to offset default mean at 1"""
                    self.kernel += gpflow.kernels.Bias(input_dim=1)

                    print("Using standard kernel")
                    print(self.kernel)

                """Specify Poisson likelihood for cox processes.
                GPFlow: For use in a Log Gaussian Cox process (doubly stochastic
                model) where the rate function of an inhomogeneous Poisson
                process is given by a GP. The intractable likelihood can be
                approximated by gridding the space (into bins of size 'binsize')
                and using this Poisson likelihood."""
                self.likelihood = gpflow.likelihoods.Poisson()

                """Choose mean function \mu(x;\theta)
                Note: Mean function does not work for non-Gaussian likelihood.
                Alternatively, choose a Bias kernel to offset default mean
                at 0 or 1."""
                self.mean_function = None

                """Build the GP model using the above settings"""
                self.m = gpflow.models.VGP(
                    X=x, Y=y,
                    kern=self.kernel,
                    likelihood=self.likelihood,
                    mean_function=self.mean_function
                )

            """Compile and optimise"""
            self.m.compile()
            gpflow.train.ScipyOptimizer().minimize(self.m)

            """Compute p(y), the objective function of the model being optimised"""
            evidence = self.m.compute_log_likelihood()

        elif self.inference_method == 'sparse_variational_inference':
            """Method 3: GPFlow: Sparse Variational Inference
            Choose inducing points Z, |Z| = M s.t. M << N
            so the time complexity reduces to O(NM^2) instead of O(N^3)
            and space from O(N^2) to O(NM)"""

            """Compile and build the model later. Order of this matters because
            certain GP settings cannot be changed in GPFlow otherwise."""
            with gpflow.defer_build():

                if self.custom_kernel is not None:
                    self.kernel = self.custom_kernel
                    print("Using custom kernel")
                    print(self.kernel)
                else:
                    self.kernel = gpflow.kernels.RBF(
                        input_dim=1,
                        variance=self.prior_signal_variance,
                        lengthscales=self.prior_lengthscale
                    )

                    """Optional: specifying prior avoids numerical overflow,
                    due to kernel not being PSD, hence cholesky decomp doesn't work"""
                    self.kernel.variance.prior = gpflow.priors.Exponential(1)
                    self.kernel.lengthscales.prior = gpflow.priors.Exponential(1)

                    """Add Bias to offset default mean at 1"""
                    self.kernel += gpflow.kernels.Bias(input_dim=1)

                    print("Using standard kernel")
                    print(self.kernel)

                self.likelihood = gpflow.likelihoods.Poisson()

                """Build the GP model using the above settings"""
                self.m = gpflow.models.SVGP(
                    X=x, Y=y,
                    kern=self.kernel,
                    likelihood=self.likelihood,
                    Z=x
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
        Note: Following the same naming convention as the PG and MDGP model"""
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
        if y == 0:
            epsilon = 0.01
            y = y + epsilon

        y = y.reshape(self.S1, self.S2)

        """Method 1:
        Evaluate the predictive posteriors of *y* for all possible run-lengths"""
        factors = (self.retained_run_lengths + 1)
        factors = factors[:,np.newaxis]
        factors = factors.astype(np.float64)
        sums = np.full((factors.size,1),  y).astype(np.float64)

        """Compute log predictive density
        p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})
        Following the notation of Rasmussen & Williams (2006)"""
        if self.inference_method == 'laplace':
            pred_log_dist = self.m.log_predictive_density(factors, sums)
        elif self.inference_method == 'variational_inference':
            pred_log_dist = self.m.predict_density(factors, sums)
        elif self.inference_method == 'sparse_variational_inference':
            pred_log_dist = self.m.predict_density(factors, sums)

        """Evaluate predictive distribution as usual using the GP.
        Flip the array because run-lengths are stored in the opposite direction"""
        pred_log_dist = np.flip(pred_log_dist)

        """Reshape to [t,] size for every time t"""
        return pred_log_dist[:,0]


    def evaluate_log_prior_predictive(self, y, t):
        """Returns the prior log density of the predictive distribution
        for all possible run-lengths.
        This is the case when no other data has been processed.
        """

        """Handle zero values for GPy error by adding a small value epsilon
        DEBUG: possibly find an alternative way to handle zero values"""
        if y == 0:
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
            pred_log_prior_dist = self.m.log_predictive_density(factors, y)
        elif self.inference_method == 'variational_inference':
            pred_log_prior_dist = self.m.predict_density(factors, y)
        elif self.inference_method == 'sparse_variational_inference':
            pred_log_prior_dist = self.m.predict_density(factors, y)

        """Reshape to [1,] size for every time t"""
        return pred_log_prior_dist[:,0]

    def update_predictive_distributions(self, y, t, r_evaluations = None):
        """Takes the next observation, *y*, at time *t* and updates the
        the lgcp by refitting it and optimising it from its previous state.
        """

        """Handle zero values for GPy error by adding a small value epsilon
        DEBUG: possibly find an alternative way to handle zero values"""
        if y == 0:
            epsilon = 0.01
            y = y + epsilon

        y = y.reshape(self.S1, self.S2)

        """Inserts *y* at the most recent point of the GP."""
        self.sums = np.insert(self.sums, self.sums.size, y, axis=0)

        """Increment run lengths by one, then insert zero at the start"""
        self.retained_run_lengths = self.retained_run_lengths + 1
        self.retained_run_lengths = np.insert(self.retained_run_lengths, 0, 0)

        """Update observations in the Gaussian Process"""
        """Note:
        - Bug in VGP (https://github.com/GPflow/GPflow/issues/809)
          Cannot update X and Y, must rebuild from scratch.
        - Use self.count for batch optimisation."""

        if self.inference_method == 'laplace':
            x = self.retained_run_lengths[:,np.newaxis].astype(np.float64)
            y = self.sums[:,:,0]

            self.m.set_XY(x, y)
            self.m.optimize(max_iters=1000)

        elif self.inference_method == 'variational_inference':

            self.count = self.count + 1

            """Optimise model for every factor of the refresh_rate"""
            if (self.count%self.refresh_rate == 0):
                with gpflow.defer_build():
                    self.m = gpflow.models.VGP(
                        X=self.retained_run_lengths[:,np.newaxis].astype(np.float64),
                        Y=self.sums[:,:,0],
                        kern=self.kernel,
                        likelihood=self.likelihood,
                        mean_function=self.mean_function
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
                    Z = self.retained_run_lengths
                else:
                    Z = np.round(np.linspace(0, self.retained_run_lengths[-1], M))

                """Method 2: Randomly choose inducing points with length M = percent * N"""
                # percent = 0.2
                # M = int(len(np.array(self.retained_run_lengths)) * percent)
                # if M == 0:
                #     """If population size is small, then just take Z=X"""
                #     Z = self.retained_run_lengths
                # else:
                #     """When population size grows, sample the population times by the precent"""
                #     Z = np.array(self.retained_run_lengths)
                #     Z_index = np.random.choice(
                #         len(np.array(self.retained_run_lengths)),
                #         size=M,
                #         replace=False
                #     )
                #     Z = Z[Z_index]

                """Method 3: Naively set X=Z, hence M=N and O(NM^2) is really O(N^3)"""
                # Z = self.retained_run_lengths

                with gpflow.defer_build():
                    self.m = gpflow.models.SVGP(
                        X=self.retained_run_lengths[:,np.newaxis].astype(np.float64),
                        Y=self.sums[:,:,0],
                        kern=self.kernel,
                        likelihood=self.likelihood,
                        Z=Z[:,np.newaxis]
                    )

                """Compile and optimise"""
                self.m.compile()
                gpflow.train.ScipyOptimizer().minimize(self.m)

    def get_posterior_expectation(self, t, r_list=None):
        """get the predicted value/expectation from the current posteriors
        at time point t, for all possible run-lengths"""

        if self.inference_method == 'laplace':
            post_mean, post_var = self.m._raw_predict(
                self.retained_run_lengths[:,np.newaxis])
        elif self.inference_method == 'variational_inference':
            post_mean, post_var = self.m.predict_y(
                self.retained_run_lengths[:,np.newaxis])
        elif self.inference_method == 'sparse_variational_inference':
            post_mean, post_var = self.m.predict_y(
                self.retained_run_lengths[:,np.newaxis])

        return post_mean[:,np.newaxis]

    def get_posterior_variance(self, t, r_list=None):
        """get the predicted variance from the current posteriors at
        time point t, for all possible run-lengths.
        """

        if self.inference_method == 'laplace':
            post_mean, post_var = self.m._raw_predict(
                self.retained_run_lengths[:,np.newaxis])
        elif self.inference_method == 'variational_inference':
            post_mean, post_var = self.m.predict_y(
                self.retained_run_lengths[:,np.newaxis])
        elif self.inference_method == 'sparse_variational_inference':
            post_mean, post_var = self.m.predict_y(
                self.retained_run_lengths[:,np.newaxis])

        return post_var[:,np.newaxis]

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
                    self.m.kern.kernels[0].variance.value)
                self.prior_lengthscale = float(
                    self.m.kern.kernels[0].lengthscales.value)
        elif self.inference_method == 'sparse_variational_inference':
            if self.custom_kernel is None:
                self.prior_signal_variance = float(
                    self.m.kern.kernels[0].variance.value)
                self.prior_lengthscale = float(
                    self.m.kern.kernels[0].lengthscales.value)

    def prior_log_density(self, y):
        """Unused. Computes the log-density of *y* under the prior."""

        if self.inference_method == 'laplace':
            """Note: Need to place a prior over the variance for this to work"""
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
