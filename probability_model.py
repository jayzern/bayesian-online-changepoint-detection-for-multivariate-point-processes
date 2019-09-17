# Python 3.5.2 |Anaconda 4.2.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited: 2017-09-13
Author: Jeremias Knoblauch (J.Knoblauch@warwick.ac.uk)
Forked by: Luke Shirley (L.Shirley@warwick.ac.uk)

Description: Implements the abstract class ProbabilityModel, which is the
parent class of all probability models used by Spatial BOCD. E.g., the
NaiveModel class has this abstract class as its parent and implements the
model where we treat each spatial location as independent from all others
(in a Gaussian setting)
"""

"""import all you need to build an abstract class"""
#from abc import ABCMeta, abstractmethod
import numpy as np
import scipy #.special #import logsumexp

class ProbabilityModel:
    #__metaclass__ = ABCMeta
    """An abstract class for all probability models that will live in the model
    universe. I.e., each explicit probability model (e.g., the naive model that
    does assume independence between locations) will inherit from this class.


    NOTE: I Should still look into superclass constructors!

    Abstract Methods:
        predictive_probabilities
        growth_probabilities
        evidence
        cp_probabilities
        run_length_distribution
        update_predictive_distributions
    """


    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS
    def evaluate_predictive_log_distribution(self, y,r):
        """Evaluate the predictive probability associated with run length
        *r*. Gives back ONE quantity reflecting the overall probability
        density of the current observation *y* across all spatial
        locations, given run length *r*"""
        pass

    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS FOR t==1!
    def initialization(self, y, cp_model, model_prior):
        """Initialize the joint_probabilities before the first iteration!"""
        pass

    def rescale_DPD_run_length_distribution(self, log_scale_new,
                                        rescaler_for_old_obs, t):
        """Rescales joint log probs"""
        if rescaler_for_old_obs is None:
            self.joint_log_probabilities = (self.joint_log_probabilities
                                            - log_scale_new)
        else:
            self.joint_log_probabilities[0] = (self.joint_log_probabilities[0]
                    - log_scale_new)
            self.joint_log_probabilities[1:] = (
                    self.joint_log_probabilities[1:] + rescaler_for_old_obs)
        self.model_log_evidence = scipy.misc.logsumexp(
                self.joint_log_probabilities)

    #SUPER CLASS IMPLEMENTATION
    #CALLED BY DETECTOR
    def update_joint_log_probabilities(
                self,
                y, t, cp_model, model_prior,
                log_model_posteriors,
                log_CP_evidence,
                log_model_posteriors_der = None,
                log_model_posteriors_der_sign = None,
                log_CP_evidence_der = None,
                log_CP_evidence_der_sign = None,
                do_general_bayesian_hyperparameter_optimization = False,
                disable_hyperparameter_optimization = False):
        #DEBUG: model_prior should not be needed anymore. Just put it into
        #       model_posterior, position 0.
        """Is called to update growth- and CP probabilities. That means it is
        called to obtain the joint probability for (y_{1:t}, r_t) for all
        possible run lengths r_t = 0,1,...,t-1.
        If t=1, a special case takes place that requires calling the
        function 'initialization' that gives us the first joint probs.
        The *model_prior* is the prior probability of the model. Unlike
        the np array in the Detector object, the passed argument here is
        only a scalar (!) corresponding to the correct entry of the np array
        in the detector for this particular model.

        NOTE: *joint_probabilities* will have size t at the
        beginning and size t+1 at the end of this function call.

        NOTE: Desirable to also implement SOR/SCR of Fearnhead & Liu (2007)
        in order for this to scale in constant time!

        NOTE: the derivatives (and signs thereof) are only needed if we do
        hyperparameter-optimization for alpha.
        """


        """STEP 1: evaluate predictive probabilities for all r.
        This means that for a given a model and the current time point,
        we return the evaluations of the pdf at time t conditional on all run
        lengths r=0,1,...t-1,>t-1 for y in log-form.
        """

        self.r0_log_prob = self.evaluate_log_prior_predictive(y,t)
        #predictive_log_probs = self.evaluate_predictive_log_distribution(y,t)
        self.one_step_ahead_predictive_log_probs = (
                self.evaluate_predictive_log_distribution(y,t))
        #DEBUG: Here, we store the predictive distros y_t|y_1:t-1,r_t-1,m_t-1
        #NOTE: we do not want the cp prob in here because that is a NEW
        """NOTE: Unclear if this needs to be treated separately from General
        Bayesian updating, since it is re-used in the gradient!
        ANSWER: It does NOT need to be treated separately. Instead, we need to
        get the gradient under generalized Bayesian updating and change
        that in the code for each model..."""
        #DEBUG: Parameter updating should be based on General Bayes, too if
        #       we base joint probability computation on it!

            #np.insert(predictive_log_probs.copy(), 0, r0_log_prob)

        #DEBUG: Here, compute joint log probs for alpha + eps and alpha - eps
        #       if required


        #DEBUG: Check out if General Bayes works at all
        #DEBUG: What about CP probability? not considered here.
        #DEBUG: Incorrect, since we have PB(y,r,m) = prior * exp(-sum(loss)),
        #       so taking logs give log(prior)  + -sum(loss),
        if self.generalized_bayes_rld == "power_divergence":
            integrals = self.get_log_integrals_power_divergence()
            """Note: If we have the power divergence loss, then our log of
            the loss should be computed from the predictive probabilities
            appropriately"""
            #print("self.one_step_ahead_predictive_log_probs", self.one_step_ahead_predictive_log_probs)
            #print("integrals", integrals)

            #DEBUG: Check if np.log(maxFloat) >= any value in integrals.
            max_val = np.log(np.finfo(np.float64).max)
            #Split integral term s.t. you set terms exceeding max_val to max_val
            #and leave the rest. Similarly for the one step ahead pred.


            """STEP 1: Correct the integral term by bounding"""
            integral_exceeds = np.where(integrals >= max_val)[0]
            integral_fine = np.where(integrals < max_val)[0]
            integral_exp = integrals

#            print("integrals", integrals)
#            print("integrals too large", integrals[integral_exceeds])
#            print("integrals fine", integrals[integral_fine])
#            print("integral exceeds", integral_exceeds)

            if len(integral_exceeds)>0:
                integral_exp[integral_exceeds] = (
                        min(1.0,(1.0/(self.alpha_rld+1.0)))*
                        (np.finfo(np.float64).max)
                )
            if len(integral_fine)>0:
                """Just do the standard thing"""
                integral_exp[integral_fine] = (
                        (1.0/(self.alpha_rld+1.0))*
                         np.exp(integrals[integral_fine])
                    )

            """STEP 2: Correct the predictive probs"""

            """STEP 2.1: Correct CP prob if necessary"""
            if self.r0_log_prob >= max_val:
                r0_log_prob_exp = (min(1.0, 1.0/self.alpha_rld) *
                                   (min(self.alpha_rld, 1) * max_val))
            else:
                r0_log_prob_exp = (1.0/self.alpha_rld)*np.exp(
                    self.r0_log_prob*self.alpha_rld)

            """STEP 2.2: Correct other probs"""
            step_ahead_exceeds = np.where(
                    self.one_step_ahead_predictive_log_probs >= max_val)[0]
            step_ahead_fine = np.where(
                    self.one_step_ahead_predictive_log_probs < max_val)[0]
            step_ahead_exp = self.one_step_ahead_predictive_log_probs.copy()
            if len(step_ahead_exceeds)>0:
                if self.alpha_rld < 1:
                    """If the max value will be downweighted by alpha"""
                    step_ahead_exp[step_ahead_exceeds] = (
                            min(1.0,1.0/(self.alpha_rld))*
                            (np.finfo(np.float64).max * self.alpha_rld)
                    )
                else:
                    """If the max value will be increased by alpha"""
                    step_ahead_exp[step_ahead_exceeds] = (
                            min(1.0,(1.0/(self.alpha_rld)))*
                            (np.finfo(np.float64).max)
                    )
            if len(step_ahead_fine)>0:
                """Just do the standard thing"""
                step_ahead_exp[step_ahead_fine] = (
                        ((1.0/(self.alpha_rld))*
                         np.exp(
                            self.one_step_ahead_predictive_log_probs[
                                    step_ahead_fine]*
                            self.alpha_rld)
                    ))


            #DEBUG: Original/naive/simple version (not numerically stable)
#            if True:
#                #NOTE: If you are going to use these, need to add .copy()
#                #       to integral and step_ahead_exp
#                self.one_step_ahead_predictive_log_loss_2 = ((1.0/self.alpha_rld)*
#                        np.power(np.exp(self.one_step_ahead_predictive_log_probs),
#                             self.alpha_rld) -
#                        (1.0/(self.alpha_rld+1.0))*np.exp(integrals[1:])
#                    )
#
#
#                self.r0_log_loss_2 = (1.0/self.alpha_rld*np.power(np.exp(
#                        self.r0_log_prob), self.alpha_rld)
#                    - (1.0/(self.alpha_rld+1.0))*np.exp(integrals[0]))

            self.one_step_ahead_predictive_log_loss = (
                    step_ahead_exp - integral_exp[1:])
            self.r0_log_loss = r0_log_prob_exp - integral_exp[0]

#            print("original growth", self.one_step_ahead_predictive_log_loss_2)
#            print("new growth", self.one_step_ahead_predictive_log_loss)
#            print("original cp", self.r0_log_loss_2)
#            print("new cp", self.r0_log_loss)
#            print("replaced growth & cp & integ", len(step_ahead_exceeds),
#                  (self.r0_log_prob >= max_val),  len(integral_exceeds))
#            print("original log prob component",
#                  ((1.0/self.alpha_rld)*
#                   np.power(np.exp(self.one_step_ahead_predictive_log_probs), self.alpha_rld)))
#            print("new log prob component",
#                  step_ahead_exp)
#            print("original integral component",
#                  ((1.0/(self.alpha_rld+1.0))*np.exp(integrals)))
#            print("new integral component",
#                  integral_exp)

            """NOTE: At this stage, we have computed the predictive densities,
            but still have the old joint probabilities. So this is the perfect
            time for updating alpha.
            UNCLEAR: Should we use the updated version in this iteration, or
                     only in the next one?"""
            #DEBUG: Need to put more inputs into update_joint_log_probs from
            #       Detector level.
            #DEBUG: Need to initialize the alpha-derivatives to make sure that
            #       first call does not fail
            #DEBUG: Make sure that we only do this from time t=2 onwards, i.e.
            #       after we have actually observed stuff! Need to
            #       pass in the relevant boolean!
            if (self.alpha_rld_learning and
                do_general_bayesian_hyperparameter_optimization and
                disable_hyperparameter_optimization is not True):
                """updates the log of the joint probabilities' derivatives
                with respect to alpha. probability_model level function"""
                self.update_alpha_derivatives(y, t,
                                 log_model_posteriors,
                                 log_model_posteriors_der,
                                 log_model_posteriors_der_sign,
                                 log_CP_evidence,
                                 log_CP_evidence_der,
                                 log_CP_evidence_der_sign,
                                 model_prior,
                                 cp_model)
                """Use the derivatives to update the value of alpha via
                stochastic gradient descent inside Detector later"""

        elif self.generalized_bayes_rld == "kullback_leibler":
            """Note: In the case of KL, the predictive density is exactly
            the loss function! So the log of the loss function is simply the
            log of the predictive density"""
            self.r0_log_loss = self.r0_log_prob
            self.one_step_ahead_predictive_log_loss = (
                    self.one_step_ahead_predictive_log_probs.copy())

        #print(predictive_log_probs.shape)


        #DEBUG: evaluated log prob for r=0 is not in here! (Is only needed for
        #       the CP probability though)
        #helper_log_probabilities = (self.joint_log_probabilities +
        #                           predictive_log_probs)

        #print("predictive_log_probs ", predictive_log_probs.shape)
        #print("self.joint_log_probabilities ", self.joint_log_probabilities.shape)
        #print("log_model_posteriors ", log_model_posteriors.shape)
        #print("predictive_log_probs ", predictive_log_probs.shape)
        #print("r0 log prob", r0_log_prob)
        #print("log CP evidence", log_CP_evidence)

        """STEP 2: update all no-CP joint probabilities, so-called 'growth
        probabilities' to follow MacKay's naming convention. There will be t
        of them, namely for r=1,2,...,t-1, > t-1, with the last probability
        always being 0 unless we have boundary conditions allowing for
        non-zero probability that we are already in the middle of a segment
        at the first observation."""
        try:
            growth_log_probabilities = (self.one_step_ahead_predictive_log_loss +
                                        self.joint_log_probabilities +
                                        log_model_posteriors +
                                        np.log(1-cp_model.hazard_vector(1, t)))
        except ValueError as v:
            print(v)
            print("log model posteriors:", log_model_posteriors)
            print("log model posteriors shape:", log_model_posteriors.shape)
        #Here, the hazard rate/r_t probability is contained in log_CP_evidence
        CP_log_prob = self.r0_log_loss + np.log(model_prior) + log_CP_evidence


        """STEP 3: Get CP & growth probabilties at time t by summing
        at each spatial location over all time points and multiplying by
        the hazard rate"""
        #DEBUG: We need to use chopping of run-lengths here, too
        #DEBUG: we need the prior predictive here!
        #CP_log_prob = scipy.misc.logsumexp(helper_log_probabilities +
        #                                  np.log(cp_model.hazard_vector(1, t)))
        #self.joint_log_probabilities = (helper_log_probabilities +
        #    np.log(1-cp_model.hazard_vector(1, t)))

        """Put together steps 2-3"""
        joint_log_probabilities_tm1 = self.joint_log_probabilities
        self.joint_log_probabilities = np.insert(growth_log_probabilities,
                                                0, CP_log_prob)

        """STEP 4: Lastly, we always want to get the new evidence for this
        particular model in the model universe, which is the sum of CP and
        growth probabilities"""
        model_log_evidence_tm1 = self.model_log_evidence
        self.model_log_evidence = scipy.misc.logsumexp(
                self.joint_log_probabilities )


        """STEP 5: If we want to optimize the hyperparameters, do so after
        all joint log probabilities, the evidence, and so on are updated"""
        if (self.hyperparameter_optimization is not None and
            self.hyperparameter_optimization is not False and
            disable_hyperparameter_optimization is not True):

            """STEP 5.2: We have P(y_1:t, r_t, m_t = m), so if we sum over all
            tuples (m_t = m, r_t = r) for fixed m and variable r, what we get
            is P(y_1:t, m_t=m), meaning P(y_1:t,r_t,m_t=m)/P(y_1:t, m_t=m) =
            P(r_t|y_1:t, m_t)"""
            run_length_log_distro = (
                    joint_log_probabilities_tm1 - model_log_evidence_tm1)

            """STEP 5.1: This is different for each model, so we evaluate the
            derivative inside the individual model objects. Note that for the
            first observation (t = lag_length+1), the gradient of the pred
            log distr will just be that of the prior."""

            """This returns log(d/dtheta loss(x)) and its signs"""
            #DEBUG: This should be differentiated w.r.t. the probability, not
            #       the loss!
            gradients_log_predictive_val, gradients_log_predictive_sign = (
                    self.differentiate_predictive_log_distribution(y,t,
                    run_length_log_distro))
            num_params = gradients_log_predictive_val.shape[0]
            run_length_num = self.retained_run_lengths.shape[0]


            """STEP 5.3: Update the derivatives of the probabilities
            P(y_1:t, r_t|m_t=m) = P(y_1:t, r_t, m_t=m)/P(m_t=m). Notice that
            P(m_t = m) = prior(m)."""
            #DEBUG: Check if this is correct formula if something fails
            model_specific_joint_log_probs = (
                    joint_log_probabilities_tm1 - np.log(model_prior))

            """STEP 5.4: Caron's gradient: vectorized update of
            log(dP(y_t|y_1:t-1)) for all parameters over which we optimize."""
            #DEBUG: Need initialization of model_specific log probabilities derivative
            #       in some way.

            """STEP 5.4.1: joint log probs derivative (jlpd) parts 1 and 2
            are obtained and combined to update the model specific joint
            log probs derivative. The signs carry over from the gradients of
            the predictive and the joint log probs"""

            """(dP(y_t|r_t-1, y_1:t-1)*P(r_t-1, y_1:t-1)*P(r_t|r_t-1)), for
            each r_t and each param, so size is num_params x run_length_num"""
            jlpd_part1_val = (
                gradients_log_predictive_val[:,1:] +
                model_specific_joint_log_probs +
                np.log(1-cp_model.hazard_vector(1, t))
                )
            jlpd_part1_sign = gradients_log_predictive_sign[:,1:]

            """(dP(r_t, y_1:t-1)*P(r_t|r_t-1)*P(y_t|r_t-1, y_1:t-1)), for
            each r_t and each param, so size is num_params x run_length_num"""
            jlpd_part2_val = (
                self.model_specific_joint_log_probabilities_derivative +
                self.r0_log_prob +
                np.log(1-cp_model.hazard_vector(1, t))
                )
            jlpd_part2_sign = (
                self.model_specific_joint_log_probabilities_derivative_sign)

            """Combine jlpd parts 1 and 2, store result in dP(r_t, y_1:t-1),
            the gradient of the joint probabilities, as before the
            size is num_params x run_length_num"""
            res_val, res_sign = scipy.misc.logsumexp(
                a = np.array([jlpd_part1_val, jlpd_part2_val]),
                b = np.array([jlpd_part1_sign, jlpd_part2_sign]),
                return_sign=True,
                axis=0
                )


            """So far, we have only computed the derivatives for non-CPs.
            Next, we compute the derivative for a CP, using the recursion:
            CP_grad_1 = sum( P(r=0|r_t-1)*dP(y_t|y_1:t-1)*P(y_1:t-1, r_t-1) )
            CP_grad_2 = sum( P(r=0|r_t-1)*P(y_t|y_1:t-1)*dP(y_1:t-1, r_t-1) ).
            Size is num_params x run_length_num
            """
            results_grad_1 = (
                scipy.misc.logsumexp(
                        a = np.array([
                             gradients_log_predictive_val[:,1:] + #[:,np.newaxis] +
                             np.log(cp_model.hazard_vector(1,t)) +
                             model_specific_joint_log_probs
                        ]),
                        b = gradients_log_predictive_sign[:,1:], #[:,np.newaxis],
                        return_sign = True,
                        axis=1
                    )
                )
            CP_grad_1_val, CP_grad_1_sign = (results_grad_1[0].flatten(),
                                             results_grad_1[1].flatten())
            results_grad_2 = (
                scipy.misc.logsumexp(
                    a = np.array([
                         self.model_specific_joint_log_probabilities_derivative +
                         np.log(cp_model.hazard_vector(1,t)) +
                         self.one_step_ahead_predictive_log_probs
                    ]),
                    b = (
                      self.model_specific_joint_log_probabilities_derivative_sign),
                    return_sign = True,
                    axis=1
                    )
                )
            CP_grad_2_val, CP_grad_2_sign = (results_grad_2[0].flatten(),
                                             results_grad_2[1].flatten())

            CP_grad_val, CP_grad_sign = scipy.misc.logsumexp(
                    a = np.array([CP_grad_1_val, CP_grad_2_val]),
                    b = np.array([CP_grad_1_sign, CP_grad_2_sign]),
                    return_sign = True,
                    axis=1
                    )






            """STEP 6: Perform gradient descent step using caron's method"""
            if (self.hyperparameter_optimization == "caron" or
                self.hyperparameter_optimization == "online"):

                """STEP 6.1: Get sum of the model specific joint log prob
                derivatives, i.e. sum_{r \in R(t)} d/dtheta P(r_t, y_1:t|m_t)"""
                joint_log_der_sum, joint_log_der_sum_signs = (
                    scipy.misc.logsumexp(
                    a=(self.model_specific_joint_log_probabilities_derivative),
                    b=self.model_specific_joint_log_probabilities_derivative_sign,
                    return_sign=True, axis=1))
                """STEP 6.2: Get the two parts needed for computing the
                derivative of P(r_t|y_1:t, m_t) w.r.t. theta"""
                log_evidence = scipy.misc.logsumexp(model_specific_joint_log_probs)
                part1 = (self.model_specific_joint_log_probabilities_derivative
                         - log_evidence)
                part2 = (joint_log_probabilities_tm1 - 2*log_evidence +
                         joint_log_der_sum[:, np.newaxis])
                """STEP 6.3: add them together with appropriate sign to compute
                the desired quantity, d/dtheta P(r_t|y_1:t, m_t). Note that we
                flip the sign of the second quantity, as it is subtracted"""
                #DEBUG: need to stretch signs of joint log der sums !
                rld_der_signs, rld_der = scipy.misc.logsumexp(
                    a = np.array([part1, part2]),
                    b = np.array([
                    self.model_specific_joint_log_probabilities_derivative_sign,
                    (-1) * joint_log_der_sum_signs[:,np.newaxis] *
                        np.ones((num_params, run_length_num))]),
                    return_sign = True, axis = 0)
                """STEP 6.4: Finally, compute the gradient of P(Y_t|Y_1:t-1)
                using all of the previous computations"""

                """NOTE: In this computation, we compute P(Y_t|Y_1:t-1), so the
                retained run-lengths are r_t-1, which means in particular that
                we do NOT allow a CP at time t. (Thus, we only take the 1:
                entries of gradients_log_predictive_val and do not use
                the CP-probability of the predictives!)"""
#                print("gradients_log_predictive_val[:,1:]", gradients_log_predictive_val[:,1:].shape)
#                print("run_length_log_distro", run_length_log_distro.shape)
#                print("rld_der", rld_der.shape)
#                print("self.one_step_ahead_predictive_log_probs[:,np.newaxis]", self.one_step_ahead_predictive_log_probs[:,np.newaxis].shape)
                gradient, gradient_signs = scipy.misc.logsumexp(
                    a = np.array([
                        gradients_log_predictive_val[:,1:] +
                            run_length_log_distro[np.newaxis,:],
                        self.one_step_ahead_predictive_log_probs[np.newaxis,:] +
                            rld_der]),
                    b = np.array([
                        gradients_log_predictive_sign[:,1:],
                        rld_der_signs]),
                    return_sign = True,
                    axis = (0,2))

                """STEP 6.5: We now also need to compute the not-derivative of
                P(Y_t|Y_1:t-1), since what we really want is
                d/dtheta{log(P(Y_t|Y_1:t-1))} =
                    d/dtheta{P(Y_t|Y_1:t-1)}/P(Y_t|Y_1:t-1)"""
                pred = scipy.misc.logsumexp(
                        self.one_step_ahead_predictive_log_probs +
                        run_length_log_distro)

                #make sure that we are not making infinitely large steps
                grad = np.nan_to_num(np.exp(gradient-pred))
                sig  = np.nan_to_num(gradient_signs)
                #nans occur due to overflow in the exponential, so just make them
                #a large number!
                #new_grad = raw_grad
                #new_grad[np.isnan(raw_grad)] = pow(10,5) + 0.001
                grad[grad == 0] = pow(10,5)
                caron_gradient = np.minimum((grad),
                                pow(10,5)*np.ones(num_params))*sig
#                print(pred)
#                print(gradient)
#                print(gradient)
#                print(gradient_signs)
#                print(grad)
#                print(caron_gradient)

                """STEP 6.5: Call the optimization routine that you selected
                for this model. 'increment' is a num_params x 1 vector that
                gives direction for each hyperparameter"""
                p, C, scale = 1.0005, 1000, 3 #Nile: 1.005, 1
                step_size = self.step_caron_pow2(t, p, C, scale)
                #increment = step_size * caron_gradient
                self.caron_hyperparameter_optimization(t, caron_gradient,
                                                       step_size)

#                sign, caron_gradient = scipy.misc.logsumexp(
#                    a=(self.model_specific_joint_log_probabilities_derivative),
#                    b=self.model_specific_joint_log_probabilities_derivative_sign,
#                    return_sign=True, axis=1)
#
#                """Notice: What we use here is that denoting by theta_1:t the
#                sequence of hyperparams used at times 1:t, the derivative
#                d/dtheta_t P(y_t|y_1:t-1) = d/dtheta_t {P(y_1:t)/P(y:1:t-1)},
#                and moreover P(y:1:t-1) does not depend on theta_t! So
#                d/dtheta_t P(y_1:t)/P(y_1:t-1)=d/dtheta_t{P(y_1:t)}/P(y_1:t-1),
#                which means that in log-scale, we have
#                log(d/dtheta_tP(y_t|y:1:t-1)) = log(d/dtheta_t{P(y_1:t)}) -
#                                                log(P(y_1:t-1))"""
#                #DEBUG: This only works if we do NOT use generalized Bayes!
#                #       Fix this.
#                caron_gradient = np.exp(caron_gradient)*sign - log_evidence
#
#                """STEP 5.5: Call the optimization routine that you selected
#                for this model. 'increment' is a num_params x 1 vector that gives
#                the direction for each hyperparameter"""
#                p, C, scale = 1.0005, 1000, 3 #Nile: 1.005, 1
#                step_size = self.step_caron_pow2(t, p, C, scale)
#                #increment = step_size * caron_gradient
#                self.caron_hyperparameter_optimization(t, caron_gradient,
#                                                       step_size)

            """STEP 7: Update the derivatives of the joint log probs. The sum
            of these quantities is d/dtheta P(y_1:t), i.e. the gradient for the
            method of Turner et al. Needs to be done this late because the
            d/dtheta P(r_t-1, y_1:t-1) needed for Caron's gradient descent"""

            """Compute P(y:1_t-1|m_t=m)"""
            log_evidence = scipy.misc.logsumexp(model_specific_joint_log_probs)
            """Update derivatives"""
            self.model_specific_joint_log_probabilities_derivative = np.insert(
                    res_val, 0, CP_grad_val, axis=1)
            self.model_specific_joint_log_probabilities_derivative_sign = (
                    np.insert(res_sign, 0, CP_grad_sign, axis=1))


    def update_alpha_derivatives(self, y, t,
                                 log_model_posteriors,
                                 log_model_posteriors_der,
                                 log_model_posteriors_der_sign,
                                 log_CP_evidence,
                                 log_CP_evidence_der,
                                 log_CP_evidence_der_sign,
                                 model_prior,
                                 cp_model):
        """Uses recursive updates to obtain the derivatives of the joint
        probabilities w.r.t. alpha.
        NOTE: Do the computations in log-form, and then convert back
        log_model_posteriors: P(m_t|m_t-1, r_t-1, y_1:t-1)
        log_model_posteriors_der: derivative w.r.t. alpha, computed in Detector
        log_CP_evidence: sum(r_t-1) sum(m_t-1) P(m_t-1, r_t-1, y_1:t-1)
        log_CP_evidence_der: derivative w.r.t. alpha, computed in Detector
        model_prior: q(m)
        cp_model: probability_cp_model object, passed in by Detector
        """

        #DEBUG: Initialize the alpha-derivatives to 0 in initiatlization.
        #DEBUG: Put this into the probability_model, since it will be
        #       the same for any model!

        """STEP 1: We need expressions that are derivatives w.r.t. alpha,
            (1) one-step-ahead predictive loss' derivative
            (2) conditional model posterior's derivative [passed in]"""
        _1, _2 = (
            self.get_one_step_ahead_log_loss_derivatives_power_divergence())
        one_step_ahead_log_loss_derivatives = _1
        one_step_ahead_log_loss_derivatives_sign = _2

        """STEP 2: Get the expressions that are NOT derivatives w.r.t. alpha.
        Note that the entire derivative can be written as
            d/dalpha P(y_1:t, r_t, m_t) = d/dalpha{P(y_1:t-1, r_t-1, m_t-1)} *
                rest_1 + d/dalpha{P(y_t|r_t, m_t, y_1:t-1) * rest_2 +
                d/dalpha P(m_t|m_t-1, r_t, y_1:t-1) * rest_3}
        """
        #DEBUG: joint log probs also massive! 10^4 to 10^13 iin log form !!!
        #DEBUG: one-step-ahead predictive log loss around 200 in log form!
        #DEBUG: log model posterior derivatives are of order 10^5 - 10^15 in log.
        full = (self.one_step_ahead_predictive_log_loss +
                self.joint_log_probabilities +
                log_model_posteriors +
                np.log(1-cp_model.hazard_vector(1, t)))
        rest_1 = full - self.joint_log_probabilities
        rest_2 = full - self.one_step_ahead_predictive_log_loss
        rest_3 = full - log_model_posteriors


        """STEP 3: Combine the expressions that are and are not derivatives
        w.r.t. alpha to update the logs of the derivatives of the joint log
        probabilities w.r.t. alpha.
        NOTE: These are the growth-probability updates!
        """
        new_derivatives, new_derivatives_sign = scipy.misc.logsumexp(
                a = np.array([
                    rest_1 + self.log_alpha_derivatives_joint_probabilities,
                    rest_2 + one_step_ahead_log_loss_derivatives[1:],
                    rest_3 + log_model_posteriors_der
                    ]),
                b = np.array([
                    self.log_alpha_derivatives_joint_probabilities_sign,
                    one_step_ahead_log_loss_derivatives_sign[1:],
                    log_model_posteriors_der_sign
                        ]),
                return_sign = True,
                axis = 0
            )

        """STEP 4: Get the CP-probability derivative"""
        CP_d_1 = (one_step_ahead_log_loss_derivatives[0] + np.log(model_prior) +
                  log_CP_evidence)
        CP_d_2 = (self.r0_log_loss + np.log(model_prior) + log_CP_evidence_der)
        CP_derivative, CP_derivative_sign = scipy.misc.logsumexp(
            a=np.array([CP_d_1, CP_d_2]),
            b = np.array([one_step_ahead_log_loss_derivatives_sign[0],
                          log_CP_evidence_der_sign]),
            return_sign = True)

        self.log_alpha_derivatives_joint_probabilities = np.insert(
                new_derivatives, 0, CP_derivative)
        self.log_alpha_derivatives_joint_probabilities_sign = np.insert(
                new_derivatives_sign,0,CP_derivative_sign)


    #@staticmethod
    def step_caron_pow2(self,t, p, C,scale):
        """for any p \in (1, \infinity), and for any finite constant C this
        sequence is square summable, but sums to infinity, too."""
        if self.has_lags:
            t_ = t-self.lag_length
        else:
            t_ = t
        return scale*C*(1/(t_+C))
#        #if t>30:
#            return C*(1/t_-30)
#        else:
#            return 0



    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS FOR t>1!
    def update_predictive_distributions(self, y, t, r_evaluations):
        """update the distributions giving rise to the predictive probabilities
        in the next step of the algorithm. Could happen computationally (e.g.,
        INLA) or exactly (i.e., with conjugacy)"""
        pass

    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS
    def get_posterior_expectation(self, t, r_list=None):
        """get the predicted value/expectation from the current posteriors
        at time point t, for all possible run-lengths"""
        pass

    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS
    def get_posterior_variance(self, t, r_list=None):
        """get the predicted variance from the current posteriors at
        time point t, for all possible run-lengths"""
        pass

    #SHOULD BE IMPLEMENTED IN EACH SUBLCASS
    def prior_log_density(self, y_flat):
        """Computes the log-density of *y_flat* under the prior. Called by
        the associated Detector object to get the next segment-density
        component of the model, denoted D(t,q) for q = NIW"""
        pass


    #CALLS trimmer(kept_run_lengths), which needs to be implemented in each
    #       subclass of probability_model
    def trim_run_length_log_distrbution(self, t, threshold, trim_type):
        """Trim the log run-length distribution of this model such that any
        run-lengths with probability mass < threshold will be deleted"""

        """If we want to apply thresholding, trim the distributions, see
        STEP 2 and STEP 3. Otherwise, don't do anything in this function
        apart from keeping track of all the run-lengths"""

        if (not ((threshold is None) or (threshold == 0) or (threshold == -1))):

            """STEP 2: Find out which run-lengths we need to trim now. Note
            that P(r_t|m_t,y_1:t) = P(r_t,m_t,y_1:t)/P(m_t,y_1:t), and that
            P(m_t,y_1:t) is what I store as model log evidence"""
            run_length_log_distr = (self.joint_log_probabilities -
                                    self.model_log_evidence)

            if trim_type == "threshold":
                # JAY: THIS SHOULD RUN INSTEAD OF keep_K
                """STEP 2A: If we trim using a threshold, trim accordingly"""
                kept_run_lengths = np.array(run_length_log_distr) >= threshold

            elif trim_type == "keep_K":
                """STEP 2B: If we trim discarding all but the K largest run
                length probabilities, trim accordingly"""
                K = min(threshold, np.size(run_length_log_distr))
                max_indices = (-run_length_log_distr).argsort()[:K]
                kept_run_lengths = np.full(np.size(run_length_log_distr),
                                           False, dtype=bool)
                kept_run_lengths[max_indices] = True

            elif trim_type == "SRC":
                """STEP 2C: If we keep all particles larger than alpha and
                resample the remaining N-A ones"""

                """STEP 2.C.0: Depending on whether threshold>0 or <0, design
                alpha as threshold or as K-th largest value's value"""
                if threshold < 0: #i.e., we give the log-prob format directly
                    alpha = threshold
                elif threshold > 0: #i.e., we give K max type input
                    #Choose K-th largest particle's weight as alpha
                    K = min(threshold, np.size(run_length_log_distr))
                    alpha_index = (-run_length_log_distr).argsort()[K-1]
                    alpha = run_length_log_distr[alpha_index]


                """STEP 2C.1: Keep all particles larger than alpha"""
                kept_run_lengths = np.array(run_length_log_distr) >= alpha
                #A = np.sum(kept_run_lengths, dtype=int)
                #smaller_alpha_indices = #not larger_alpha

                """STEP 2C.2: For the remaining particles, apply stratified
                resampling"""
                u = np.random.uniform(low=0, high=alpha)
                for (index, rl_particle) in zip(range(0, np.size(
                        run_length_log_distr)), run_length_log_distr):
                    if rl_particle < alpha:
                        #u = u-rl_particle in non-log format is <= 0 iff
                        #it holds that u<=rl_particle
                        if u <= rl_particle:
                            kept_run_lengths[index] = True
                            run_length_log_distr[index] = alpha
                            #DEBUG: log sum exp
                            u = scipy.misc.logsumexp(
                                    np.array([u, alpha, -rl_particle]))


            """STEP 3: Drop the quantities associate with dropped run-lengths.
            If there is at least one probability kept, do as usual. If the
            number of probabilities kept is 0, retain r=t-1"""
            """NOTE: Should only be a problem if trim_type is threshold"""
            if trim_type == "SRC" and np.sum(kept_run_lengths)>1:
                #we need to replace the joint_log_probs inside the model object
                #with the new alpha-replaced ones.
                self.joint_log_probabilities = (run_length_log_distr +
                                   self.model_log_evidence)
                self.trimmer(kept_run_lengths)
            elif np.sum(kept_run_lengths)>1:
                # Jay: This should run only after initialization
                self.trimmer(kept_run_lengths)
            elif np.sum(kept_run_lengths)<=1:
                #get the two largest entries and only keep them.
                max_indices = (-run_length_log_distr).argsort()[:2]
                kept_run_lengths[max_indices] = True
                self.trimmer(kept_run_lengths)


    def alpha_param_gradient_computation(self, y, t, cp_model,
                         model_prior,
                         log_model_posteriors,
                         log_CP_evidence, eps ):
        """empty function that does nothing. If the relevant model class is
        using the DPD loss, this relevant model class extends the function
        and performs an update on alpha_param"""
        pass


    def DPD_joint_log_prob_updater(self,
                alpha_direction, y, t, cp_model, model_prior,
                log_model_posteriors,  log_CP_evidence):
        """called inside DPD_alpha_update for those classes that implement
        DPD"""

#        """STEP 1: Discern direction of alpha-deviation (i.e., whether we have
#        alpha + eps or alpha - eps)"""
#        if alpha_direction > 0:

        #DEBUG: I think the if statment is not needed, as the 'direction' is
        #       directly passed on to evaluate_pred_log_distr, and not relevant
        #       anywhere else!
        """STEP 1A: If we want alpha + eps, compute accordingly"""
        r0_log_prob = self.evaluate_log_prior_predictive(y,t,
                            store_posterior_predictive_quantities = False)
         #IMPLEMENT: Needs to be adapted s.t. I can get evaluations for
        #           the different sets of parameters (i.e. L_rt_p_eps etc)
        one_step_ahead_predictive_log_probs = (
                self.evaluate_predictive_log_distribution(y,t,
                            store_posterior_predictive_quantities = False,
                            alpha_direction = alpha_direction))

        if self.generalized_bayes_rld == "power_divergence":

            """Note: The alpha we use here is the RLD-alpha, so we don't
            need to adjust it for alpha + eps or alpha - eps"""
            #print("DPD")
            #DEBUG: Problem with length mismatch between joint log probs
            #       and the integrals (log probs 1 unit longer!)
            #Q: Is that because we basically have no influence of alpha on
            #   r = 0, i.e. we should really just look at the
            #NOTE: Atm, we fix this by inserting the prior log det again in
            #       the first position, but this is not correct
            integrals = self.get_log_integrals_power_divergence(DPD_call=True)


            #DEBUG: Check if np.log(maxFloat) >= any value in integrals.
            max_val = np.log(np.finfo(np.float64).max)
            #Split integral term s.t. you set terms exceeding max_val to max_val
            #and leave the rest. Similarly for the one step ahead pred.


            """STEP 1: Correct the integral term by bounding"""
            integral_exceeds = np.where(integrals >= max_val)[0]
            integral_fine = np.where(integrals < max_val)[0]
            integral_exp = integrals

            if len(integral_exceeds)>0:
                integral_exp[integral_exceeds] = (
                        min(1.0,(1.0/(self.alpha_rld+1)))*
                        (np.finfo(np.float64).max)
                )
            if len(integral_fine)>0:
                """Just do the standard thing"""
                integral_exp[integral_fine] = (
                        min(1.0,(1.0/(self.alpha_rld+1)))*
                         np.exp(integrals[integral_fine])
                    )

            """STEP 2: Correct the predictive probs"""

            """STEP 2.1: Correct CP prob if necessary"""
            if self.r0_log_prob >= max_val:
                r0_log_prob_exp = (min(1.0, 1.0/self.alpha_rld) *
                                   (min(self.alpha_rld, 1) * max_val))
            else:
                r0_log_prob_exp = (1.0/self.alpha_rld*np.exp(
                    r0_log_prob*self.alpha_rld))

            """STEP 2.2: Correct other probs"""
            step_ahead_exceeds = np.where(
                    one_step_ahead_predictive_log_probs >= max_val)[0]
            step_ahead_fine = np.where(
                    one_step_ahead_predictive_log_probs < max_val)[0]
            step_ahead_exp = one_step_ahead_predictive_log_probs
            if len(step_ahead_exceeds)>0:
                if self.alpha_rld < 1:
                    """If the max value will be downweighted by alpha"""
                    step_ahead_exp[step_ahead_exceeds] = (
                            min(1.0,1.0/(self.alpha_rld))*
                            (np.finfo(np.float64).max * self.alpha_rld)
                    )
                else:
                    """If the max value will be increased by alpha"""
                    step_ahead_exp[step_ahead_exceeds] = (
                            min(1.0,(1.0/(self.alpha_rld)))*
                            (np.finfo(np.float64).max)
                    )
            if len(step_ahead_fine)>0:
                """Just do the standard thing"""
                step_ahead_exp[step_ahead_fine] = (
                        ((1.0/(self.alpha_rld+1))*
                         np.exp(
                            one_step_ahead_predictive_log_probs[
                                    step_ahead_fine]*
                            self.alpha_rld)
                    ))

            one_step_ahead_predictive_log_probs = (
                    step_ahead_exp - integral_exp[1:])
            r0_log_prob = r0_log_prob_exp - integral_exp[0]




            """Note: If we have the power divergence loss, then our log of
            the loss should be computed from the predictive probabilities
            appropriately"""
            #DEBUG: Original/numerically unstable version
#            one_step_ahead_predictive_log_probs = (1.0/self.alpha_rld*
#                    np.power(np.exp(one_step_ahead_predictive_log_probs),
#                         self.alpha_rld) -
#                    (1.0/(self.alpha_rld+1.0))*np.exp(integrals[1:])
#                )
#            r0_log_prob = (1.0/self.alpha_rld*np.power(np.exp(r0_log_prob),
#                                                       self.alpha_rld)
#                - (1.0/(self.alpha_rld+1.0))*np.exp(integrals[0]))


        """STEP 2: update all no-CP joint probabilities, so-called 'growth
        probabilities' to follow MacKay's naming convention. There will be t
        of them, namely for r=1,2,...,t-1, > t-1, with the last probability
        always being 0 unless we have boundary conditions allowing for
        non-zero probability that we are already in the middle of a segment
        at the first observation."""
        growth_log_probabilities = (one_step_ahead_predictive_log_probs +
                                    self.joint_log_probabilities +
                                    log_model_posteriors +
                                    np.log(1-cp_model.hazard_vector(1, t)))
        CP_log_prob = r0_log_prob + np.log(model_prior) + log_CP_evidence
        joint_log_probabilities = np.insert(growth_log_probabilities,
                                                0, CP_log_prob)

        return joint_log_probabilities
