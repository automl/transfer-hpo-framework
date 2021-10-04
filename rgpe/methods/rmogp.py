import copy
from typing import Optional

import numpy as np
from scipy.stats import norm

from smac.epm.base_epm import AbstractEPM
from smac.optimizer.acquisition import AbstractAcquisitionFunction, EI

from rgpe.utils import sample_sobol


class MixtureOfGPs(AbstractAcquisitionFunction):

    def __init__(self,
                 model: AbstractEPM,
                 use_expectation=True,
                 use_global_incumbent=False,
                 ):
        """Ranking-weighted Mixture of Gaussian Processes acquisition function

        Parameters
        ----------
        model : AbstractEPM
            An linearly-weighted ensemble which contains a model for each base task and the target
            task.
        use_expectation : bool
            Whether to compute the expectation per base task. Defaults to ``True``,
            using ``False`` makes the acquisition function behave similar to the transfer
            acquisition function (Wistuba et al., Machine Learning 2018).
        use_global_incumbent : bool
            Whether to use a global incumbent or an incumbent per task. Defaults to ``False``,
            using ``True`` makes the acquisition function behave more similar to 'Active Testing'
            from Leite and Brazdil (2012).
        """

        super().__init__(model)
        self.long_name = 'Transfer Acquisition Function'
        self.eta = None
        self.etas = None
        self.n_models = 0
        self.use_expectation = use_expectation
        self.use_global_incumbent = use_global_incumbent

        self.base_models = None

    def update(self, **kwargs):
        """SMAC's acquisition function update mechanism.

        This is a fast implementation which copies the base models once in the beginning. Do use
        with care if moving the acquisition function to a new version of SMAC, a different
        Bayesian optimization library or somehow else change the experimental setup. If you are
        unsure about this, please use the slower implementation below which does a deepcopy in
        every iteration.
        """
        model = kwargs['model']
        self.n_models = len(self.model.model_list_)

        X = kwargs['X']

        if self.base_models is None:
            self.base_models = copy.deepcopy(model.base_models)
        model = copy.copy(model)

        etas = []
        for i, (submodel, weight) in enumerate(zip(model.model_list_, model.weights_)):
            if weight <= 0:
                etas.append(np.inf)
                continue
            if self.use_expectation and i != self.n_models - 1:
                # Use the re-parametrization trick to get rid of noise
                original_training_data = submodel.gp.X_train_.copy()
                integrate = np.vstack((original_training_data, X))
                sample, _ = submodel.predict(integrate)
                theta = self.base_models[i].gp.kernel.theta
                theta[-1] = -25
                self.base_models[i].gp.kernel.theta = theta
                self.base_models[i]._train(integrate, sample, do_optimize=False)
            if self.use_global_incumbent:
                eta, _ = submodel.predict(self.incumbent_array)
            else:
                means, _ = submodel.predict(X)
                eta = np.min(means)
            etas.append(eta)

        model.base_models = []
        model.model_list_ = []
        for submodel in self.base_models:
            model.base_models.append(submodel)
            model.model_list_.append(submodel)
        model.model_list_.append(model.target_model)

        self.model = model
        self.etas = etas

    def update_slow(self, **kwargs):
        """SMAC's acquisition function update mechanism."""
        model = kwargs['model']
        self.n_models = len(self.model.model_list_)

        X = kwargs['X']

        self._do_integrate = True
        model_ = copy.deepcopy(model)
        etas = []
        for submodel, weight in zip(model_.model_list_, model_.weights_):
            if weight <= 0:
                etas.append(np.inf)
                continue
            if self.use_expectation:
                # Use the re-parametrization trick to get rid of noise
                original_training_data = submodel.gp.X_train_.copy()
                integrate = np.vstack((original_training_data, X))
                sample, _ = submodel.predict(integrate)
                print('before', submodel.gp.kernel.theta)
                theta = submodel.gp.kernel.theta
                theta[-1] = -25
                submodel.gp.kernel.theta = theta
                print('after', submodel.gp.kernel.theta)
                submodel._train(integrate, sample, do_optimize=False)
            if self.use_global_incumbent:
                eta, _ = submodel.predict(self.incumbent_array)
            else:
                means, _ = submodel.predict(X)
                eta = np.min(means)
            etas.append(eta)

        self.model = model_
        self.etas = etas

    def _compute(self, X: np.ndarray, **kwargs):
        """SMAC's acquisition function computation mechanism."""

        ei_values = []

        for i, (weight, model) in enumerate(zip(self.model.weights_, self.model.model_list_)):
            if weight == 0:
                continue
            else:
                eta = self.etas[i]
                if self.use_expectation or i == self.n_models - 1:

                    m, v = model.predict(X)
                    s = np.sqrt(v)
                    eta_minus_m = eta - m

                    def calculate_f():
                        z = eta_minus_m / s
                        return eta_minus_m * norm.cdf(z) + s * norm.pdf(z)

                    if np.any(s == 0.0):
                        # if std is zero, we have observed x on all instances
                        # using a RF, std should be never exactly 0.0
                        # Avoid zero division by setting all zeros in s to one.
                        # Consider the corresponding results in f to be zero.
                        self.logger.warning("Predicted std is 0.0 for at least one sample.")
                        s_copy = np.copy(s)
                        s[s_copy == 0.0] = 1.0
                        ei = calculate_f()
                        ei[s_copy == 0.0] = 0.0
                    else:
                        ei = calculate_f()
                    if (ei < 0).any():
                        raise ValueError(
                            "Expected Improvement is smaller than 0 for at least one "
                            "sample.")

                    ei_values.append(ei * weight)
                else:
                    m, _ = model.predict(X)
                    improvement = eta - m
                    improvement = improvement
                    improvement = np.maximum(improvement, 0)
                    ei_values.append(improvement * weight)

        rval = np.sum(ei_values, axis=0)
        rval = rval.reshape((-1, 1))

        return rval

