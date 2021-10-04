import copy
from typing import Optional

import numpy as np
from scipy.stats import norm

from smac.epm.base_epm import AbstractEPM
from smac.optimizer.acquisition import AbstractAcquisitionFunction, EI

from rgpe.utils import sample_sobol


class NoisyEI(AbstractAcquisitionFunction):
    """Implements the Noisy Expected Improvement by Letham et al. described in
    https://arxiv.org/abs/1706.07094 and used in https://arxiv.org/abs/1802.02219

    This implementation requires an ensemble of methods, for example RGPE and assumes that each
    method itself is a Gaussian Process as implemented in SMAC.

    If you are looking for a general implementation of NoisyEI we recommend having a look at
    BoTorch.
    """

    def __init__(
        self,
        model: AbstractEPM,
        target_model_incumbent: bool,
        acquisition_function: Optional[AbstractAcquisitionFunction] = None,
        par: float = 0.0,
        n_samples: int = 30,
    ):

        super().__init__(model)
        self.long_name = 'Noisy Expected Improvement'
        self.par = par
        self.eta = None
        self.target_model_incumbent = target_model_incumbent

        if acquisition_function is None:
            self.acq = EI(model=None)
        else:
            self.acq = acquisition_function
        self.n_samples = n_samples

        self._functions = None
        self._do_integrate = True

        self.base_models = None

    def update(self, model: AbstractEPM, **kwargs):

        X = kwargs['X']

        if model.weights_[-1] != 1:

            del kwargs['eta']
            models = []
            etas = []

            input_locations = []
            samples = []

            self._do_integrate = True

            if self.base_models is None:
                self.base_models = []
                for _ in range(self.n_samples):
                    model_ = copy.deepcopy(model)
                    self.base_models.append(model_.base_models)

            # First, create samples from each model of the ensemble to integrate over
            for model_idx, weight in enumerate(model.weights_):
                if weight <= 0:
                    # Ignore models with zero weight
                    samples.append(None)
                    input_locations.append(None)
                    continue
                submodel = model.model_list_[model_idx]
                original_training_data = submodel.gp.X_train_
                if model_idx == len(model.weights_) - 1:
                    integrate = original_training_data.copy()
                else:
                    integrate = np.vstack((original_training_data, X))
                try:
                    sample = sample_sobol(submodel, integrate, self.n_samples, model.rng.randint(10000))
                except:
                    sample = submodel.predict(integrate)[0].transpose()
                    sample = np.tile(sample, reps=self.n_samples)
                samples.append(sample)
                input_locations.append(integrate)

            # Second, train the integrated GPs for each base model
            for sample_idx in range(self.n_samples):

                # Copy the individual models
                # This is substantially faster than doing a deepcopy of all models as it avoids
                # doing a deepcopy of the base models
                model_ = copy.copy(model)
                model_.base_models = self.base_models[sample_idx]
                # do a deep copy of the target model so we don't mess with it's original noise
                # estimate. The original noise estimate will be used as the basis for the GPs HPO
                # when fitting it the next time.
                model_.target_model = copy.deepcopy(model.target_model)
                model_.model_list_ = model_.base_models + [model_.target_model]
                models.append(model_)

                # Train the individual models
                for model_idx, (submodel, weight) in enumerate(zip(model_.model_list_, model_.weights_)):
                    if weight <= 0:
                        continue
                    theta = submodel.gp.kernel.theta
                    theta[-1] = -25
                    submodel.gp.kernel.theta = theta
                    sample = samples[model_idx][sample_idx].reshape((-1, 1))
                    submodel._train(input_locations[model_idx], sample, do_optimize=False)

            for model_ in models:
                if self.target_model_incumbent:
                    predictions, _ = model_.target_model.predict(X)
                    predictions = predictions * model_.Y_std_ + model_.Y_mean_
                else:
                    predictions, _ = model_.predict(X)
                etas.append(np.min(predictions))

            if self._functions is None or len(self._functions) != len(models):
                self._functions = [copy.deepcopy(self.acq) for _ in models]
            for model, func, eta in zip(models, self._functions, etas):
                func.update(model=model, eta=eta, **kwargs)
        else:
            print('No need to integrate...')
            self._do_integrate = False
            del kwargs['eta']
            predictions, _ = model.predict(X)
            kwargs['eta'] = np.min(predictions)
            self.acq.update(model=model, **kwargs)

    def _compute(self, X: np.ndarray):
        if self._do_integrate:
            val = np.array([func._compute(X) for func in self._functions]).mean(axis=0)
            return val
        else:
            return self.acq._compute(X)


class ClosedFormNei(AbstractAcquisitionFunction):
    """Closed-form adaptation of the Noisy Expected Improvement.

    While it is substantially faster to compute it does not consider the uncertainty about
    which noisy observation is the best observation made so far.
    """

    def update(self, **kwargs):

        X = kwargs['X']
        self.model = kwargs['model']
        # Model prediction is only used when not integrating over base models
        prediction, _ = self.model.predict(X)
        self.incumbent_array = X[np.argmin(prediction)].reshape((1, -1))
        self.eta = np.min(prediction)

    def _compute(self, X: np.ndarray, **kwargs):

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        if self.model.weights_[-1] != 1:
            # Due to the joint prediction, it is not possible to compute EI only with respect to
            # the predicted value on the target task
            X_new = np.concatenate((self.incumbent_array, X), axis=0)
            m_pred, v_pred = self.model._predict(X_new, cov_return_type='full_cov')
            m_inc = m_pred[0]
            v_inc = v_pred[0][0]
            m_cand = m_pred[1:]
            cov = v_pred[0][1:]
            v_cand = np.diag(v_pred)[1:]
            m = m_inc - m_cand
            v = v_inc + v_cand - 2 * cov
            s = np.sqrt(v)
            eta_minus_m = m.reshape((-1, 1))
            s = s.reshape((-1, 1))
        else:
            m, v = self.model.predict(X)
            s = np.sqrt(v)
            eta_minus_m = self.eta - m

        def calculate_f():
            z = (eta_minus_m) / s
            return (eta_minus_m) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()
        if (f < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one "
                "sample.")
        return f
