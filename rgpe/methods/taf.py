import numpy as np

from smac.epm.base_epm import AbstractEPM
from smac.optimizer.acquisition import AbstractAcquisitionFunction, EI


class TAF(AbstractAcquisitionFunction):

    def __init__(self, model: AbstractEPM):
        """Transfer acquisition function from "Scalable Gaussian process-based transfer surrogates
        for hyperparameter optimization" by Wistuba, Schilling and Schmidt-Thieme,
        Machine Learning 2018, https://link.springer.com/article/10.1007/s10994-017-5684-y

        Works both with TST-R and RGPE weighting.
        """

        super().__init__(model)
        self.long_name = 'Transfer Acquisition Function'
        self.eta = None
        self.acq = EI(model=None)

    def update(self, **kwargs):

        X = kwargs['X']
        prediction, _ = self.model.target_model.predict(X)
        self.incumbent_array = X[np.argmin(prediction)].reshape((1, -1))
        eta = np.min(prediction)
        assert (id(kwargs['model']) == id(self.model))
        kwargs = {}
        kwargs['model'] = self.model.target_model
        kwargs['eta'] = eta
        self.acq.model = None
        self.acq.update(**kwargs)
        best_values = []
        for weight, base_model in zip(self.model.weights_, self.model.base_models):
            if weight == 0:
                best_values.append(None)
            else:
                values, _ = base_model.predict(X)
                min_value = np.min(values)
                best_values.append(min_value)
        self.best_values = best_values

    def _compute(self, X: np.ndarray, **kwargs):

        ei = self.acq._compute(X)

        if self.model.weights_[-1] == 1:
            return ei

        else:
            improvements = []

            for weight, best_value, base_model in zip(self.model.weights_, self.best_values, self.model.base_models):
                if weight == 0:
                    continue
                else:
                    predictions, _ = base_model._predict(X, cov_return_type=None)
                    improvement = np.maximum(best_value - predictions, 0).flatten() * weight
                    improvements.append(improvement)

            improvements = np.sum(improvements, axis=0)

            rval = ei.flatten() * self.model.weights_[-1] + improvements
            rval = rval.reshape((-1, 1))

            return rval
