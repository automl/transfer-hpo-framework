import json
from typing import Dict, List, Optional, Tuple, Union

from ConfigSpace import Configuration
import numpy as np
from sklearn.linear_model import SGDRegressor
from smac.configspace import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM

from rgpe.utils import get_gaussian_process


class WarmstartingAC(AbstractEPM):

    """Weighting method from "Warmstarting of Model-based Algorithm Configuration" by Lindauer
    and Hutter, AAAI 2018

    https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17235/15829
    """

    def __init__(
        self,
        training_data: Dict[int, Dict[str, Union[List[Configuration], np.ndarray]]],
        variance_mode: str = 'average',
        ** kwargs
    ):
        if kwargs.get('instance_features') is not None:
            raise NotImplementedError()
        super().__init__(**kwargs)
        self.training_data = training_data

        self.rng = np.random.RandomState(self.seed)
        self.variance_mode = variance_mode

        base_models = []
        for task in training_data:
            model = get_gaussian_process(
                bounds=self.bounds,
                types=self.types,
                configspace=self.configspace,
                rng=self.rng,
                kernel=None,
            )
            Y = training_data[task]['y']
            mean = Y.mean()
            std = Y.std()
            if std == 0:
                std = 1

            y_scaled = (Y - mean) / std
            y_scaled = y_scaled.flatten()
            configs = training_data[task]['configurations']
            X = convert_configurations_to_array(configs)

            model.train(
                X=X,
                Y=y_scaled,
            )
            base_models.append(model)
        self.base_models = base_models
        self.sgd = SGDRegressor(random_state=12345, warm_start=True, max_iter=100)

        self.weights_over_time = []

    def _compute_weights(self, X, y):
        if X.shape[0] == 1:
            self.weights_ = np.ones(len(self.base_models) + 1) / (len(self.base_models) + 1)
            return
        predictions = []
        for base_model in self.base_models:
            m, _ = base_model.predict(X)
            predictions.append(m.flatten())
        loo_predictions = []
        for i in range(X.shape[0]):
            X_tmp = list(X)
            x_loo = X_tmp[i]
            del X_tmp[i]
            X_tmp = np.array(X_tmp)
            y_tmp = list(y)
            del y_tmp[i]
            y_tmp = np.array(y_tmp)
            self.target_model._train(X_tmp, y_tmp, do_optimize=False)
            m, _ = self.target_model.predict(np.array([x_loo]))
            loo_predictions.append(m)
        predictions.append(np.array(loo_predictions).flatten())
        predictions = np.array(predictions)
        self.sgd.fit(predictions.transpose(), y)
        self.weights_ = self.sgd.coef_
        # Counteract the following weird failure case:
        # * all observations so far have the same value -> normalization makes them all 0.0
        # * all predictions via cross-validation have a value of 0.0
        # -> this results in SGD having all weights being zero
        if np.sum(self.weights_) == 0:
            self.weights_[-1] = 1

    def _train(self, X: np.ndarray, Y: np.ndarray) -> AbstractEPM:
        Y = Y.flatten()
        mean = Y.mean()
        std = Y.std()
        if std == 0:
            std = 1

        y_scaled = (Y - mean) / std
        self.Y_mean_ = mean
        self.Y_std_ = std

        target_model = get_gaussian_process(
            bounds=self.bounds,
            types=self.types,
            configspace=self.configspace,
            rng=self.rng,
            kernel=None,
        )
        self.target_model = target_model.train(X, y_scaled)
        self.model_list_ = self.base_models + [target_model]
        self._compute_weights(X, Y)
        print('Weights', self.weights_)
        self.weights_over_time.append(self.weights_)

        # create model and acquisition function
        return self

    def _predict(self, X: np.ndarray, cov_return_type) -> Tuple[np.ndarray, np.ndarray]:

        # compute posterior for each model
        weighted_means = []
        weighted_covars = []

        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights_ ** 2 > 0).nonzero()[0]
        non_zero_weights = self.weights_[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            weight = non_zero_weights[non_zero_weight_idx]
            mean, covar = self.model_list_[raw_idx]._predict(X, cov_return_type)
            weighted_means.append(weight * mean)
            if self.variance_mode == 'average':
                weighted_covars.append(covar * weight)
            elif self.variance_mode == 'correct-average':
                weighted_covars.append(covar * weight ** 2)
            elif self.variance_mode == 'target':
                if raw_idx + 1 == len(self.weights_):
                    weighted_covars.append(covar)
            else:
                raise ValueError()

        if self.variance_mode == 'target':
            assert len(weighted_covars) == 1

        # set mean and covariance to be the rank-weighted sum the means and covariances
        # of the
        # base models and target model
        mean_x = np.sum(np.stack(weighted_means), axis=0) * self.Y_std_ + self.Y_mean_
        covar_x = np.sum(weighted_covars, axis=0) * (self.Y_std_ ** 2)
        return mean_x, covar_x
