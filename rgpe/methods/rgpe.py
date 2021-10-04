import json
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np

from ConfigSpace import Configuration
from smac.configspace import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM
from smac.epm.gaussian_process import GaussianProcess
from rgpe.utils import get_gaussian_process, sample_sobol, copula_transform


def roll_col(X: np.ndarray, shift: int) -> np.ndarray:
    """
    Rotate columns to right by shift.
    """
    return np.concatenate((X[:, -shift:], X[:, :-shift]), axis=1)


def compute_ranking_loss(
    f_samps: np.ndarray,
    target_y: np.ndarray,
    target_model: bool,
) -> np.ndarray:
    """
    Compute ranking loss for each sample from the posterior over target points.
    """
    y_stack = np.tile(target_y.reshape((-1, 1)), f_samps.shape[0]).transpose()
    rank_loss = np.zeros(f_samps.shape[0])
    if not target_model:
        for i in range(1, target_y.shape[0]):
            rank_loss += np.sum(
                (roll_col(f_samps, i) < f_samps) ^ (roll_col(y_stack, i) < y_stack),
                axis=1
            )
    else:
        for i in range(1, target_y.shape[0]):
            rank_loss += np.sum(
                (roll_col(f_samps, i) < y_stack) ^ (roll_col(y_stack, i) < y_stack),
                axis=1
            )

    return rank_loss


def get_target_model_loocv_sample_preds(
    train_x: np.ndarray,
    train_y: np.ndarray,
    num_samples: int,
    model: GaussianProcess,
    engine_seed: int,
) -> np.ndarray:
    """
    Use LOOCV to fit len(train_y) independent GPs and sample from their posterior to obtain an
    approximate sample from the target model.

    This sampling does not take into account the correlation between observations which occurs
    when the predictive uncertainty of the Gaussian process is unequal zero.
    """
    masks = np.eye(len(train_x), dtype=np.bool)
    train_x_cv = np.stack([train_x[~m] for m in masks])
    train_y_cv = np.stack([train_y[~m] for m in masks])
    test_x_cv = np.stack([train_x[m] for m in masks])

    samples = np.zeros((num_samples, train_y.shape[0]))
    for i in range(train_y.shape[0]):
        loo_model = get_gaussian_process(
            configspace=model.configspace,
            bounds=model.bounds,
            types=model.types,
            rng=model.rng,
            kernel=model.kernel,
        )
        loo_model._train(X=train_x_cv[i], y=train_y_cv[i], do_optimize=False)

        samples_i = sample_sobol(loo_model, test_x_cv[i], num_samples, engine_seed).flatten()

        samples[:, i] = samples_i

    return samples


def compute_target_model_ranking_loss(
    train_x: np.ndarray,
    train_y: np.ndarray,
    num_samples: int,
    model: GaussianProcess,
    engine_seed: int,
) -> np.ndarray:
    """
    Use LOOCV to fit len(train_y) independent GPs and sample from their posterior to obtain an
    approximate sample from the target model.

    This function does joint draws from all observations (both training data and left out sample)
    to take correlation between observations into account, which can occur if the predictive
    variance of the Gaussian process is unequal zero. To avoid returning a tensor, this function
    directly computes the ranking loss.
    """
    masks = np.eye(len(train_x), dtype=np.bool)
    train_x_cv = np.stack([train_x[~m] for m in masks])
    train_y_cv = np.stack([train_y[~m] for m in masks])

    ranking_losses = np.zeros(num_samples, dtype=np.int)
    for i in range(train_y.shape[0]):
        loo_model = get_gaussian_process(
            configspace=model.configspace,
            bounds=model.bounds,
            types=model.types,
            rng=model.rng,
            kernel=model.kernel,
        )
        loo_model._train(X=train_x_cv[i], y=train_y_cv[i], do_optimize=False)
        samples_i = sample_sobol(loo_model, train_x, num_samples, engine_seed)

        for j in range(len(train_y)):
            ranking_losses += (samples_i[:, i] < samples_i[:, j]) ^ (train_y[i] < train_y[j])

    return ranking_losses


def compute_rank_weights(
        train_x: np.ndarray,
        train_y: np.ndarray,
        base_models: List[GaussianProcess],
        target_model: GaussianProcess,
        num_samples: int,
        sampling_mode: str,
        weight_dilution_strategy: Union[int, Callable],
        number_of_function_evaluations,
        rng: np.random.RandomState,
        alpha: float = 0.0,
) -> np.ndarray:
    """
    Compute ranking weights for each base model and the target model
    (using LOOCV for the target model).

    Returns
    -------
    weights : np.ndarray
    """

    if sampling_mode == 'bootstrap':

        predictions = []
        for model_idx in range(len(base_models)):
            model = base_models[model_idx]
            predictions.append(model.predict(train_x)[0].flatten())

        masks = np.eye(len(train_x), dtype=np.bool)
        train_x_cv = np.stack([train_x[~m] for m in masks])
        train_y_cv = np.stack([train_y[~m] for m in masks])
        test_x_cv = np.stack([train_x[m] for m in masks])

        loo_prediction = []
        for i in range(train_y.shape[0]):
            loo_model = get_gaussian_process(
                configspace=target_model.configspace,
                bounds=target_model.bounds,
                types=target_model.types,
                rng=target_model.rng,
                kernel=target_model.kernel,
            )
            loo_model._train(X=train_x_cv[i], y=train_y_cv[i], do_optimize=False)
            loo_prediction.append(loo_model.predict(test_x_cv[i])[0][0][0])
        predictions.append(loo_prediction)
        predictions = np.array(predictions)

        bootstrap_indices = rng.choice(predictions.shape[1],
                                       size=(num_samples, predictions.shape[1]),
                                       replace=True)

        bootstrap_predictions = []
        bootstrap_targets = train_y[bootstrap_indices].reshape((num_samples, len(train_y)))
        for m in range(len(base_models) + 1):
            bootstrap_predictions.append(predictions[m, bootstrap_indices])

        ranking_losses = np.zeros((len(base_models) + 1, num_samples))
        for i in range(len(base_models)):

            for j in range(len(train_y)):
                ranking_losses[i] += np.sum(
                    (
                        roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i])
                        ^ (roll_col(bootstrap_targets, j) < bootstrap_targets
                    ), axis=1
                )
        for j in range(len(train_y)):
            ranking_losses[-1] += np.sum(
                (
                    (roll_col(bootstrap_predictions[-1], j) < bootstrap_targets)
                    ^ (roll_col(bootstrap_targets, j) < bootstrap_targets)
                ), axis=1
            )

    elif sampling_mode in ['simplified', 'correct']:
        # Use the original strategy as described in v1: https://arxiv.org/pdf/1802.02219v1.pdf
        ranking_losses = []
        # compute ranking loss for each base model
        for model_idx in range(len(base_models)):
            model = base_models[model_idx]
            # compute posterior over training points for target task
            f_samps = sample_sobol(model, train_x, num_samples, rng.randint(10000))
            # compute and save ranking loss
            ranking_losses.append(compute_ranking_loss(f_samps, train_y, target_model=False))

        # compute ranking loss for target model using LOOCV
        if sampling_mode == 'simplified':
            # Independent draw of the leave one out sample, other "samples" are noise-free and the
            # actual observation
            f_samps = get_target_model_loocv_sample_preds(train_x, train_y, num_samples, target_model,
                                                          rng.randint(10000))
            ranking_losses.append(compute_ranking_loss(f_samps, train_y, target_model=True))
        elif sampling_mode == 'correct':
            # Joint draw of the leave one out sample and the other observations
            ranking_losses.append(
                compute_target_model_ranking_loss(train_x, train_y, num_samples, target_model,
                                                  rng.randint(10000))
            )
        else:
            raise ValueError(sampling_mode)
    else:
        raise NotImplementedError(sampling_mode)

    if isinstance(weight_dilution_strategy, int):
        weight_dilution_percentile_target = weight_dilution_strategy
        weight_dilution_percentile_base = 50
    elif weight_dilution_strategy is None or weight_dilution_strategy in ['probabilistic', 'probabilistic-ld']:
        pass
    else:
        raise ValueError(weight_dilution_strategy)

    ranking_loss = np.array(ranking_losses)

    # perform model pruning
    p_drop = []
    if weight_dilution_strategy in ['probabilistic', 'probabilistic-ld']:
        for i in range(len(base_models)):
            better_than_target = np.sum(ranking_loss[i, :] < ranking_loss[-1, :])
            worse_than_target = np.sum(ranking_loss[i, :] >= ranking_loss[-1, :])
            correction_term = alpha * (better_than_target + worse_than_target)
            proba_keep = better_than_target / (better_than_target + worse_than_target + correction_term)
            if weight_dilution_strategy == 'probabilistic-ld':
                proba_keep = proba_keep * (1 - len(train_x) / float(number_of_function_evaluations))
            proba_drop = 1 - proba_keep
            p_drop.append(proba_drop)
            r = rng.rand()
            if r < proba_drop:
                ranking_loss[i, :] = np.max(ranking_loss) * 2 + 1
    elif weight_dilution_strategy is not None:
        # Use the original strategy as described in v1: https://arxiv.org/pdf/1802.02219v1.pdf
        percentile_base = np.percentile(ranking_loss[: -1, :], weight_dilution_percentile_base, axis=1)
        percentile_target = np.percentile(ranking_loss[-1, :], weight_dilution_percentile_target)
        for i in range(len(base_models)):
            if percentile_base[i] >= percentile_target:
                ranking_loss[i, :] = np.max(ranking_loss) * 2 + 1

    # compute best model (minimum ranking loss) for each sample
    # this differs from v1, where the weight is given only to the target model in case of a tie.
    # Here, we distribute the weight fairly among all participants of the tie.
    minima = np.min(ranking_loss, axis=0)
    assert len(minima) == num_samples
    best_models = np.zeros(len(base_models) + 1)
    for i, minimum in enumerate(minima):
        minimum_locations = ranking_loss[:, i] == minimum
        sample_from = np.where(minimum_locations)[0]

        for sample in sample_from:
            best_models[sample] += 1. / len(sample_from)

    # compute proportion of samples for which each model is best
    rank_weights = best_models / num_samples
    return rank_weights, p_drop


class RGPE(AbstractEPM):

    def __init__(
        self,
        training_data: Dict[int, Dict[str, Union[List[Configuration], np.ndarray]]],
        num_posterior_samples: int,
        weight_dilution_strategy: Union[int, str],
        number_of_function_evaluations: int,
        sampling_mode: str = 'correct',
        variance_mode: str = 'average',
        normalization: str = 'mean/var',
        alpha: float = 0.0,
        **kwargs
    ):
        """Ranking-Weighted Gaussian Process Ensemble.

        Parameters
        ----------
        training_data
            Dictionary containing the training data for each meta-task. Mapping from an integer (
            task ID) to a dictionary, which is a mapping from configuration to performance.
        num_posterior_samples
            Number of samples to draw for approximating the posterior probability of a model
            being the best model to explain the observations on the target task.
        weight_dilution_strategy
            Can be one of the following four:
            * ``'probabilistic-ld'``: the method presented in the paper
            * ``'probabilistic'``: the method presented in the paper, but without the time-dependent
              pruning of meta-models
            * an integer: a deterministic strategy described in https://arxiv.org/abs/1802.02219v1
            * ``None``: no weight dilution prevention
        number_of_function_evaluations
            Optimization horizon - used to compute the time-dependent factor in the probability
            of dropping base models for the weight dilution prevention strategy
            ``'probabilistic-ld'``.
        sampling_mode
            Can be any of:
            * ``'bootstrap'``
            * ``'correct'``
            * ``'simplified'``
        variance_mode
            Can be either ``'average'`` to return the weighted average of the variance
            predictions of the individual models or ``'target'`` to only obtain the variance
            prediction of the target model. Changing this is only necessary to use the model
            together with the expected improvement.
        normalization
            Can be either:
            * ``None``: No normalization per task
            * ``'mean/var'``: Zero mean unit standard deviation normalization per task as
              proposed by Yogatama et al. (AISTATS 2014).
            * ``'Copula'``: Copula transform as proposed by Salinas et al., 2020
        alpha
            Regularization hyperparameter to increase aggressiveness of dropping base models when
            using the weight dilution strategies ``'probabilistic-ld'`` or ``'probabilistic'``.
        """

        if kwargs.get('instance_features') is not None:
            raise NotImplementedError()
        super().__init__(**kwargs)
        self.training_data = training_data

        self.number_of_function_evaluations = number_of_function_evaluations
        self.num_posterior_samples = num_posterior_samples
        self.rng = np.random.RandomState(self.seed)
        self.sampling_mode = sampling_mode
        self.variance_mode = variance_mode
        self.normalization = normalization
        self.alpha = alpha

        if self.normalization not in ['None', 'mean/var', 'Copula']:
            raise ValueError(self.normalization)

        if weight_dilution_strategy is None or weight_dilution_strategy == 'None':
            weight_dilution_strategy = None
        elif weight_dilution_strategy in ['probabilistic', 'probabilistic-ld']:
            pass
        else:
            weight_dilution_strategy = int(weight_dilution_strategy)

        self.weight_dilution_strategy = weight_dilution_strategy

        base_models = []
        for task in training_data:
            model = get_gaussian_process(
                bounds=self.bounds,
                types=self.types,
                configspace=self.configspace,
                rng=self.rng,
                kernel=None,
            )
            y = training_data[task]['y']
            if self.normalization == 'mean/var':
                mean = y.mean()
                std = y.std()
                if std == 0:
                    std = 1

                y_scaled = (y - mean) / std
                y_scaled = y_scaled.flatten()
            elif self.normalization == 'Copula':
                y_scaled = copula_transform(y)
            elif self.normalization == 'None':
                y_scaled = y
            else:
                raise ValueError(self.normalization)
            configs = training_data[task]['configurations']
            X = convert_configurations_to_array(configs)

            model.train(
                X=X,
                Y=y_scaled,
            )
            base_models.append(model)
        self.base_models = base_models
        self.weights_over_time = []
        self.p_drop_over_time = []

    def _train(self, X: np.ndarray, Y: np.ndarray) -> AbstractEPM:
        """SMAC training function"""
        print(self.normalization)
        if self.normalization == 'mean/var':
            Y = Y.flatten()
            mean = Y.mean()
            std = Y.std()
            if std == 0:
                std = 1

            y_scaled = (Y - mean) / std
            self.Y_std_ = std
            self.Y_mean_ = mean
        elif self.normalization in ['None', 'Copula']:
            self.Y_mean_ = 0.
            self.Y_std_ = 1.
            y_scaled = Y
            if self.normalization == 'Copula':
                y_scaled = copula_transform(Y)
        else:
            raise ValueError(self.normalization)

        target_model = get_gaussian_process(
            bounds=self.bounds,
            types=self.types,
            configspace=self.configspace,
            rng=self.rng,
            kernel=None,
        )
        self.target_model = target_model.train(X, y_scaled)
        self.model_list_ = self.base_models + [target_model]

        if X.shape[0] < 3:
            self.weights_ = np.ones(len(self.model_list_)) / len(self.model_list_)
            p_drop = np.ones((len(self.base_models, ))) * np.NaN
        else:
            try:
                self.weights_, p_drop = compute_rank_weights(
                    train_x=X,
                    train_y=y_scaled,
                    base_models=self.base_models,
                    target_model=target_model,
                    num_samples=self.num_posterior_samples,
                    sampling_mode=self.sampling_mode,
                    weight_dilution_strategy=self.weight_dilution_strategy,
                    number_of_function_evaluations=self.number_of_function_evaluations,
                    rng=self.rng,
                    alpha=self.alpha,
                )
            except Exception as e:
                print(e)
                self.weights_ = np.zeros((len(self.model_list_, )))
                self.weights_[-1] = 1
                p_drop = np.ones((len(self.base_models, ))) * np.NaN

        print('Weights', self.weights_)
        self.weights_over_time.append(self.weights_)
        self.p_drop_over_time.append(p_drop)

        return self

    def _predict(self, X: np.ndarray, cov_return_type='diagonal_cov') -> Tuple[np.ndarray, np.ndarray]:
        """SMAC predict function"""

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
                weighted_covars.append(covar * weight ** 2)
            elif self.variance_mode == 'target':
                if raw_idx + 1 == len(self.weights_):
                    weighted_covars.append(covar)
            else:
                raise ValueError()

        if len(weighted_covars) == 0:
            if self.variance_mode != 'target':
                raise ValueError(self.variance_mode)
            _, covar = self.model_list_[-1]._predict(X, cov_return_type=cov_return_type)
            weighted_covars.append(covar)

        mean_x = np.sum(np.stack(weighted_means), axis=0) * self.Y_std_ + self.Y_mean_
        covar_x = np.sum(weighted_covars, axis=0) * (self.Y_std_ ** 2)
        return mean_x, covar_x

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        """
        Sample function values from the posterior of the specified test points.
        """

        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights_ ** 2 > 0).nonzero()[0]
        non_zero_weights = self.weights_[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        samples = []
        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            weight = non_zero_weights[non_zero_weight_idx]

            funcs = self.model_list_[raw_idx].sample_functions(X_test, n_funcs)
            funcs = funcs * weight
            samples.append(funcs)
        samples = np.sum(samples, axis=0)
        return samples
