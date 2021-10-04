import functools
import json
from typing import Dict, List, Optional, Tuple, Union

from ConfigSpace import Configuration
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.optimize
import sklearn.metrics
from smac.configspace import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM

from rgpe.utils import get_gaussian_process

# Code from https://github.com/HIPS/Spearmint/blob/PESC/spearmint/acquisition_functions/predictive_entropy_search.py#L944
"""
See Miguel's paper (http://arxiv.org/pdf/1406.2541v1.pdf) section 2.1 and Appendix A
Returns a function the samples from the approximation...
if testing=True, it does not return the result but instead the random cosine for testing only
We express the kernel as an expectation. But then we approximate the expectation with a weighted sum
theta are the coefficients for this weighted sum. that is why we take the dot product of theta at 
the end
we also need to scale at the end so that it's an average of the random features. 
if use_woodbury_if_faster is False, it never uses the woodbury version
"""

def chol2inv(chol):
    return spla.cho_solve((chol, False), np.eye(chol.shape[0]))


def sample_gp_with_random_features(gp, nFeatures, rng, testing=False, use_woodbury_if_faster=True):
    d = len(gp.configspace.get_hyperparameters())
    N_data = gp.gp.X_train_.shape[0]

    nu2 = np.exp(gp.gp.kernel.theta[-1])

    sigma2 = np.exp(gp.gp.kernel.theta[0])  # the kernel amplitude

    # We draw the random features - in contrast to the original code we only support Matern5/2
    m = 5.0 / 2.0
    W = (
        rng.randn(nFeatures, d) / gp.gp.kernel.theta[1: -1] /
        np.sqrt(rng.gamma(shape=m, scale=1.0 / m, size=(nFeatures, 1)))
    )
    b = rng.uniform(low=0, high=2 * np.pi, size=nFeatures)[:, None]

    # Just for testing the  random features in W and b... doesn't test the weights theta
    if testing:
        return lambda x: np.sqrt(2 * sigma2 / nFeatures) * np.cos(np.dot(W, x.T) + b)
    # K(x1, x2) \approx np.dot(test(x1).T, tst_fun(x2))

    randomness = rng.randn(nFeatures)

    # W has size nFeatures by d
    # tDesignMatrix has size Nfeatures by Ndata
    # woodbury has size Ndata by Ndata
    # z is a vector of length nFeatures

    gp_inputs = gp.gp.X_train_

    # tDesignMatrix has size Nfeatures by Ndata
    tDesignMatrix = np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, gp_inputs.T) + b)

    if use_woodbury_if_faster and N_data < nFeatures:
        # you can do things in cost N^2d instead of d^3 by doing this woodbury thing

        # We obtain the posterior on the coefficients
        woodbury = np.dot(tDesignMatrix.T, tDesignMatrix) + nu2 * np.eye(N_data)
        chol_woodbury = spla.cholesky(woodbury)
        # inverseWoodbury = chol2inv(chol_woodbury)
        z = np.dot(tDesignMatrix, gp.gp.y_train_ / nu2)
        # m = z - np.dot(tDesignMatrix, np.dot(inverseWoodbury, np.dot(tDesignMatrix.T, z)))
        m = z - np.dot(tDesignMatrix,
                       spla.cho_solve((chol_woodbury, False), np.dot(tDesignMatrix.T, z)))
        # (above) alternative to original but with cho_solve

        # z = np.dot(tDesignMatrix, gp.observed_values / nu2)
        # m = np.dot(np.eye(nFeatures) - \
        # np.dot(tDesignMatrix, spla.cho_solve((chol_woodbury, False), tDesignMatrix.T)), z)

        # woodbury has size N_data by N_data
        D, U = npla.eigh(woodbury)
        # sort the eigenvalues (not sure if this matters)
        idx = D.argsort()[::-1]  # in decreasing order instead of increasing
        D = D[idx]
        U = U[:, idx]
        R = 1.0 / (np.sqrt(D) * (np.sqrt(D) + np.sqrt(nu2)))
        # R = 1.0 / (D + np.sqrt(D*nu2))

        # We sample from the posterior of the coefficients
        theta = randomness - \
                np.dot(tDesignMatrix,
                       np.dot(U, (R * np.dot(U.T, np.dot(tDesignMatrix.T, randomness))))) + m

    else:
        # all you are doing here is sampling from the posterior of the linear model
        # that approximates the GP
        # Sigma = matrixInverse(np.dot(tDesignMatrix, tDesignMatrix.T) / nu2 + np.eye(
        # nFeatures))
        # m = np.dot(Sigma, np.dot(tDesignMatrix, gp.observed_values / nu2))
        # theta = m + np.dot(randomness, spla.cholesky(Sigma, lower=False)).T

        # Sigma = matrixInverse(np.dot(tDesignMatrix, tDesignMatrix.T) + nu2*np.eye(nFeatures))
        # m = np.dot(Sigma, np.dot(tDesignMatrix, gp.observed_values))
        # theta = m + np.dot(randomness, spla.cholesky(Sigma*nu2, lower=False)).T

        approx_Kxx = np.dot(tDesignMatrix, tDesignMatrix.T)
        while True:
            try:
                print(approx_Kxx, nu2)
                chol_Sigma_inverse = spla.cholesky(approx_Kxx + nu2 * np.eye(nFeatures))
                break
            except np.linalg.LinAlgError:
                nu2 = np.log(nu2)
                nu2 += 1
                nu2 = np.exp(nu2)
        Sigma = chol2inv(chol_Sigma_inverse)
        m = spla.cho_solve((chol_Sigma_inverse, False),
                           np.dot(tDesignMatrix, gp.gp.y_train_))
        theta = m + np.dot(randomness, spla.cholesky(Sigma * nu2, lower=False)).T
        # the above commented out version might be less stable? i forget why i changed it
        # that's ok.

    def wrapper(gradient, x):
        # the argument "gradient" is
        # not the usual compute_grad that computes BOTH when true
        # here it only computes the objective when true
        if x.ndim == 1:
            x = x[None]

        if not gradient:
            result = np.dot(theta.T, np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, x.T) + b))
            if result.size == 1:
                result = float(
                    result)  # if the answer is just a number, take it out of the numpy array
                # wrapper
                # (failure to do so messed up NLopt and it only gives a cryptic error message)
            return result
        else:
            grad = np.dot(theta.T,
                          -np.sqrt(2.0 * sigma2 / nFeatures) * np.sin(np.dot(W, x.T) + b) * W)
            return grad

    return wrapper


class KLWeighting(AbstractEPM):

    """Weighting method from "Information-theoretic Transfer Learning framework for Bayesian
    optimization" by Ramachandran et al., MLKDD 2018

    This does not implement PES!
    """

    def __init__(
        self,
        training_data: Dict[int, Dict[str, Union[List[Configuration], np.ndarray]]],
        eta: float,  # https://github.com/AnilRamachandran/ITTLBO/blob/master/BO_TL_PES_loop.m#L218
        variance_mode: str = 'target',
        ** kwargs
    ):
        if kwargs.get('instance_features') is not None:
            raise NotImplementedError()
        super().__init__(**kwargs)
        self.training_data = training_data
        self.eta = eta

        self.rng = np.random.RandomState(self.seed)
        self.variance_mode = variance_mode

        # https://github.com/AnilRamachandran/ITTLBO/blob/master/BO_TL_PES_loop.m#L153
        self.num_samples = 100
        self.num_features = 500

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

        self.weights_over_time = []

        bounds = [(0, 1)] * len(self.configspace.get_hyperparameters())
        samples = []
        for s in range(len(base_models)):
            samples_base_task = []
            for _ in range(self.num_samples):
                x0 = self.configspace.sample_configuration().get_array()
                base_gp_sample = sample_gp_with_random_features(self.base_models[s], self.num_features,
                                                                self.rng)
                opt_base = scipy.optimize.minimize(functools.partial(base_gp_sample, False), x0,
                                                   jac=functools.partial(base_gp_sample, True),
                                                   bounds=bounds)
                samples_base_task.append(opt_base.x)

            samples.append(np.array(samples_base_task))
        self.samples = samples

    def _compute_weights(self):

        pseudo_weights = []
        bounds = [(0, 1)] * len(self.configspace.get_hyperparameters())

        samples_target_task = []
        for _ in range(self.num_samples):
            target_gp_sample = sample_gp_with_random_features(self.target_model, self.num_features,
                                                              self.rng)
            x0 = self.configspace.sample_configuration().get_array()
            opt_target = scipy.optimize.minimize(functools.partial(target_gp_sample, False), x0,
                                                 jac=functools.partial(target_gp_sample, True),
                                                 bounds=bounds)
            samples_target_task.append(opt_target.x)
        samples_target_task = np.array(samples_target_task)

        for s in range(len(self.model_list_)):
            if s == len(self.model_list_) - 1:
                pseudo_weights.append(1)
            else:
                exp_arg = 0

                masks = np.eye(self.num_samples, dtype=bool)
                for i in range(self.num_samples):
                    samples_base_task = self.samples[s]
                    tau_i = sklearn.metrics.pairwise_distances(
                        samples_target_task[i].reshape((1, -1)),
                        Y=samples_base_task, metric='euclidean').min() + 1e-14
                    rho_i = sklearn.metrics.pairwise_distances(
                        samples_base_task[i].reshape((1, -1)),
                        Y=samples_base_task[~masks[i]], metric='euclidean'
                    ).min() + 1e-14
                    exp_arg += np.log(tau_i / rho_i)
                    #print(tau_i, rho_i, np.log(tau_i / rho_i))

                exp_arg *= (len(self.configspace.get_hyperparameters()) / self.num_samples)
                exp_arg += np.log(self.num_samples / (self.num_samples - 1))
                #print(exp_arg)
                pseudo_weights.append(np.exp(- exp_arg / self.eta))

        pseudo_weights = np.array(pseudo_weights)
        #print(pseudo_weights)
        self.weights_ = pseudo_weights / np.sum(pseudo_weights)

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
        try:
            self._compute_weights()
        except Exception as e:
            print(e)
            self.weights_ = np.zeros((len(self.model_list_, )))
            self.weights_[-1] = 1
        print('Weights', self.weights_)
        self.weights_over_time.append(self.weights_)

        # create model and acquisition function
        return self

    def _predict(self, X: np.ndarray, cov_return_type: bool) -> Tuple[np.ndarray, np.ndarray]:

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
            mean, covar = self.model_list_[raw_idx]._predict(X, cov_return_type=cov_return_type)
            weighted_means.append(weight * mean)
            if self.variance_mode == 'average':
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
