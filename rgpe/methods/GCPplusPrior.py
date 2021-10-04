from typing import Dict, List,  Tuple, Union

from ConfigSpace import Configuration
import numpy as np
from scipy.stats import norm
import torch.nn as nn
import torch

from smac.configspace import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from rgpe.utils import get_gaussian_process, copula_transform


class GCPplusPrior(AbstractEPM):

    def __init__(
        self,
        training_data: Dict[int, Dict[str, Union[List[Configuration], np.ndarray]]],
        **kwargs
    ):
        """
        Gaussian Copula Process plus prior from "A Quantile-based Approach for Hyperparameter
        Transfer Learning" by  Salinas, Shen and Perrone, ICML 2020,
        https://proceedings.icml.cc/static/paper_files/icml/2020/4367-Paper.pdf

        This is a re-implementation that is not based on the original code which can be found at
        https://github.com/geoalgo/A-Quantile-based-Approach-for-Hyperparameter-Transfer-Learning

        Parameters
        ----------
        training_data
            Dictionary containing the training data for each meta-task. Mapping from an integer (
            task ID) to a dictionary, which is a mapping from configuration to performance.
        """

        if kwargs.get('instance_features') is not None:
            raise NotImplementedError()
        super().__init__(**kwargs)
        self.training_data = training_data

        self.categorical_mask = np.array(self.types) > 0
        self.n_categories = np.sum(self.types)

        torch.manual_seed(self.seed)
        self.rng = np.random.RandomState(self.seed)

        X_train = []
        y_train = []
        for task in training_data:
            Y = training_data[task]['y']
            y_scaled = copula_transform(Y)
            configs = training_data[task]['configurations']
            X = convert_configurations_to_array(configs)
            for x, y in zip(X, y_scaled):
                X_train.append(x)
                y_train.append(y)
        X_train = np.array(X_train)
        X_train = self._preprocess(X_train)
        y_train = np.array(y_train)

        class NLLHLoss(nn.Module):

            def forward(self, input, target):
                # Assuming network outputs var
                std = torch.log(1 + torch.exp(input[:, 1])) + 10e-12
                mu = input[:, 0].view(-1, 1)

                # Pytorch Normal indeed takes the standard deviation as argument
                n = torch.distributions.normal.Normal(mu, std)
                loss = n.log_prob(target)
                return -torch.mean(loss)

        # TODO we could add embeddings for categorical hyperparameters here to improve performance?
        model = torch.nn.Sequential(
            torch.nn.Linear(X_train.shape[1], 50).float(),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50).float(),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50).float(),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 2).float(),
        )
        loss_fn = NLLHLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=1000,
                                                    gamma=0.2)
        for iter in range(3000):

            batch = self.rng.choice(len(X_train), size=64)
            x_batch = torch.tensor(X_train[batch]).float()
            y_batch = torch.tensor(y_train[batch]).float()

            y_pred = model(x_batch)

            # Compute and print loss.
            loss = loss_fn(y_pred, y_batch)
            if iter % 100 == 99:
                print(iter, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.prior = model

    def _train(self, X: np.ndarray, Y: np.ndarray) -> AbstractEPM:
        X = self._preprocess(X)
        prior_prediction = self.prior.forward(torch.tensor(X).float()).detach().numpy()
        prior_mean = prior_prediction[:, 0].flatten()
        prior_std = prior_prediction[:, 1].flatten()
        prior_std = np.log(1 + np.exp(prior_std)) + 10e-12

        y_scaled = copula_transform(Y).flatten()
        residual = (y_scaled - prior_mean) / prior_std

        self.target_model = get_gaussian_process(
            bounds=self.bounds,
            types=self.types,
            configspace=self.configspace,
            rng=self.rng,
            kernel=None,
        )
        self.target_model._train(X, residual)

        return self

    def predict(self, X: np.ndarray, cov_return_type: str = 'diagonal_cov') -> Tuple[np.ndarray, np.ndarray]:
        X = self._preprocess(X)
        prior_prediction = self.prior.forward(torch.tensor(X).float()).detach().numpy()
        prior_mean = prior_prediction[:, 0]
        prior_std = prior_prediction[:, 1]
        prior_std = (np.log(1 + np.exp(prior_std)) + 10e-12)
        gp_mean, gp_var = self.target_model._predict(X)
        mean_x = gp_mean * prior_std + prior_mean
        covar_x = np.sqrt(gp_var) * prior_std
        return mean_x.reshape((-1, 1)), covar_x.reshape((-1, 1))

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Perform one-hot-encoding of categorical hyperparameters."""
        categories_array = np.zeros((X.shape[0], self.n_categories))
        categories_idx = 0
        for idx in range(len(self.types)):
            if self.types[idx] == 0:
                continue
            else:
                for j in range(self.types[idx]):
                    mask = X[:, idx] == j
                    categories_array[mask, categories_idx] = 1
                    categories_idx += 1
        numerical_array = X[:, ~self.categorical_mask]
        X = np.concatenate((numerical_array, categories_array), axis=1)
        X[np.isnan(X)] = -1.0
        return X


class CustomEI(AbstractAcquisitionFunction):
    """EI for residual GP as defined in Section 4.2 of Salinas et al."""

    def __init__(self, model: AbstractEPM):

        super().__init__(model)
        self.eta = None
        self._required_updates = ('model', 'eta')

    def _compute(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.eta - m) / v
            return v * (z * norm.cdf(z) + norm.pdf(z))

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
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
