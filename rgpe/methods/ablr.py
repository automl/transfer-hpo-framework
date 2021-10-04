from typing import Dict, List, Tuple, Union

from ConfigSpace import Configuration
import numpy as np
import scipy.optimize
import torch.nn as nn
import torch

from smac.configspace import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM

from rgpe.utils import copula_transform

D = 50  # Hidden layer size

precision = 32
if precision == 32:
    t_dtype = torch.float32
    np_dtype = np.float32
else:
    t_dtype = torch.float64
    np_dtype = np.float64


class Net(torch.nn.Module):
    """
    Implementation of the Adaptive Bayesian Linear Regression (ABLR) for multi-task
    hyperparameter optimization.

    For details see https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning.pdf

    This class implements the neural network. For the class connecting it to SMAC see below."""

    def __init__(self, num_tasks, n_attributes, meta_data=None, use_copula_transform=False):

        self.num_tasks = num_tasks
        self.n_attributes = n_attributes
        self.meta_data = meta_data
        self.use_copula_transform = use_copula_transform

        self.mean_ = None
        self.std_ = None

        super().__init__()
        self.total_n_params = 0

        hidden1 = nn.Linear(self.n_attributes, D)
        hidden2 = nn.Linear(D, D)
        hidden3 = nn.Linear(D, D)
        self.layers = [
            hidden1, hidden2, hidden3
        ]
        if precision == 32:
            self.layers = [layer.float() for layer in self.layers]
        else:
            self.layers = [layer.double() for layer in self.layers]

        # initialization of alpha and beta
        # Instead of alpha, we model 1/alpha and use a different range for the values
        # (i.e. 1e-6 to 1 instead of 1 to 1e6)
        self.alpha_t = torch.tensor([1] * self.num_tasks, requires_grad=True, dtype=t_dtype)
        self.total_n_params += len(self.alpha_t)
        self.beta_t = torch.tensor([1e3] * self.num_tasks, requires_grad=True, dtype=t_dtype)
        self.total_n_params += len(self.beta_t)

        # initialization of the weights
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weight)
            if len(layer.weight.shape) == 1:
                size = layer.weight.shape[0]
            else:
                size = layer.weight.shape[0] * layer.weight.shape[1]
            self.total_n_params += size

        # initialize arrays for the optimization of sum log-likelihood
        self.K_t = [torch.tensor(0.0, dtype=t_dtype) for i in range(self.num_tasks)]
        self.L_t = [torch.tensor(0.0, dtype=t_dtype) for i in range(self.num_tasks)]
        self.L_t_inv = [torch.tensor(0.0, dtype=t_dtype) for i in range(self.num_tasks)]
        self.e_t = [torch.tensor(0.0, dtype=t_dtype) for i in range(self.num_tasks)]

    def forward(self, x):
        """
        Simple forward pass through the neural network
        """

        for layer in self.layers:
            x = layer(x)
            x = torch.tanh(x)

        return x

    def loss(self, hp, training_datasets):
        """
        Negative log marginal likelihood of multi-task ABLR
        hp : np.ndarray
            Contains the weights of the network, alpha and beta
        training_datasets : list
            tuples (X, y) for the meta-datasets and the current dataset
        """
        # Apply the flattened hyperparameter array to the neural network

        if precision == 32:
            hp = hp.astype(np.float32)

        idx = 0
        for layer in self.layers:
            weights = layer.weight.data.numpy().astype(np_dtype)
            if len(weights.shape) == 1:
                size = weights.shape[0]
            else:
                size = weights.shape[0] * weights.shape[1]
            layer.weight.data = torch.from_numpy(hp[idx: idx + size].reshape(weights.shape))
            layer.weight.requires_grad_()
            idx += size

        self.alpha_t.data = torch.from_numpy(hp[idx: idx + self.num_tasks])
        idx += self.num_tasks
        self.alpha_t.requires_grad_()
        self.beta_t.data = torch.from_numpy(hp[idx: idx + self.num_tasks])
        idx += self.num_tasks
        self.beta_t.requires_grad_()
        assert idx == self.total_n_params

        # Likelihood computation starts here
        self.likelihood = None

        for i, (x, y) in enumerate(training_datasets):

            out = self.forward(x)

            # Loss function calculations, see 6th Equation on the first page of the Appendix
            # https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning-supplemental.zip
            assert (torch.t(out).shape == (D, x.shape[0]))
            # Remember that we model 1/alpha instead of alpha
            r = self.beta_t[i] * self.alpha_t[i]
            K_t = torch.add(
                torch.eye(D, dtype=t_dtype),
                r * torch.matmul(torch.t(out), out)
            )
            self.K_t[i] = K_t.clone()
            assert (K_t.shape == (D, D))

            L_t = torch.cholesky(K_t, upper=False)
            self.L_t[i] = L_t.clone()
            # Naive version:
            # self.L_t_inv[i] = torch.inverse(L_t)
            # e_t = torch.matmul(self.L_t_inv[i], torch.matmul(torch.t(out), y))
            e_t = torch.triangular_solve(torch.matmul(torch.t(out), y), L_t, upper=False).solution
            self.e_t[i] = e_t.view((D, 1)).clone()
            assert (self.e_t[i].shape == (D, 1))

            norm_y_t = torch.norm(y, 2, 0)
            norm_c_t = torch.norm(e_t[i], 2, 0)

            L1 = -(x.shape[0] / 2 * torch.log(self.beta_t[i]))
            L2 = self.beta_t[i] / 2 * (torch.pow(norm_y_t, 2) -r * torch.pow(norm_c_t, 2))
            L3 = torch.sum(torch.log(torch.diag(L_t)))
            L = L1 + L2 + L3

            if self.likelihood is None:
                self.likelihood = L
            else:
                self.likelihood = torch.add(self.likelihood, L)

        # Get the gratient and put transform it into the flat array structure required by
        # scipy.optimize
        g = np.zeros((self.total_n_params))
        self.likelihood.backward()

        idx = 0
        for layer in self.layers:
            gradients = layer.weight.grad.data.numpy().astype(np_dtype)
            if len(gradients.shape) == 1:
                size = gradients.shape[0]
            else:
                size = gradients.shape[0] * gradients.shape[1]
            g[idx: idx + size] = gradients.flatten()
            idx += size
            layer.weight.grad.zero_()

        g[idx: idx + self.num_tasks] = self.alpha_t.grad.data.numpy().astype(np_dtype)
        idx += self.num_tasks
        g[idx: idx + self.num_tasks] = self.beta_t.grad.data.numpy().astype(np_dtype)
        idx += self.num_tasks
        self.alpha_t.grad.data.zero_()
        self.beta_t.grad.data.zero_()
        self._gradient = g

        return self.likelihood

    def gradient(self, hp, training_datasets):
        """
        Gradient of the parameters of the network that are optimized through LBFGS

        The gradient is actually stored during the forward pass, this is only a convenience
        function to work with the LBFGS interface of scipy.
        """

        return self._gradient

    def optimize(self, training_datasets):
        """
        Optimize weights, alpha and beta with LBFGSB
        """

        # Initial flattened array of weights used as a starting point of LBFGS
        init = np.ones((self.total_n_params), dtype=np_dtype)

        idx = 0
        for layer in self.layers:
            weights = layer.weight.data.numpy().astype(np_dtype)
            if len(weights.shape) == 1:
                size = weights.shape[0]
            else:
                size = weights.shape[0] * weights.shape[1]
            init[idx: idx + size] = weights.flatten()
            idx += size
        mybounds = [[None, None] for i in range(idx)]

        init[idx: idx + self.num_tasks] = self.alpha_t.data.numpy().astype(np_dtype)
        idx += self.num_tasks
        mybounds.extend([[1e-3, 1e3]] * self.num_tasks)
        init[idx: idx + self.num_tasks] = self.beta_t.data.numpy().astype(np_dtype)
        idx += self.num_tasks
        mybounds.extend([[1, 1e6]] * self.num_tasks)

        assert self.total_n_params == len(mybounds), (self.total_n_params, len(mybounds))
        assert self.total_n_params == idx

        res = scipy.optimize.fmin_l_bfgs_b(
            lambda *args: float(self.loss(*args)),
            x0=init,
            bounds=mybounds,
            fprime=self.gradient,
            args=(training_datasets, ),
        )
        print(self.loss(res[0], training_datasets))  # This updates the internal states
        print(res)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Optimize the neural network given training data ``X``.

        Training data is concatenated with meta-data and then passed to the optimize function.
        """
        y = y.reshape((y.shape[0], 1))

        training_datasets = []
        for meta_task in self.meta_data:
            meta_task_data = self.meta_data[meta_task]
            X_t = meta_task_data[0]
            y_t = meta_task_data[1]
            if X_t.shape[1] != self.n_attributes:
                raise ValueError((X_t.shape[1], self.n_attributes))

            if self.use_copula_transform:
                y_t = copula_transform(y_t)
            else:
                mean = y_t.mean()
                std = y_t.std()
                if std == 0:
                    std = 1
                y_t = (y_t.copy() - mean) / std
                y_t = y_t.reshape(y_t.shape[0], 1)

            training_datasets.append((
                torch.tensor(X_t, dtype=t_dtype),
                torch.tensor(y_t, dtype=t_dtype),
            ))

        if X.shape[1] != self.n_attributes:
            raise ValueError((X.shape[1], self.n_attributes))

        if self.use_copula_transform:
            self.mean_ = 0
            self.std_ = 1
            y_ = copula_transform(y.copy())
        else:
            self.mean_ = y.mean()
            self.std_ = y.std()
            if self.std_ == 0:
                self.std_ = 1
            y_ = (y.copy() - self.mean_) / self.std_

        training_datasets.append((
            torch.tensor(X, dtype=t_dtype),
            torch.tensor(y_, dtype=t_dtype),
        ))
        if len(training_datasets) != self.num_tasks:
            raise ValueError((len(training_datasets), self.num_tasks))

        self.optimize(training_datasets)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the predictive mean and variance of the objective function at
        the given test points.
        """
        X_test = torch.tensor(X_test, dtype=t_dtype)
        out = self.forward(X_test)

        # Naive implementation:
        #m = torch.matmul(torch.matmul(torch.t(self.e_t[-1]), self.L_t_inv[-1]), torch.t(out))
        m = torch.matmul(
            torch.t(self.e_t[-1]),
            torch.triangular_solve(torch.t(out), self.L_t[-1], upper=False).solution,
        )
        # Remember that we model 1/alpha instead of alpha
        m = (self.beta_t[-1] * self.alpha_t[-1]) * m.reshape((m.shape[1], 1))
        assert (m.shape == (X_test.shape[0], 1))
        if not torch.isfinite(m).all():
            raise ValueError('Infinite predictions %s for input %s' % (m, X_test))
        m = m * self.std_ + self.mean_

        # Naive implementation
        #v = torch.matmul(self.L_inv_t[-1], torch.t(out))
        v = torch.triangular_solve(torch.t(out), self.L_t[-1], upper=False).solution
        # Remember that we model 1/alpha instead of alpha
        v = self.alpha_t[-1] * torch.pow(torch.norm(v, dim=0), 2)
        v = v.reshape((-1, 1))
        assert (v.shape == (X_test.shape[0], 1)), v.shape
        if not torch.isfinite(v).all():
            raise ValueError('Infinite predictions %s for input %s' % (v, X_test))
        v = v * (self.std_ ** 2)

        return m.detach().numpy(), v.detach().numpy()


class ABLR(AbstractEPM):
    """
    Implementation of the Adaptive Bayesian Linear Regression (ABLR) for multi-task
    hyperparameter optimization.

    For details see https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning.pdf

    This is the wrapper class to be used with SMAC, which internally uses the neural network
    class in the code above.
    """

    def __init__(
            self,
            training_data: Dict[int, Dict[str, Union[List[Configuration], np.ndarray]]],
            use_copula_transform: bool = False,
            **kwargs
    ):
        if kwargs.get('instance_features') is not None:
            raise NotImplementedError()
        super().__init__(**kwargs)
        self.training_data = training_data
        self.use_copula_transform = use_copula_transform
        self.nn = None
        torch.manual_seed(self.seed)
        self.rng = np.random.RandomState(self.seed)

        self.categorical_mask = np.array(self.types) > 0
        self.n_categories = np.sum(self.types)

    def _train(self, X: np.ndarray, Y: np.ndarray) -> AbstractEPM:
        meta_data = dict()
        for id_ in self.training_data:
            configs = self.training_data[id_]['configurations']
            X_ = convert_configurations_to_array(configs)
            X_ = self._preprocess(X_)
            meta_data[id_] = (
                X_,
                self.training_data[id_]['y'].flatten(),
                None,
            )

        X = self._preprocess(X)
        for i in range(10):
            try:
                # Sometimes the neural network training fails due to numerical issues - we
                # then retrain the network from scratch
                if self.nn is None:
                    self.nn = Net(
                        num_tasks=len(self.training_data) + 1,
                        n_attributes=X.shape[1],
                        meta_data=meta_data,
                        use_copula_transform=self.use_copula_transform,
                    )
                self.nn.train(X, Y)
                break
            except Exception as e:
                print('Training failed %d/%d!' % (i + 1, 10))
                print(e)
                self.nn = None

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = self._preprocess(X)
        if self.nn:
            return self.nn.predict(X)
        else:
            return self.rng.randn(X.shape[0], 1), self.rng.randn(X.shape[0], 1)

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
