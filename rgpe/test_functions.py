import gzip
import os
import pickle
import sys
from typing import Dict, Optional

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from hpolib.abstract_benchmark import AbstractBenchmark
import hpolib
import lockfile

import numpy as np
import scipy.optimize


class Alpine1D(AbstractBenchmark):
    """Modified Alpine1D function as used in v1: https://arxiv.org/pdf/1802.02219v1.pdf"""

    def __init__(self, task, load_all=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task

    @AbstractBenchmark._check_configuration
    def objective_function(self, configuration: Configuration, **kwargs) -> Dict:
        x = configuration['x']

        shift = kwargs.get('task')
        if shift is None:
            shift = self.task
        shift = shift * np.pi / 12

        rval = (x * np.sin(x + np.pi + shift) + 0.1 * x)
        return {'function_value': rval}

    def objective_function_test(self, configuration: Configuration, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformFloatHyperparameter('x', -10, 10))
        return cs

    @staticmethod
    def get_meta_information():
        return {
            'num_function_evals': 50,
            'name': 'Modified Alpine 1D',
            'reference': """@inproceedings{feurer-automl18,
  author    = {Matthias Feurer and Benjamin Letham and Eytan Bakshy},
  title     = {Scalable Meta-Learning for Bayesian Optimization using Ranking-Weighted Gaussian Process Ensembles},
  booktitle = {ICML 2018 AutoML Workshop},
  year      = {2018},
  month     = jul,
}
""",
        }

    def get_num_base_tasks(self) -> int:
        return 5

    def get_empirical_f_opt(self, task: Optional[int] = None) -> float:
        """Return the empirical f_opt.

        Because ``get_meta_information`` is a static function it has no access to the actual
        function values predicted by the surrogate. This helper function gives access.

        Returns
        -------
        Configuration
        """
        if task is None:
            task = self.task

        cs = self.get_configuration_space()
        bounds = [(-10, 10)]
        def target(x, task):
            config = Configuration(cs, {'x': x[0]})
            return float(self.objective_function(config, task=task)['function_value'])
        res = scipy.optimize.differential_evolution(
            func=target, bounds=bounds, args=(task, ), popsize=1000, polish=True,
            seed=self.rng,
        )
        return res.fun

    def get_empirical_f_worst(self, task: Optional[int] = None) -> float:
        """Return the empirical f_worst.

        Because ``get_meta_information`` is a static function it has no access to the actual
        function values predicted by the surrogate. This helper function gives access.

        Returns
        -------
        Configuration
        """
        if task is None:
            task = self.task

        cs = self.get_configuration_space()
        bounds = [(-10, 10)]
        def target(x, task):
            try:
                config = Configuration(cs, {'x': x[0]})
                return -float(self.objective_function(config, task=task)['function_value'])
            except:
                return -1e10
        res = scipy.optimize.differential_evolution(
            func=target, bounds=bounds, args=(task, ), popsize=1000, polish=True,
            seed=self.rng,
        )
        return -res.fun

    def get_meta_data(self, num_base_tasks: Optional[int] = None, fixed_grid: Optional[bool] = False):
        # Sample data for each base task
        if num_base_tasks is None:
            num_base_tasks = self.get_num_base_tasks()

        if fixed_grid:
            seed = self.rng.randint(0, 10000)
        else:
            seed = None

        data_by_task = {}
        for task in range(num_base_tasks + 1):
            if task == self.task:
                continue

            cs = self.get_configuration_space()
            if fixed_grid:
                cs.seed(seed)
                num_training_points = 20
            else:
                num_training_points = self.rng.randint(low=15, high=25)
                cs.seed(self.rng.randint(0, 10000))
            configurations = cs.sample_configuration(num_training_points)

            # get observed values
            train_y = [
                self.objective_function(config, task=task)['function_value']
                for config in configurations
            ]
            train_y = np.array(train_y)
            # store training data
            data_by_task[task] = {
                # scale x to [0, 1]
                'configurations': configurations,
                'y': train_y,
            }

        return data_by_task


num_dimensions = 3
class Quadratic(AbstractBenchmark):
    """Quadratic function as used by Perrone et al., 2018"""

    def __init__(self, task, load_all=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self._functions = dict()
        self._sample_coefficients(task)
        self._cache_dir = os.path.join(hpolib._config.data_dir, "artificial", "quadratic")
        try:
            os.makedirs(self._cache_dir)
        except:
            pass

    def _sample_coefficients(self, task):
        rng = np.random.RandomState(task)
        coefficients = rng.rand(3) * (10 - 0.1) + 0.1
        self._functions[task] = coefficients

    @AbstractBenchmark._check_configuration
    def objective_function(self, configuration: Configuration, **kwargs) -> Dict:
        x = []
        for i in range(1, num_dimensions + 1):
            x.append(configuration['x%d' % i])
        x = np.array(x)

        task = kwargs.get('task')
        if task is None:
            task = self.task
        if task not in self._functions:
            self._sample_coefficients(task)
        a, b, c = self._functions[task]

        rval = 0.5 * a * np.linalg.norm(x) ** 2 + b * np.sum(x) + 3 * c
        return {'function_value': rval}

    def objective_function_test(self, configuration: Configuration, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_configuration_space(cls):
        cs = ConfigurationSpace()
        for i in range(1, num_dimensions + 1):
            cs.add_hyperparameter(UniformFloatHyperparameter('x%d' % i, -5, 5))
        return cs

    @staticmethod
    def get_meta_information():
        return {
            'num_function_evals': 50,
            'name': '3D Quadratic Function',
            'reference': """@incollection{NIPS2018_7917,
title = {Scalable Hyperparameter Transfer Learning},
author = {Perrone, Valerio and Jenatton, Rodolphe and Seeger, Matthias W and Archambeau, Cedric},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {6845--6855},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning.pdf}
}
""",
        }

    def get_num_base_tasks(self) -> int:
        return 29

    def get_cache_key(self) -> str:
        return '-'.join([str(float(entry)) for entry in self._functions[self.task]])

    def get_empirical_f_opt(self) -> float:
        """Return the empirical f_opt.

        Because ``get_meta_information`` is a static function it has no access to the actual
        function values predicted by the surrogate. This helper function gives access.

        Returns
        -------
        Configuration
        """

        cache_key = self.get_cache_key()
        opt_file_name = os.path.join(self._cache_dir, cache_key + 'opt')

        while True:
            try:
                if not os.path.exists(opt_file_name):
                    with lockfile.LockFile(opt_file_name, timeout=10):

                        cs = self.get_configuration_space()
                        bounds = [(-5, 5)] * num_dimensions

                        def target(x, task):
                            try:
                                config = Configuration(
                                    cs,
                                    {'x%d' % (i + 1): x[i] for i in range(num_dimensions)},
                                )
                                return float(
                                    self.objective_function(config, task=task)['function_value'])
                            except:
                                return 1e10

                        res = scipy.optimize.differential_evolution(
                            func=target, bounds=bounds, args=(self.task,), popsize=1000,
                            polish=True,
                            seed=self.rng,
                        )
                        opt = res.fun

                        with open(opt_file_name, 'wb') as fh:
                            pickle.dump(opt, fh)
                    break
                else:
                    try:
                        with open(opt_file_name, 'rb') as fh:
                            opt = pickle.load(fh)
                        break
                    except:
                        continue
            except lockfile.LockTimeout:
                pass

        return opt


    def get_empirical_f_worst(self) -> float:
        """Return the empirical f_opt.

        Because ``get_meta_information`` is a static function it has no access to the actual
        function values predicted by the surrogate. This helper function gives access.

        Returns
        -------
        Configuration
        """

        cache_key = self.get_cache_key()
        opt_file_name = os.path.join(self._cache_dir, cache_key + 'worst')

        while True:
            try:
                if not os.path.exists(opt_file_name):
                    with lockfile.LockFile(opt_file_name, timeout=10):

                        cs = self.get_configuration_space()
                        bounds = [(-5, 5)] * num_dimensions

                        def target(x, task):
                            try:
                                config = Configuration(
                                    cs,
                                    {'x%d' % (i + 1): x[i] for i in range(num_dimensions)},
                                )
                                return -float(self.objective_function(config, task=task)['function_value'])
                            except:
                                return -1e10

                        res = scipy.optimize.differential_evolution(
                            func=target, bounds=bounds, args=(self.task,), popsize=1000,
                            polish=True,
                            seed=self.rng,
                        )
                        opt = -res.fun

                        with open(opt_file_name, 'wb') as fh:
                            pickle.dump(opt, fh)
                    break
                else:
                    try:
                        with open(opt_file_name, 'rb') as fh:
                            opt = pickle.load(fh)
                        break
                    except:
                        continue
            except lockfile.LockTimeout:
                pass

        return opt

    def get_meta_data(self, num_base_tasks: Optional[int] = None, fixed_grid: Optional[bool] = False):
        # Sample data for each base task
        if num_base_tasks is None:
            num_base_tasks = self.get_num_base_tasks()

        if fixed_grid:
            seed = self.rng.randint(0, 10000)
        else:
            seed = None

        data_by_task = {}
        for task_ in range(num_base_tasks + 1):
            if self.task == task_:
                continue

            cs = self.get_configuration_space()
            if fixed_grid:
                cs.seed(seed)
            else:
                cs.seed(self.rng.randint(0, 10000))
            configurations = cs.sample_configuration(10)

            # get observed values
            train_y = [
                self.objective_function(config, task=task_)['function_value']
                for config in configurations
            ]
            train_y = np.array(train_y)
            # store training data
            data_by_task[task_] = {
                'configurations': configurations,
                'y': train_y,
            }

        return data_by_task


_adaboost_data = None
_svm_data = None


class WistubaAndSchillingGrid(AbstractBenchmark):
    """Base class for SVM and Adaboost data used by Schilling et al. (ECML 2016) and Wistuba et
    al. (ECML 2016)."""

    _file_dir = None
    _name = None
    _num_hyperparameters = None
    _hp_lower_bounds = None
    _hp_upper_bounds = None

    def __init__(self, task, load_all=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self._cache_dir = os.path.join(hpolib._config.data_dir, 'WistubaAndSchilling')
        try:
            os.makedirs(self._cache_dir)
        except:
            pass
        self.data = self._load_data(load_all)

    def _load_data(self, load_all):
        global _adaboost_data
        global _svm_data
        if self._name == 'Adaboost':
            if _adaboost_data is not None:
                return _adaboost_data
        elif self._name == 'SVM':
            if _svm_data is not None:
                return _svm_data

        data = {}

        current_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(current_dir, self._file_dir)
        files = [
            'A9A', 'W8A', 'abalone', 'appendicitis', 'australian', 'automobile', 'banana',
            'bands', 'breast-cancer', 'bupa', 'car', 'chess', 'cod-rna', 'coil2000',
            'colon-cancer', 'crx', 'diabetes', 'ecoli', 'german-numer', 'haberman',
            'housevotes', 'ijcnn1', 'kr-vs-k', 'led7digit', 'letter', 'lymphography',
            'magic', 'monk-2', 'pendigits', 'phoneme', 'pima', 'ring', 'saheart', 'segment',
            'seismic', 'shuttle', 'sonar-scale', 'spambase', 'spectfheart', 'splice',
            'tic-tac-toe', 'titanic', 'twonorm', 'usps', 'vehicle', 'wdbc', 'wine',
            'winequality-red', 'wisconsin', 'yeast',
        ]

        cache_file = os.path.join(self._cache_dir, self._name + '.pkl')
        while True:

            try:
                with open(cache_file, 'rb') as fh:
                    data = pickle.load(fh)
                break
            except:
                pass

            try:
                with lockfile.LockFile(cache_file, timeout=10):

                    for i, file_name in enumerate(files):
                        if not load_all and i != self.task:
                            continue
                        print(i, file_name)
                        file_name = os.path.join(data_dir, file_name)
                        with open(file_name) as fh:
                            raw_data = fh.readlines()
                        raw_data = [line.split(' ') for line in raw_data]
                        targets = [1 - float(line[0]) for line in raw_data]
                        print(len(raw_data), len(targets))
                        print(targets)
                        configurations = []
                        for line in raw_data:
                            line = line[1:]
                            line = {int(entry.split(':')[0]): float(entry.split(':')[1]) for
                                    entry in line}
                            config = Configuration(
                                values={
                                    'x%d' % (j + 1): line.get(j, 0)
                                    for j in range(self._num_hyperparameters)
                                },
                                configuration_space=self.get_configuration_space(),
                            )
                            configurations.append(config)

                        data[i] = {config: target for config, target in zip(configurations, targets)}

                    with open(cache_file, 'wb') as fh:
                        pickle.dump(data, fh)
                break
            except lockfile.LockTimeout:
                pass

        # Shuffle data after returning it
        for i in data:
            configurations = list(data[i].keys())
            targets = list(data[i].values())
            shuffle_indices = self.rng.permutation(list(range(len(configurations))))
            configurations = [configurations[shuffle_indices[j]] for j in
                              range(len(configurations))]
            targets = [targets[shuffle_indices[j]] for j in range(len(targets))]
            data[i] = {config: target for config, target in zip(configurations, targets)}

        if self._name == 'Adaboost':
            _adaboost_data = data
        elif self._name == 'SVM':
            _svm_data = data

        return data

    @AbstractBenchmark._check_configuration
    def objective_function(self, configuration: Configuration, **kwargs) -> Dict:
        print(configuration.origin)
        return {'function_value': self.data[self.task][configuration]}

    def objective_function_test(self, configuration: Configuration, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_configuration_space(cls):
        cs = ConfigurationSpace()
        for i in range(cls._num_hyperparameters):
            cs.add_hyperparameter(UniformFloatHyperparameter(
                'x%d' % (i + 1),
                cls._hp_lower_bounds[i], cls._hp_upper_bounds[i]))
        return cs

    @classmethod
    def get_meta_information(cls):
        return {
            'num_function_evals': 50,
            'name': '%s grid data' % cls._name,
            'reference': """""",
        }

    def get_num_base_tasks(self) -> int:
        return 49

    def get_empirical_f_opt(self) -> float:
        """Return the empirical f_opt.

        Because ``get_meta_information`` is a static function it has no access to the actual
        function values predicted by the surrogate. This helper function gives access.

        Returns
        -------
        Configuration
        """

        return min(list(self.data[self.task].values()))

    def get_empirical_f_worst(self) -> float:
        """Return the empirical f_opt.

        Because ``get_meta_information`` is a static function it has no access to the actual
        function values predicted by the surrogate. This helper function gives access.

        Returns
        -------
        Configuration
        """

        return max(list(self.data[self.task].values()))

    def get_meta_data(self, num_base_tasks: Optional[int] = None, fixed_grid: Optional[bool] = False):
        # Sample data for each base task
        if num_base_tasks is None:
            num_base_tasks = self.get_num_base_tasks()

        data_by_task = {}

        if fixed_grid:
            indices = self.rng.choice(
                len(self.data[0]),
                replace=False,
                size=self.get_meta_information()['num_function_evals'],
            )

        for task_ in range(num_base_tasks + 1):
            if self.task == task_:
                continue

            if not fixed_grid:
                indices = self.rng.choice(
                    len(self.data[task_]),
                    replace=False,
                    size=self.get_meta_information()['num_function_evals'],
                )
                data = self.data[task_]
            else:
                data = {
                    key: self.data[task_][key]
                    for key in sorted(self.data[task_], key=lambda c: np.sum(c.get_array()))
                }

            data = [(k, v) for i, (k, v) in enumerate(data.items()) if i in indices]
            configurations = [val[0] for val in data]
            train_y = np.array([val[1] for val in data])

            # store training data
            data_by_task[task_] = {
                'configurations': configurations,
                'y': train_y,
            }

        return data_by_task


class AdaboostGrid(WistubaAndSchillingGrid):
    _file_dir = 'adaboost'
    _name = 'Adaboost'
    _num_hyperparameters = 2
    _hp_lower_bounds = [0.07525749891599529, 0.2037950470905062]
    _hp_upper_bounds = [1, 1]


class SVMGrid(WistubaAndSchillingGrid):
    _file_dir = 'svm'
    _name = 'SVM'
    _num_hyperparameters = 6
    _hp_lower_bounds = [0, 0, 0, -0.8333333333333334, -1, 0]
    _hp_upper_bounds = [1, 1, 1, 1.0, 0.75, 1.0]


_nn_data = None
class NNGrid(AbstractBenchmark):
    """LCBench as described in Zimmer et al., 2021"""

    def __init__(self, task, load_all=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self._cache_dir = os.path.join(hpolib._config.data_dir, 'LCBench')
        try:
            os.makedirs(self._cache_dir)
        except:
            pass
        self.data = self._load_data()

    def _load_data(self):
        global _nn_data
        if _nn_data is not None:
            return _nn_data

        data = {}

        allowed_hp_names = self.get_configuration_space().get_hyperparameter_names()

        cache_file = os.path.join(self._cache_dir, 'hpobenchmark.pkl.gz')
        while True:
            try:

                import time
                try:
                    st = time.time()
                    with gzip.open(cache_file, 'rb') as fh:
                        content = fh.read()
                        data = pickle.loads(content)
                    print(time.time() - st)
                    break
                except Exception as e:
                    pass

                with lockfile.LockFile(cache_file, timeout=10):

                    sys.path.append('../../LCBench')
                    from api import Benchmark

                    data_dir = '../../LCBench/data_2k_lw.json'
                    bench = Benchmark(data_dir=data_dir, cache=True,
                                      cache_dir=os.path.dirname(data_dir))
                    ds_names = bench.get_dataset_names()

                    configuration_space = self.get_configuration_space()

                    for i, ds_name in enumerate(ds_names):
                        print(ds_name)
                        configurations = []
                        targets = []

                        n_configs = bench.get_number_of_configs(ds_name)
                        if n_configs is None:
                            raise ValueError(
                                'Could not read the number of configs for dataset %s' % ds_name)

                        for j in range(n_configs):
                            try:
                                config_dict = bench.query(ds_name, 'config', j)
                                config_dict = {
                                    key: value for key, value in config_dict.items()
                                    if key in allowed_hp_names
                                }
                            except ValueError as e:
                                continue
                            try:
                                config = Configuration(
                                    values=config_dict,
                                    configuration_space=configuration_space,
                                )
                            except:
                                print(config_dict)
                                continue
                            configurations.append(config)
                            val_acc = 1 - bench.query(ds_name, "final_val_balanced_accuracy", j)
                            targets.append(val_acc)
                        data[i] = {config: target for config, target in
                                   zip(configurations, targets)}

                    with gzip.open(cache_file, 'wb') as fh:
                        pickle.dump(data, fh)
                break

            except lockfile.LockTimeout:
                pass

        # Shuffle data after returning it
        for i in data:
            configurations = list(data[i].keys())
            targets = list(data[i].values())
            shuffle_indices = self.rng.permutation(list(range(len(configurations))))
            configurations = [configurations[shuffle_indices[j]] for j in
                              range(len(configurations))]
            targets = [targets[shuffle_indices[j]] for j in range(len(targets))]
            data[i] = {config: target for config, target in zip(configurations, targets)}

        _nn_data = data

        return data

    @AbstractBenchmark._check_configuration
    def objective_function(self, configuration: Configuration, **kwargs) -> Dict:
        return {'function_value': self.data[self.task][configuration]}

    def objective_function_test(self, configuration: Configuration, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_configuration_space(cls):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformIntegerHyperparameter('batch_size', 16, 512, log=True))
        cs.add_hyperparameter(UniformFloatHyperparameter('learning_rate', 1e-4, 1e-1, log=True))
        cs.add_hyperparameter(UniformFloatHyperparameter('momentum', 0.1, 0.99))
        cs.add_hyperparameter(UniformFloatHyperparameter('weight_decay', 1e-5, 1e5))
        cs.add_hyperparameter(UniformIntegerHyperparameter('num_layers', 1, 5))
        cs.add_hyperparameter(UniformIntegerHyperparameter('max_units', 16, 1024, log=True))
        cs.add_hyperparameter(UniformFloatHyperparameter('max_dropout', 0.0, 1.0))
        return cs

    @classmethod
    def get_meta_information(cls):
        return {
            'num_function_evals': 50,
            'name': 'Neural Network grid data',
            'reference': """""",
        }

    def get_num_base_tasks(self) -> int:
        n_base_tasks = len(self.data)
        assert n_base_tasks == 35
        return n_base_tasks - 1

    def get_empirical_f_opt(self) -> float:
        """Return the empirical f_opt.
        Because ``get_meta_information`` is a static function it has no access to the actual
        function values predicted by the surrogate. This helper function gives access.
        Returns
        -------
        Configuration
        """

        return min(list(self.data[self.task].values()))

    def get_empirical_f_worst(self) -> float:
        """Return the empirical f_opt.
        Because ``get_meta_information`` is a static function it has no access to the actual
        function values predicted by the surrogate. This helper function gives access.
        Returns
        -------
        Configuration
        """

        return max(list(self.data[self.task].values()))

    def get_meta_data(self, num_base_tasks: Optional[int] = None, fixed_grid: Optional[bool] = False):
        # Sample data for each base task
        if num_base_tasks is None:
            num_base_tasks = self.get_num_base_tasks()

        if fixed_grid:
            indices = self.rng.choice(
                len(self.data[0]),
                replace=False,
                size=self.get_meta_information()['num_function_evals'],
            )

        data_by_task = {}
        for task_ in range(num_base_tasks + 1):
            if self.task == task_:
                continue

            if not fixed_grid:
                indices = self.rng.choice(
                    len(self.data[task_]),
                    replace=False,
                    size=self.get_meta_information()['num_function_evals'],
                )
                data = self.data[task_]
            else:
                data = {
                    key: self.data[task_][key]
                    for key in sorted(self.data[task_], key=lambda c: np.sum(c.get_array()))
                }

            data = [(k, v) for i, (k, v) in enumerate(data.items()) if i in indices]
            configurations = [val[0] for val in data]
            train_y = np.array([val[1] for val in data])

            # store training data
            data_by_task[task_] = {
                'configurations': configurations,
                'y': train_y,
            }

        return data_by_task
