import argparse
import copy
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--benchmark',
    choices=['alpine', 'quadratic', 'adaboost', 'svm', 'openml-svm', 'openml-xgb',
             'openml-glmnet', 'nn'],
    default='alpine',
)
parser.add_argument('--task', type=int)
parser.add_argument(
    '--method',
    choices=['gpmap', 'gcp', 'random', 'rgpe', 'ablr', 'tstr', 'taf', 'wac', 'rmogp',
             'gcp+prior', 'klweighting'],
    default='random',
)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n-init', type=int, default=3)
parser.add_argument('--output-file', type=str, default=None)
parser.add_argument('--iteration-multiplier', type=int, default=1)
parser.add_argument('--empirical-meta-configs', action='store_true')
parser.add_argument('--grid-meta-configs', action='store_true')
parser.add_argument('--learned-initial-design', choices=['None', 'unscaled', 'scaled', 'copula'],
                    default='None')
parser.add_argument('--search-space-pruning', choices=['None', 'complete', 'half'], default='None')
parser.add_argument('--percent-meta-tasks', default=1.0)
parser.add_argument('--percent-meta-data', default=1.0)
args, unknown = parser.parse_known_args()

output_file = args.output_file
if output_file is not None:
    try:
        with open(output_file, 'r') as fh:
            json.load(fh)
        print('Output file %s exists - shutting down.' % output_file)
        exit(1)
    except Exception as e:
        print(e)
        pass

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from rgpe.exploring_openml import SVM, XGBoost, GLMNET
import numpy as np
import pandas as pd
from smac.optimizer.acquisition import EI
from smac.optimizer.ei_optimization import FixedSet
from smac.scenario.scenario import Scenario
from smac.facade.roar_facade import ROAR
from smac.facade.smac_bo_facade import SMAC4BO
from smac.initial_design.latin_hypercube_design import LHDesign

import rgpe
import rgpe.test_functions
from rgpe.methods.noisy_ei import NoisyEI, ClosedFormNei


try:
    os.makedirs(os.path.dirname(output_file))
except:
    pass


kwargs = {}
assert len(unknown) % 2 == 0, (len(unknown), unknown)
for i in range(int(len(unknown) / 2)):
    # Drop the initial "--"
    key = unknown[i * 2][2:].replace('-', '_')
    value = unknown[i * 2 + 1]
    kwargs[key] = value

seed = args.seed
np.random.seed(seed)

benchmark_name = args.benchmark
method_name = args.method
empirical_meta_configs = args.empirical_meta_configs
grid_meta_configs = args.grid_meta_configs
if empirical_meta_configs and grid_meta_configs:
    raise ValueError('Only one allowed at a time!')
learned_initial_design = args.learned_initial_design
# Use the same seed for each method benchmarked!
rng_initial_design = np.random.RandomState(seed)
search_space_pruning = args.search_space_pruning

# Set up the benchmark function
if benchmark_name in ['alpine']:
    task = args.task
    if benchmark_name == 'alpine':
        benchmark = rgpe.test_functions.Alpine1D(rng=seed, task=task)
    else:
        raise ValueError()
    data_by_task = benchmark.get_meta_data(fixed_grid=grid_meta_configs)
    acquisition_function_maximizer = None
    acquisition_function_maximizer_kwargs = None
    initial_design = LHDesign
    initial_design_kwargs = {'init_budget': args.n_init, 'rng': rng_initial_design}
    initial_configurations = None

elif benchmark_name == 'quadratic':
    task = args.task
    if task is None:
        raise TypeError('Task must not be None!')
    benchmark = rgpe.test_functions.Quadratic(rng=seed, task=task)
    data_by_task = benchmark.get_meta_data(fixed_grid=grid_meta_configs)
    acquisition_function_maximizer = None
    acquisition_function_maximizer_kwargs = None
    initial_design = LHDesign
    initial_design_kwargs = {'init_budget': args.n_init, 'rng': rng_initial_design}
    initial_configurations = None

elif benchmark_name in ['openml-svm', 'openml-xgb', 'openml-glmnet']:
    task = args.task
    if task is None:
        raise TypeError('Task must not be None!')
    acquisition_function_maximizer = None
    acquisition_function_maximizer_kwargs = None
    initial_design = LHDesign
    initial_design_kwargs = {'init_budget': args.n_init, 'rng': rng_initial_design}
    initial_configurations = None
    task_to_dataset_mapping = [
        3, 31, 37, 44, 50, 151, 312, 333, 334, 335,
        1036, 1038, 1043, 1046, 1049, 1050, 1063, 1067, 1068,
        1120, 1176,
        1461, 1462, 1464, 1467, 1471, 1479, 1480, 1485, 1486, 1487, 1489, 1494,
        1504, 1510, 1570, 4134, 4534,
    ]
    task_to_dataset_mapping = {i: task_to_dataset_mapping[i] for i in range(len(task_to_dataset_mapping))}
    print('OpenML dataset ID', task_to_dataset_mapping[task])
    if benchmark_name == 'openml-svm':
        benchmark = SVM(dataset_id=task_to_dataset_mapping[task], rng=task_to_dataset_mapping[task])
    elif benchmark_name == 'openml-xgb':
        benchmark = XGBoost(dataset_id=task_to_dataset_mapping[task], rng=task_to_dataset_mapping[task])
    elif benchmark_name == 'openml-glmnet':
        benchmark = GLMNET(dataset_id=task_to_dataset_mapping[task], rng=task_to_dataset_mapping[task])

    # Set up the meta-data to be used
    data_by_task = dict()
    rng = np.random.RandomState(seed)
    for i, dataset_id in enumerate(task_to_dataset_mapping.values()):
        if task == i:
            continue
        if benchmark_name == 'openml-svm':
            meta_benchmark = SVM(dataset_id=dataset_id, rng=dataset_id)
        elif benchmark_name == 'openml-xgb':
            meta_benchmark = XGBoost(dataset_id=dataset_id, rng=dataset_id)
        else:
            meta_benchmark = GLMNET(dataset_id=dataset_id, rng=dataset_id)
        num_data = len(meta_benchmark.configurations)
        if grid_meta_configs:
            cs = meta_benchmark.get_configuration_space()
            cs.seed(seed)
            configurations = cs.sample_configuration(50)
            targets = np.array(
                [
                    meta_benchmark.objective_function(config)['function_value']
                    for config in configurations
                ]
            )
        else:
            choices = rng.choice(num_data, size=50, replace=False)
            configurations = [meta_benchmark.configurations[choice] for choice in choices]
            targets = np.array([meta_benchmark.targets[choice] for choice in choices])
        data_by_task[i] = {'configurations': configurations, 'y': targets}

elif benchmark_name in ['adaboost', 'svm', 'nn']:
    task = args.task
    if task is None:
        raise TypeError('Task must not be None!')
    if benchmark_name == 'adaboost':
        benchmark = rgpe.test_functions.AdaboostGrid(rng=seed, task=task)
        n_params = 2
    elif benchmark_name == 'svm':
        benchmark = rgpe.test_functions.SVMGrid(rng=seed, task=task)
        n_params = 6
    elif benchmark_name == 'nn':
        benchmark = rgpe.test_functions.NNGrid(rng=seed, task=task)
        n_params = 7
    else:
        raise ValueError(benchmark_name)

    data_by_task = benchmark.get_meta_data(fixed_grid=grid_meta_configs)
    acquisition_function_maximizer = FixedSet
    acquisition_function_maximizer_kwargs = {
        'configurations': list(benchmark.data[task].keys())
    }
    initial_design = None
    initial_design_kwargs = None

else:
    raise ValueError(benchmark_name)


# Do custom changes to the number of function evaluations.
# The multiplier is used for random 2x, random 4x, etc.
# However, some benchmarks don't provide enough recorded configurations
# for this, so we cap the number of function evaluations here.
iteration_multiplier = args.iteration_multiplier
num_function_evals = benchmark.get_meta_information()['num_function_evals'] * iteration_multiplier
if benchmark_name == 'adaboost':
    if num_function_evals > 108:
        num_function_evals = 108
        print('Clamping num function evals to %d' % num_function_evals)
if benchmark_name == 'svm':
    if num_function_evals > 288:
        num_function_evals = 288
        print('Clamping num function evals to %d' % num_function_evals)
if benchmark_name == 'nn':
    if num_function_evals > len(benchmark.data[task]):
        num_function_evals = len(benchmark.data[task])
        print('Clamping num function evals to %d' % num_function_evals)


if 'openml' in benchmark_name:
    # TODO: the benchmark queries `config.get_array()` which relies on the ConfigSpace behind
    # config. If we use search space pruning, this will of course be shrinked and we will no
    # longer get the correct mapping. Therefore, we create a new configuration object with the
    # correct configuration space. This function should actually live in the benchmark itself.
    def wrapper(config: Configuration, **kwargs) -> float:
        values = config.get_dictionary()
        new_config = Configuration(benchmark.get_configuration_space(), values=values)
        return benchmark.objective_function(new_config)['function_value']
else:
    def wrapper(config: Configuration, **kwargs) -> float:
        return benchmark.objective_function(config)['function_value']

# Disable SMAC using the pynisher to limit memory and time usage of subprocesses. If you're using
# this code for some real benchmarks, make sure to enable this again!!!
tae_kwargs = {'use_pynisher': False}

# Now load data from previous runs if possible
if empirical_meta_configs is True:
    data_by_task_new = {}
    meta_config_files_dir, _ = os.path.split(output_file)
    for task_id in data_by_task:
        if benchmark_name in ['openml-glmnet', 'openml-svm', 'openml-xgb']:
            data_by_task_new[task_id] = {}
        # TODO change this number if changing the number of seed!
        seed = benchmark.rng.randint(15)
        meta_config_file = os.path.join(meta_config_files_dir, '..', 'gpmap-10',
                                        '%d_50_%d.configs' % (seed, task_id))
        with open(meta_config_file) as fh:
            metadata = json.load(fh)
        configurations = []
        targets = []
        for config, target in metadata:
            configurations.append(Configuration(
                configuration_space=benchmark.get_configuration_space(), values=config)
            )
            targets.append(target)
        targets = np.array(targets)
        data_by_task_new[task_id] = {'configurations': configurations, 'y': targets}
    data_by_task = data_by_task_new
    print('Metadata available for tasks:', {key: len(data_by_task[key]['y']) for key in data_by_task})

# subsample data and/or number of meta-tasks
dropping_rng = np.random.RandomState(seed + 13475)
percent_meta_tasks = args.percent_meta_tasks
if percent_meta_tasks == 'rand':
    percent_meta_tasks = dropping_rng.uniform(0.1, 1.0)
else:
    percent_meta_tasks = float(percent_meta_tasks)
if percent_meta_tasks < 1:
    actual_num_base_tasks = len(data_by_task)
    keep_num_base_tasks = int(np.ceil(actual_num_base_tasks * percent_meta_tasks))
    print('Percent meta tasks', percent_meta_tasks, 'keeping only', keep_num_base_tasks, 'tasks')
    if keep_num_base_tasks < actual_num_base_tasks:
        base_tasks_to_drop = dropping_rng.choice(
            list(data_by_task.keys()),
            replace=False,
            size=actual_num_base_tasks - keep_num_base_tasks,
        )
        for base_task_to_drop in base_tasks_to_drop:
            del data_by_task[base_task_to_drop]
percent_meta_data = args.percent_meta_data
if percent_meta_data == 'rand' or float(percent_meta_data) < 1:
    for task_id in data_by_task:
        if percent_meta_data == 'rand':
            percent_meta_data_ = dropping_rng.uniform(0.1, 1.0)
        else:
            percent_meta_data_ = float(percent_meta_data)
        num_configurations = len(data_by_task[task_id]['configurations'])
        keep_num_configurations = int(np.ceil(num_configurations * percent_meta_data_))
        print('Percent meta data', percent_meta_data, 'keeping only', keep_num_configurations,
              'configurations for task', task_id)
        if keep_num_configurations < num_configurations:
            keep_data_mask = dropping_rng.choice(
                num_configurations, replace=False, size=keep_num_configurations,
            )
            data_by_task[task_id] = {
                'configurations': [
                    config
                    for i, config in enumerate(data_by_task[task_id]['configurations'])
                    if i in keep_data_mask
                ],
                'y': np.array([
                    y
                    for i, y in enumerate(data_by_task[task_id]['y'])
                    if i in keep_data_mask
                ]),
            }

# Conduct search psace pruning
if search_space_pruning is not 'None':

    full_search_space = benchmark.get_configuration_space()
    to_optimize = [True if isinstance(hp, (UniformIntegerHyperparameter,
                                           UniformFloatHyperparameter)) else False
                   for hp in full_search_space.get_hyperparameters()]

    # Section 4 of Perrone et al., 2019
    if search_space_pruning == 'complete':
        minima_by_dimension = [hp.upper if to_optimize[i] else None
                               for i, hp in enumerate(full_search_space.get_hyperparameters())]
        maxima_by_dimension = [hp.lower if to_optimize[i] else None
                               for i, hp in enumerate(full_search_space.get_hyperparameters())]
        print(minima_by_dimension, maxima_by_dimension, to_optimize)
        for task_id, metadata in data_by_task.items():
            argmin = np.argmin(metadata['y'])
            best_config = metadata['configurations'][argmin]
            for i, hp in enumerate(full_search_space.get_hyperparameters()):
                if to_optimize[i]:
                    value = best_config[hp.name]
                    if value is None:
                        continue
                    if value < minima_by_dimension[i]:
                        minima_by_dimension[i] = value
                    if value > maxima_by_dimension[i]:
                        maxima_by_dimension[i] = value

    # Section 5 of Perrone et al., 2019
    elif search_space_pruning == 'half':

        num_hyperparameters = len(full_search_space.get_hyperparameters())
        num_tasks = len(data_by_task)
        bounds = [(0, 1)] * (num_hyperparameters * 2)

        optima = []
        for task_id, metadata in data_by_task.items():
            argmin = np.argmin(metadata['y'])
            best_config = metadata['configurations'][argmin]
            optima.append(best_config.get_array())
            bounds.append((0, 1))
            bounds.append((0, 1))
        import scipy.optimize
        optima = np.array(optima)

        def _optimizee(x, lambda_, return_n_violations=False):
            x = np.round(x, 12)
            l = x[: num_hyperparameters]
            u = x[num_hyperparameters: 2 * num_hyperparameters]
            slack_minus = x[num_hyperparameters * 2: num_hyperparameters * 2 + num_tasks]
            slack_plus = x[num_hyperparameters * 2 + num_tasks:]

            n_violations = 0
            for t in range(num_tasks):
                for i in range(num_hyperparameters):
                    if not to_optimize[i]:
                        continue
                    if np.isfinite(optima[t][i]) and (optima[t][i] < l[i] or optima[t][i] > u[i]):
                        n_violations += 1
                        break
            if return_n_violations:
                return n_violations

            rval = (
                (lambda_ / 2 * np.power(np.linalg.norm(u - l, 2), 2))
                + (1 / (2 * num_tasks) * np.sum(slack_minus) + np.sum(slack_plus))
            )
            return rval

        minima_by_dimension = [0 for i in
                               range(len(full_search_space.get_hyperparameters()))]
        maxima_by_dimension = [1 for i in
                               range(len(full_search_space.get_hyperparameters()))]
        # The paper isn't specific about the values to use for lambda...
        for lambda_ in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
            init = []
            while len(init) < 1:
                cand = [0] * num_hyperparameters
                cand.extend([1] * num_hyperparameters)
                cand.extend([0] * (2 * num_tasks))
                cand = np.array(cand)
                if _optimizee(cand, lambda_) < 10e7:
                    init.append(cand)
            init = np.array(init)

            constraints = []
            class LowerConstraint:
                def __init__(self, i, t):
                    self.i = i
                    self.t = t

                def __call__(self, x):
                    rval = x[self.i] - x[num_hyperparameters * 2 + self.t] - optima[self.t, self.i]
                    return rval if np.isfinite(rval) else 0

            class UpperConstraint:
                def __init__(self, i, t):
                    self.i = i
                    self.t = t

                def __call__(self, x):
                    rval = optima[self.t, self.i] - x[num_hyperparameters * 2 + num_tasks + self.t] - x[num_hyperparameters + self.i]
                    return rval if np.isfinite(rval) else 0

            for t in range(num_tasks):
                for i in range(num_hyperparameters):

                    if not to_optimize[i]:
                        continue

                    constraints.append(scipy.optimize.NonlinearConstraint(
                        LowerConstraint(i, t), -np.inf, 0)
                    )
                    constraints.append(scipy.optimize.NonlinearConstraint(
                        UpperConstraint(i, t), -np.inf, 0)
                    )

            res = scipy.optimize.minimize(
                _optimizee, bounds=bounds, args=(lambda_, ),
                x0=init[0],
                tol=1e-12,
                constraints=constraints,
            )
            print(res)
            n_violations = _optimizee(res.x, lambda_, return_n_violations=True)
            print('Number of violations', n_violations)
            if n_violations > 25:
                continue
            else:
                result = np.round(res.x, 12)
                minima_by_dimension = [result[i] for i in
                                       range(len(full_search_space.get_hyperparameters()))]
                maxima_by_dimension = [result[num_hyperparameters + i] for i in
                                       range(len(full_search_space.get_hyperparameters()))]
                break

    else:
        raise ValueError(search_space_pruning)

    print('Original configuration space')
    print(benchmark.get_configuration_space())
    print('Pruned configuration space')
    configuration_space = ConfigurationSpace()
    for i, hp in enumerate(full_search_space.get_hyperparameters()):
        if to_optimize[i]:
            if search_space_pruning == 'half':
                tmp_config = full_search_space.get_default_configuration()
                vector = tmp_config.get_array()
                vector[i] = minima_by_dimension[i]
                tmp_config = Configuration(full_search_space, vector=vector)
                new_lower = tmp_config[hp.name]
                vector[i] = maxima_by_dimension[i]
                tmp_config = Configuration(full_search_space, vector=vector)
                new_upper = tmp_config[hp.name]
            else:
                new_lower = minima_by_dimension[i]
                new_upper = maxima_by_dimension[i]
            if isinstance(hp, UniformFloatHyperparameter):
                new_hp = UniformFloatHyperparameter(
                    name=hp.name,
                    lower=new_lower,
                    upper=new_upper,
                    log=hp.log,
                )
            elif isinstance(hp, UniformIntegerHyperparameter):
                new_hp = UniformIntegerHyperparameter(
                    name=hp.name,
                    lower=new_lower,
                    upper=new_upper,
                    log=hp.log,
                )
            else:
                raise ValueError(type(hp))
        else:
            new_hp = copy.deepcopy(hp)
        configuration_space.add_hyperparameter(new_hp)

    for condition in full_search_space.get_conditions():
        hp1 = configuration_space.get_hyperparameter(condition.child.name)
        hp2 = configuration_space.get_hyperparameter(condition.parent.name)
        configuration_space.add_condition(type(condition)(hp1, hp2, condition.value))

    print(configuration_space)
    reduced_configuration_space = configuration_space

    if benchmark_name in ['adaboost', 'svm', 'nn']:
        fixed_set_configurations = []
        for config in benchmark.data[task].keys():
            try:
                Configuration(reduced_configuration_space, config.get_dictionary())
                fixed_set_configurations.append(config)
            except:
                continue
        acquisition_function_maximizer_kwargs['configurations'] = fixed_set_configurations
        print('Using only %d configurations' %
              len(acquisition_function_maximizer_kwargs['configurations']))
        configuration_space = benchmark.get_configuration_space()

else:
    configuration_space = benchmark.get_configuration_space()
    reduced_configuration_space = configuration_space


scenario = Scenario({
    'run_obj': 'quality',
    'runcount_limit': num_function_evals,
    'cs': reduced_configuration_space,
    'deterministic': True,
    'output_dir': None,
})

# Mapping the latin hypercube initial design to the available points of the grid via a nearest
# neighbor method.
if benchmark_name in ['adaboost', 'svm', 'nn']:
    import copy
    import pyDOE
    from sklearn.neighbors import NearestNeighbors

    lhd = pyDOE.lhs(n=n_params, samples=args.n_init)
    initial_configurations = []
    vectors = []
    conf_list = []
    for config in benchmark.data[task].keys():
        try:
            new_config = Configuration(reduced_configuration_space, config.get_dictionary())
            conf_list.append(config)
            vectors.append(new_config.get_array())
        except:
            continue
    vectors = np.array(vectors)

    taken_indices = []
    for design in lhd:
        nbrs = NearestNeighbors(n_neighbors=len(vectors)).fit(vectors)
        _, indices = nbrs.kneighbors([design])
        for ind in indices.flatten():
            if ind not in taken_indices:
                taken_indices.append(ind)
                initial_configurations.append(conf_list[ind])
                break
    assert len(initial_configurations) == args.n_init
    print(initial_configurations)

# Now learn an initial design
if learned_initial_design in ['scaled', 'unscaled', 'copula']:
    from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
    from smac.runhistory.runhistory import RunHistory
    from smac.tae.execute_ta_run import StatusType
    from rgpe.utils import get_gaussian_process
    from smac.epm.util_funcs import get_types

    if learned_initial_design != 'copula':
        print('Learned init with scaled/unscaled')
        rh2epm = RunHistory2EPM4Cost(
            scenario=scenario,
            num_params=len(scenario.cs.get_hyperparameter_names()),
            success_states=[StatusType.SUCCESS],
        )
    else:
        print('Learned init with copula transform')
        from rgpe.utils import copula_transform
        class CopulaRH2EPM(RunHistory2EPM4Cost):

            def transform_response_values(self, values: np.ndarray) -> np.ndarray:
                return copula_transform(values)
        rh2epm = CopulaRH2EPM(
            scenario=scenario,
            num_params=len(scenario.cs.get_hyperparameter_names()),
            success_states=[StatusType.SUCCESS],
        )

    new_initial_design = []

    minima = {}
    maxima = {}
    candidate_configurations = list()
    candidate_set = set()
    benchmark_class = benchmark.__class__
    meta_models = {}
    meta_models_rng = np.random.RandomState(seed)
    for task_id, metadata in data_by_task.items():
        if benchmark_name in ['openml-svm', 'openml-xgb', 'openml-glmnet']:
            dataset_id = task_to_dataset_mapping[task_id]
            meta_benchmark = benchmark_class(rng=seed, dataset_id=dataset_id)
        else:
            meta_benchmark = benchmark_class(task=task_id, rng=seed, load_all=False)
        if learned_initial_design == 'scaled':
            minima[task_id] = meta_benchmark.get_empirical_f_opt()
            if hasattr(meta_benchmark, 'get_empirical_f_worst'):
                maxima[task_id] = meta_benchmark.get_empirical_f_worst()
            elif hasattr(meta_benchmark, 'get_empirical_f_max'):
                maxima[task_id] = meta_benchmark.get_empirical_f_max()
            else:
                raise NotImplementedError()
        rh = RunHistory()
        for config, target in zip(metadata['configurations'], metadata['y']):
            rh.add(config=config, cost=target, time=0, status=StatusType.SUCCESS)

        types, bounds = get_types(benchmark.get_configuration_space(), None)
        gp = get_gaussian_process(meta_benchmark.get_configuration_space(), rng=meta_models_rng,
                                  bounds=bounds, types=types, kernel=None)
        X, y = rh2epm.transform(rh)
        gp.train(X, y)
        meta_models[task_id] = gp
        for config in metadata['configurations']:
            if config not in candidate_set:
                if benchmark_name in ['adaboost', 'svm', 'nn']:
                    if config not in acquisition_function_maximizer_kwargs['configurations']:
                        continue
                else:
                    try:
                        Configuration(reduced_configuration_space, config.get_dictionary())
                    except Exception as e:
                        continue
                candidate_configurations.append(config)
                candidate_set.add(config)

    print('Using %d candidates for the initial design' % len(candidate_configurations))
    predicted_losses_cache = dict()
    def target_function(config, previous_losses):
        losses = []
        for i, (task_id, meta_benchmark) in enumerate(data_by_task.items()):
            meta_model = meta_models[task_id]

            key = (config, task_id)
            if key in predicted_losses_cache:
                loss_cfg = predicted_losses_cache[key]
            else:
                loss_cfg, _ = meta_model.predict(config.get_array().reshape((1, -1)))
                if learned_initial_design == 'scaled':
                    minimum = minima[task_id]
                    diff = maxima[task_id] - minimum
                    diff = diff if diff > 0 else 1
                    loss_cfg = (loss_cfg - minimum) / diff
                predicted_losses_cache[key] = loss_cfg
            if loss_cfg < previous_losses[i]:
                tmp_loss = loss_cfg
            else:
                tmp_loss = previous_losses[i]
            losses.append(tmp_loss)

        return np.mean(losses), losses

    current_loss_cache = [np.inf] * len(data_by_task)
    for i in range(args.n_init):
        losses = []
        loss_cache = []
        for j, candidate_config in enumerate(candidate_configurations):
            loss, loss_cache_tmp = target_function(candidate_config, current_loss_cache)
            losses.append(loss)
            loss_cache.append(loss_cache_tmp)
        min_loss = np.min(losses)
        min_losses_indices = np.where(losses == min_loss)[0]
        argmin = meta_models_rng.choice(min_losses_indices)
        print(argmin, losses[argmin], len(losses), losses)
        new_initial_design.append(candidate_configurations[argmin])
        current_loss_cache = loss_cache[argmin]
        del candidate_configurations[argmin]

    initial_configurations = copy.deepcopy(new_initial_design)
    initial_design = None
    initial_design_kwargs = None

    del meta_models

    print('Learned initial design')
    print(initial_configurations)


# Set up the optimizer
if method_name == 'random':
    method = ROAR(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
    )
elif method_name == 'gpmap':
    method = SMAC4BO(
        model_type='gp',
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
    )

elif method_name == 'gcp':

    from rgpe.utils import copula_transform
    from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost

    class CopulaRH2EPM(RunHistory2EPM4Cost):

        def transform_response_values(self, values: np.ndarray) -> np.ndarray:
            return copula_transform(values)

    method = SMAC4BO(
        model_type='gp',
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
        runhistory2epm=CopulaRH2EPM,
    )

elif method_name == 'ablr':
    from rgpe.methods.ablr import ABLR
    normalization = kwargs['normalization']
    if normalization == 'mean/var':
        use_copula_transform = False
    elif normalization == 'Copula':
        use_copula_transform = True
    else:
        raise ValueError(normalization)
    print('ABLR', use_copula_transform)
    method = SMAC4BO(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        model=ABLR,
        model_kwargs={'training_data': data_by_task, 'use_copula_transform': use_copula_transform},
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
    )

elif method_name == 'gcp+prior':
    from rgpe.methods.GCPplusPrior import GCPplusPrior, CustomEI
    method = SMAC4BO(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        model=GCPplusPrior,
        model_kwargs={'training_data': data_by_task},
        acquisition_function=CustomEI,
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
    )

elif method_name == 'taf':
    from rgpe.methods.taf import TAF
    weighting_mode = kwargs['weighting_mode']
    normalization = kwargs['normalization']
    if weighting_mode == 'rgpe':
        from rgpe.methods.rgpe import RGPE
        model = RGPE
        weight_dilution_strategy = kwargs['weight_dilution_strategy']
        sampling_mode = kwargs['sampling_mode']
        model_kwargs = {
            'weight_dilution_strategy': weight_dilution_strategy,
            'number_of_function_evaluations': num_function_evals,
            'training_data': data_by_task,
            'num_posterior_samples': 1000,
            'sampling_mode': sampling_mode,
            'normalization': normalization,
        }
    elif weighting_mode == 'tstr':
        from rgpe.methods.tstr import TSTR
        model = TSTR
        weight_dilution_strategy = kwargs['weight_dilution_strategy']
        bandwidth = float(kwargs['bandwidth'])
        normalization = kwargs['normalization']
        model_kwargs = {
            'weight_dilution_strategy': weight_dilution_strategy,
            'number_of_function_evaluations': num_function_evals,
            'training_data': data_by_task,
            'bandwidth': bandwidth,
            'normalization': normalization,
        }
    else:
        raise ValueError(weighting_mode)
    print(model, normalization, weight_dilution_strategy)
    method = SMAC4BO(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        acquisition_function=TAF,
        model=model,
        model_kwargs=model_kwargs,
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
    )

elif method_name == 'tstr':
    from rgpe.methods.tstr import TSTR
    from rgpe.utils import EI as EI4TSTR

    if kwargs['acquisition_function_name'] == 'NoisyEI':
        acquisition_function = NoisyEI
    elif kwargs['acquisition_function_name'] == 'fullmodelEI':
        acquisition_function = EI
    elif kwargs['acquisition_function_name'] == 'EI':
        acquisition_function = EI4TSTR
    else:
        raise ValueError(kwargs['acquisition_function_name'])

    bandwidth = float(kwargs['bandwidth'])
    variance_mode = kwargs['variance_mode']
    normalization = kwargs['normalization']
    weight_dilution_strategy = kwargs['weight_dilution_strategy']

    method = SMAC4BO(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        model=TSTR,
        model_kwargs={
            'training_data': data_by_task,
            'bandwidth': bandwidth,
            'variance_mode': variance_mode,
            'normalization': normalization,
            'weight_dilution_strategy': weight_dilution_strategy,
            'number_of_function_evaluations': num_function_evals,
        },
        acquisition_function=acquisition_function,
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
    )

elif method_name == 'wac':
    from rgpe.methods.warmstarting_ac import WarmstartingAC

    variance_mode = 'average'
    acquisition_function = EI

    method = SMAC4BO(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        model=WarmstartingAC,
        model_kwargs={'training_data': data_by_task, 'variance_mode': variance_mode},
        acquisition_function=acquisition_function,
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
    )

elif method_name == 'klweighting':
    from rgpe.methods.kl_weighting import KLWeighting

    eta = float(kwargs['eta'])
    print('KLWeighting', eta)

    method = SMAC4BO(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        model=KLWeighting,
        model_kwargs={'training_data': data_by_task, 'eta': eta},
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
    )

elif method_name == 'rgpe':
    from rgpe.methods.rgpe import RGPE
    if kwargs['acquisition_function_name'] == 'EI':
        from rgpe.utils import EI as EI4RGPE
        acquisition_function = EI4RGPE
        acquisition_function_kwargs = {}
    elif kwargs['acquisition_function_name'] == 'fullmodelEI':
        acquisition_function = EI
        acquisition_function_kwargs = {}
    elif kwargs['acquisition_function_name'] == 'CFNEI':
        acquisition_function = ClosedFormNei
        acquisition_function_kwargs = {}
    else:
        # This will fail if it is not an integer below
        acquisition_function = NoisyEI
        target_model_incumbent = kwargs['target_model_incumbent']
        if target_model_incumbent == 'True':
            target_model_incumbent = True
        elif target_model_incumbent == 'False':
            target_model_incumbent = False
        else:
            raise ValueError(target_model_incumbent)
        acquisition_function_kwargs = {
            'n_samples': int(kwargs['acquisition_function_name']),
            'target_model_incumbent': target_model_incumbent,
        }

    num_posterior_samples = int(kwargs['num_posterior_samples'])
    sampling_mode = kwargs['sampling_mode']
    variance_mode = kwargs['variance_mode']
    normalization = kwargs['normalization']

    weight_dilution_strategy = kwargs['weight_dilution_strategy']
    print(num_posterior_samples, acquisition_function, sampling_mode, variance_mode,
          weight_dilution_strategy, normalization)
    method = SMAC4BO(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        model=RGPE,
        model_kwargs={
            'training_data': data_by_task,
            'weight_dilution_strategy': weight_dilution_strategy,
            'number_of_function_evaluations': num_function_evals,
            'variance_mode': variance_mode,
            'num_posterior_samples': num_posterior_samples,
            'sampling_mode': sampling_mode,
            'normalization': normalization,
        },
        acquisition_function_kwargs=acquisition_function_kwargs,
        acquisition_function=acquisition_function,
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
    )

elif method_name == 'rmogp':
    from rgpe.methods.rmogp import MixtureOfGPs, NoisyMixtureOfGPs
    from rgpe.methods.rgpe import RGPE

    model = RGPE
    weight_dilution_strategy = kwargs['weight_dilution_strategy']
    sampling_mode = kwargs['sampling_mode']
    num_posterior_samples = int(kwargs['num_posterior_samples'])
    use_expectation = bool(kwargs['use_expectation'] == 'True')
    use_global_incumbent = bool(kwargs['use_global_incumbent'] == 'True')
    normalization = kwargs['normalization']
    alpha = float(kwargs['alpha'])

    acq_func = MixtureOfGPs
    acq_func_kwargs = {
        'use_expectation': use_expectation,
        'use_global_incumbent': use_global_incumbent,
    }

    print('MOGP', num_posterior_samples, sampling_mode, acq_func, use_expectation,
          use_global_incumbent, weight_dilution_strategy, alpha, normalization)
    model_kwargs = {
        'weight_dilution_strategy': weight_dilution_strategy,
        'number_of_function_evaluations': num_function_evals,
        'training_data': data_by_task,
        'variance_mode': 'target',
        'num_posterior_samples': num_posterior_samples,
        'sampling_mode': sampling_mode,
        'normalization': normalization,
        'alpha': alpha,
    }

    method = SMAC4BO(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        acquisition_function=acq_func,
        acquisition_function_kwargs=acq_func_kwargs,
        model=model,
        model_kwargs=model_kwargs,
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
    )
else:
    raise ValueError(method_name)

# Reduce the local search for XGB to finish BO in a reasonable amount of time
if benchmark_name in ['openml-xgb']:
    if method_name != 'random':
        method.solver.epm_chooser.acq_optimizer.local_search.n_steps_plateau_walk = 1
        method.solver.epm_chooser.acq_optimizer.n_sls_iterations = 5

# Disable random configurations form SMAC
method.solver.epm_chooser.random_configuration_chooser = None

# And now run the optimizer
method.optimize()

if hasattr(method.solver.epm_chooser.model, 'weights_over_time'):
    weight_file = output_file
    weight_file = weight_file.replace('.json', '.weights')
    weights_over_time = method.solver.epm_chooser.model.weights_over_time
    weights_over_time = [
        [float(weight) for weight in weights]
        for weights in weights_over_time
    ]
    with open(weight_file, 'w') as fh:
        json.dump(weights_over_time, fh, indent=4)
if hasattr(method.solver.epm_chooser.model, 'p_drop_over_time'):
    p_drop_file = output_file
    p_drop_file = p_drop_file.replace('.json', '.pdrop')
    p_drop_over_time = method.solver.epm_chooser.model.p_drop_over_time
    p_drop_over_time = [
        [float(p_drop) for p_drop in drops]
        for drops in p_drop_over_time
    ]
    with open(p_drop_file, 'w') as fh:
        json.dump(p_drop_over_time, fh, indent=4)

# Dump the evaluated configurations as meta-data for later runs
print(method_name, method_name == 'gpmap')
if method_name == 'gpmap':
    rh = method.get_runhistory()
    evaluated_configurations = []
    for config in rh.config_ids:
        cost = rh.get_cost(config)
        print(cost)
        evaluated_configurations.append([config.get_dictionary(), cost])
    print(evaluated_configurations)
    evaluated_configs_file = output_file
    evaluated_configs_file = evaluated_configs_file.replace('.json', '.configs')
    print(evaluated_configs_file)
    with open(evaluated_configs_file, 'w') as fh:
        json.dump(evaluated_configurations, fh)

# Now output the optimization trajectory (performance)
traj = method.get_trajectory()

trajectory = pd.Series([np.NaN] * (num_function_evals + 1))
for entry in traj:
    trajectory[entry.ta_runs] = entry.train_perf
trajectory.fillna(inplace=True, method='ffill')

trajectory = trajectory.to_numpy()
best = benchmark.get_empirical_f_opt()

regret = trajectory[-1] - best
regret_trajectory = trajectory - best

if (hasattr(benchmark, 'get_empirical_f_worst') or hasattr(benchmark, 'get_empirical_f_max')) and\
        benchmark_name not in ['alpine', 'quadratic']:
    if hasattr(benchmark, 'get_empirical_f_worst'):
        worst = benchmark.get_empirical_f_worst()
    else:
        worst = benchmark.get_empirical_f_max()
    normalizer = worst - best
    print('Normalization constants', normalizer, best, worst)
    print('Unnormalized trajectory', trajectory)
    print('Unnormalized regret trajectory', regret_trajectory)
    regret = regret / normalizer
    regret_trajectory = regret_trajectory / normalizer
print('Trajectory', regret_trajectory)
print('Final regret', regret)

if output_file is not None:
    results_dict = {
        'regret': float(regret),
        'regret_trajectory': [float(v) for v in regret_trajectory],
        'trajectory': [float(v) for v in trajectory],
    }
    with open(output_file, 'w') as fh:
        json.dump(results_dict, fh, indent=4)


