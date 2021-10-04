"""Generate commands to reproduce experiments."""

import argparse
import glob
import itertools
import os
import random
from typing import List

parser = argparse.ArgumentParser()

benchmarks = {
    'alpine': (1, 50),
    'quadratic': (30, 15),
    'adaboost': (50, 15),
    'svm': (50, 15),
    'openml-glmnet': (38, 15),
    'openml-svm': (38, 15),
    'openml-xgb': (38, 15),
    'nn': (35, 15),
}

normalization_to_initial_design = {
    'None': 'unscaled',
    'mean/var': 'scaled',
    'Copula': 'copula',
}

all_setups = {
    "": "",
    "-learnedinit": "--learned-initial-design {learned_initial_design}",
    "-gpmetadata": "--empirical-meta-configs",
    "-gpmetadata-learnedinit": "--empirical-meta-configs --learned-initial-design {learned_initial_design}",
    "-gridmetadata": "--grid-meta-configs",
    "-gridmetadata-learnedinit": "--grid-meta-configs --learned-initial-design {learned_initial_design}"
}
all_setups_args = [setup[1:] if len(setup) > 0 else setup for setup in all_setups] + ['None']

parser.add_argument(
    '--benchmark',
    choices=benchmarks.keys(),
    required=True,
    help="Which benchmark to create the commands file for."
)
parser.add_argument(
    '--setup',
    choices=all_setups_args,
    nargs='*',
    help="For which setup of meta-data (grid, from the pre-evaluated grid; gp, from a previous "
         "run of the GP) and learned init or not to create the commands."
)
parser.add_argument(
    '--results-directory',
    type=str,
    help="If given, this script will check which output files already exist and not add those "
         "call to the commands file."
)

args = parser.parse_args()

results_dir = args.results_directory

if results_dir:
    glob_dir = '%s/*/*/*' % glob.escape(results_dir)
    available_files = glob.glob(glob_dir)
    to_drop = len(results_dir)
    available_files = set([available_file[to_drop:] for available_file in available_files])
else:
    available_file = []

def add_seeds_and_tasks(
    template: str,
    n_seeds: int,
    n_tasks: int,
    relative_output_file_template: str,
) -> List[str]:
    rval = []
    for seed in range(n_seeds):
        for task_id in range(n_tasks):

            if results_dir:
                relative_output_file = relative_output_file_template.format(seed=seed, task_id=task_id)
                # output_file = os.path.join(results_dir, relative_output_file)
                if relative_output_file in available_files:
                    continue

            rval.append(template.format(seed=seed, task_id=task_id))
    return rval

output_directory = "/home/feurerm/projects/2018_fb/results_smac"
#output_directory = "/work/ws/nemo/fr_mf1066-2019_rgpe-0/"
run_script = "python /home/feurerm/sync_dir/projects/2018_fb/rgpe_code/scripts/run_benchmark_smac.py"
#run_script = "python /home/fr/fr_fr/fr_mf1066/repositories/2019_rgpe/rgpe/scripts/run_benchmark_smac.py"

for benchmark, (n_tasks, n_seeds) in benchmarks.items():
    if benchmark != args.benchmark:
        continue
    setups_args = args.setup
    for i in range(len(setups_args)):
        if setups_args[i] != 'None':
            setups_args[i] = '-' + setups_args[i]
    setups = {}
    for setup in all_setups:
        if setup in setups_args:
            setups[setup] = all_setups[setup]
    if 'None' in setups_args:
        setups[''] = all_setups['']
    print(setups)

    commands = []

    # Random search
    n_init = 1
    for multiplier in (1, 50):
        for setup_name, setup_string in setups.items():
            for search_space_pruning in (False, True):
                if 'learned' in setup_name:
                    continue
                elif not search_space_pruning and setup_name != '':
                    continue

                filename = "{seed}_50_{task_id}.json"
                if search_space_pruning:
                    method_name = "random%d-ssp%s-%d" % (multiplier, setup_name, n_init)
                else:
                    method_name = "random%s%s-%d" % (multiplier, setup_name, n_init)
                relative_output_file_template = os.path.join(benchmark, method_name, filename)
                output_file = os.path.join(
                    output_directory, benchmark, method_name, filename)

                command = (
                    "{run_script} --benchmark {benchmark} --method random --seed {seed} "
                    "--task {task_id} --iteration-multiplier {multiplier} --n-init {n_init} "
                    "--output-file {output_file} {setup_string}"
                )
                if search_space_pruning:
                    command += " --search-space-pruning complete"

                template = command.format(**{
                    'run_script': run_script,
                    'benchmark': benchmark,
                    'multiplier': multiplier,
                    'output_file': output_file,
                    'seed': '{seed}',
                    'n_init': n_init,
                    'task_id': '{task_id}',
                    'setup_string': setup_string,
                })
                commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))

    # Baseline
    for n_init in (10, 50):
        for method in ('gpmap', 'gcp'):
            for search_space_pruning in (False, True):
                for setup_name, setup_string in setups.items():

                    filename = "{seed}_50_{task_id}.json"

                    if search_space_pruning:
                        method_name = "%s-ssp%s-%d" % (method, setup_name, n_init)
                    else:
                        method_name = "%s%s-%d" % (method, setup_name, n_init)
                    output_file = os.path.join(
                        output_directory, benchmark, method_name, filename)
                    relative_output_file_template = os.path.join(benchmark, method_name, filename)
                    command = (
                        "{run_script} --benchmark {benchmark} --method {method} --seed {seed} "
                        "--task {task_id} {setup_string} --n-init {n_init} --output-file {output_file}"
                    )
                    if search_space_pruning:
                        command += " --search-space-pruning complete"
                    template = command.format(**{
                        'method': method,
                        'run_script': run_script,
                        'benchmark': benchmark,
                        'output_file': output_file,
                        'seed': '{seed}',
                        'task_id': '{task_id}',
                        'setup_string': setup_string.format(
                            learned_initial_design='scaled' if method == 'gpmap' else 'copula',
                        ),
                        'n_init': n_init,
                    })
                    if benchmark == 'alpine' and method == 'gpmap' and setup_name == '':
                        commands.extend(add_seeds_and_tasks(template, n_seeds, 6, relative_output_file_template))
                        pass
                    else:
                        commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))
                        pass

    # ABLR
    method = 'ablr'
    for n_init in (1, ):
        for normalization in ('mean/var', 'Copula'):
            for setup_name, setup_string in setups.items():
                if 'learnedinit' not in setup_name:
                    continue
                if normalization in ['Copula', 'mean/var'] and n_init < 2:
                    n_init_ = 2
                else:
                    n_init_ = 1
                filename = "{seed}_50_{task_id}.json"
                output_file = os.path.join(
                    output_directory, benchmark,
                    "%s-%s%s-%d" % (method, normalization.replace('/', ''),
                                    setup_name, n_init_), filename)
                relative_output_file_template = os.path.join(benchmark, method_name, filename)
                command = (
                    "{run_script} --benchmark {benchmark} --method {method} --seed {seed} "
                    "--normalization {normalization} "
                    "--task {task_id} {setup_string} --n-init {n_init} --output-file {output_file}"
                )
                template = command.format(**{
                    'method': method,
                    'run_script': run_script,
                    'benchmark': benchmark,
                    'output_file': output_file,
                    'seed': '{seed}',
                    'task_id': '{task_id}',
                    'n_init': n_init_,
                    'setup_string': setup_string.format(
                        learned_initial_design=normalization_to_initial_design[normalization]
                    ),
                    'normalization': normalization,
                })
                commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))

    # TSTR
    method = 'tstr'
    for n_init in (1, ):
        for weight_dilution_strategy in ('None', 'probabilistic-ld'):
            for bandwidth in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                for setup_name, setup_string in setups.items():
                    for normalization in ('mean/var', 'Copula', 'None'):
                        for acquisition_function in ('targetEI', 'fullmodeltargetEI',):
                            if 'learnedinit' not in setup_name:
                                continue
                            if normalization in ['Copula', 'mean/var'] and n_init < 2:
                                n_init_ = 2
                            else:
                                n_init_ = n_init
                            filename = "{seed}_50_{task_id}.json"
                            method_name = "%s-%s-%s-%s-%f%s-%d" % (
                                method, acquisition_function, normalization.replace('/', ''),
                                weight_dilution_strategy,
                                bandwidth, setup_name, n_init_
                            )
                            output_file = os.path.join(output_directory, benchmark, method_name, filename)
                            relative_output_file_template = os.path.join(benchmark, method_name, filename)
                            command = (
                                "{run_script} --benchmark {benchmark} --method {method} --seed {seed} "
                                "--task {task_id} {setup_string} --n-init {n_init} --bandwidth {bandwidth} "
                                "--normalization {normalization} --output-file {output_file} "
                                "--weight_dilution_strategy {weight_dilution_strategy} "
                            )
                            if acquisition_function == 'targetEI':
                                command = '%s --variance-mode target --acquisition-function-name EI' % command
                            elif acquisition_function == 'fullmodeltargetEI':
                                command = '%s --variance-mode target --acquisition-function-name fullmodelEI' % command
                            else:
                                raise ValueError(acquisition_function)

                            template = command.format(**{
                                'method': method,
                                'run_script': run_script,
                                'benchmark': benchmark,
                                'output_file': output_file,
                                'seed': '{seed}',
                                'task_id': '{task_id}',
                                'n_init': n_init_,
                                'setup_string': setup_string.format(
                                    learned_initial_design=normalization_to_initial_design[normalization]
                                ),
                                'bandwidth': bandwidth,
                                'normalization': normalization,
                                'weight_dilution_strategy': weight_dilution_strategy,
                            })
                            commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))

    # KL-dirvergence-based distance measure
    method = 'klweighting'
    if benchmark not in ['openml-svm', 'openml-xgb']:
        for n_init in (2, ):
            for eta in (1, 2, 5, 10, 20, 50, 100):
                for setup_name, setup_string in setups.items():
                    if 'learnedinit' not in setup_name:
                        continue
                    filename = "{seed}_50_{task_id}.json"
                    method_name = "%s-%f%s-%d" % (method, eta, setup_name, n_init)
                    output_file = os.path.join(output_directory, benchmark, method_name, filename)
                    relative_output_file_template = os.path.join(benchmark, method_name, filename)
                    command = (
                        "{run_script} --benchmark {benchmark} --method {method} --seed {seed} "
                        "--task {task_id} {setup_string} --n-init {n_init} --eta {eta} "
                        "--output-file {output_file} "
                    )
                    template = command.format(**{
                        'method': method,
                        'run_script': run_script,
                        'benchmark': benchmark,
                        'output_file': output_file,
                        'seed': '{seed}',
                        'task_id': '{task_id}',
                        'n_init': n_init,
                        'setup_string': setup_string.format(learned_initial_design='scaled'),
                        'eta': eta,
                    })
                    commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))

    # WAC
    for n_init in (2, ):
        for setup_name, setup_string in setups.items():
            if 'learnedinit' not in setup_name:
                continue
            filename = "{seed}_50_{task_id}.json"
            method_name = "wac%s-%d" % (setup_name, n_init)
            output_file = os.path.join(output_directory, benchmark, method_name, filename)
            relative_output_file_template = os.path.join(benchmark, method_name, filename)
            command = (
                "{run_script} --benchmark {benchmark} --method {method} --seed {seed} "
                "--task {task_id} {setup_string} --n-init {n_init} --output-file {output_file} "
                "--acquisition-function-name EI"
            )
            template = command.format(**{
                'method': 'wac',
                'run_script': run_script,
                'benchmark': benchmark,
                'output_file': output_file,
                'seed': '{seed}',
                'task_id': '{task_id}',
                'n_init': n_init,
                'setup_string': setup_string.format(learned_initial_design='scaled'),
            })
            commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))

    # GCP+Prior
    for n_init in (2, ):
        for setup_name, setup_string in setups.items():
            if 'learnedinit' not in setup_name:
                continue
            filename = "{seed}_50_{task_id}.json"
            method_name = "gcp+prior%s-%d" % (setup_name, n_init)
            output_file = os.path.join(output_directory, benchmark, method_name, filename)
            relative_output_file_template = os.path.join(benchmark, method_name, filename)
            command = (
                "{run_script} --benchmark {benchmark} --method {method} --seed {seed} "
                "--task {task_id} {setup_string} --n-init {n_init} --output-file {output_file} "
                "--acquisition-function-name EI"
            )
            template = command.format(**{
                'method': 'gcp+prior',
                'run_script': run_script,
                'benchmark': benchmark,
                'output_file': output_file,
                'seed': '{seed}',
                'task_id': '{task_id}',
                'n_init': n_init,
                'setup_string': setup_string.format(learned_initial_design='copula'),
            })
            commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))

    # TAF-TSTR
    method = 'taf'
    for n_init in (1, ):
        for weight_dilution_strategy in ('None', 'probabilistic-ld'):
            for bandwidth in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                for setup_name, setup_string in setups.items():
                    for normalization in ['None', 'Copula', 'mean/var']:
                        if 'learnedinit' not in setup_name:
                            continue
                        if normalization in ['Copula', 'mean/var'] and n_init < 2:
                            n_init_ = 2
                        else:
                            n_init_ = n_init
                        filename = "{seed}_50_{task_id}.json"
                        method_name = "%s-tstr-%s-%s-%f%s-%d" % (
                            method, normalization.replace('/', ''), weight_dilution_strategy,
                            bandwidth, setup_name, n_init_,
                        )
                        output_file = os.path.join(output_directory, benchmark, method_name, filename)
                        relative_output_file_template = os.path.join(benchmark, method_name, filename)
                        command = (
                            "{run_script} --benchmark {benchmark} --method {method} --seed {seed} "
                            "--task {task_id} {setup_string} --bandwidth {bandwidth} "
                            "--weighting-mode tstr --n-init {n_init} --normalization {normalization} "
                            "--weight_dilution_strategy {weight_dilution_strategy} "
                            "--output-file {output_file} "
                        )
                        template = command.format(**{
                            'method': method,
                            'run_script': run_script,
                            'benchmark': benchmark,
                            'output_file': output_file,
                            'seed': '{seed}',
                            'task_id': '{task_id}',
                            'n_init': n_init_,
                            'setup_string': setup_string.format(
                                learned_initial_design=normalization_to_initial_design[normalization]),
                            'bandwidth': bandwidth,
                            'normalization': normalization,
                            'weight_dilution_strategy': weight_dilution_strategy,
                        })
                        commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))

    # TAF-RGPE
    method = 'taf'
    for n_init in (1, ):
        for weight_dilution_strategy in ('None', 'probabilistic-ld'):
            for sampling_strategy in ('bootstrap', ):
                for normalization in ('None', 'Copula', 'mean/var'):
                    for setup_name, setup_string in setups.items():
                        if 'learnedinit' not in setup_name and n_init == 1:
                            continue
                        if normalization in ['Copula', 'mean/var'] and n_init < 2:
                            n_init_ = 2
                        else:
                            n_init_ = n_init
                        filename = "{seed}_50_{task_id}.json"
                        method_name = "%s-rgpe-%s-%s-%s-1000-%s-%d" % (
                            method, sampling_strategy, normalization.replace('/', ''),
                            weight_dilution_strategy, setup_name, n_init_,
                        )
                        output_file = os.path.join(output_directory, benchmark, method_name, filename)
                        relative_output_file_template = os.path.join(benchmark, method_name, filename)
                        command = (
                            "{run_script} --benchmark {benchmark} --method {method} --seed {seed} "
                            "--task {task_id} {setup_string} --weight-dilution-strategy {weight_dilution_strategy} "
                            "--sampling-mode {sampling_strategy} "
                            "--weighting-mode rgpe --n-init {n_init} --output-file {output_file} "
                            "--normalization {normalization}"
                        )

                        template = command.format(**{
                            'method': method,
                            'run_script': run_script,
                            'benchmark': benchmark,
                            'output_file': output_file,
                            'seed': '{seed}',
                            'task_id': '{task_id}',
                            'n_init': n_init_,
                            'setup_string': setup_string.format(
                                learned_initial_design=normalization_to_initial_design[normalization]),
                            'weight_dilution_strategy': weight_dilution_strategy,
                            'sampling_strategy': sampling_strategy,
                            'normalization': normalization,
                        })
                        commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))

    # RGPE with Mixture of Gaussian Processes
    for method in ('rmogp', ):
        for n_init in (1, ):
            for (
                weight_dilution_strategy, sampling_strategy, use_expectation,
                use_global_incumbent, num_posterior_samples, alpha, normalization,
            ) in (
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 0.0, 'None'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 0.0, 'mean/var'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 0.0, 'Copula'),
                ('None', 'bootstrap', 'True', 'False', 1000, 0.0, 'mean/var'),
                ('None', 'bootstrap', 'True', 'False', 1000, 0.0, 'None'),
                ('probabilistic-ld', 'correct', 'True', 'False', 1000, 0.0, 'mean/var'),
                ('probabilistic-ld', 'correct', 'True', 'False', 1000, 0.0, 'None'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 100, 0.0, 'mean/var'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 100, 0.0, 'None'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 10000, 0.0, 'mean/var'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 10000, 0.0, 'None'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 100000, 0.0, 'mean/var'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 100000, 0.0, 'None'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 0.1, 'mean/var'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 0.1, 'None'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 0.2, 'mean/var'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 0.2, 'None'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 0.5, 'mean/var'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 0.5, 'None'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 1, 'mean/var'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 1, 'None'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 2, 'mean/var'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 2, 'None'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 5, 'mean/var'),
                ('probabilistic-ld', 'bootstrap', 'True', 'False', 1000, 5, 'None'),
            ):

                for setup_name, setup_string in setups.items():
                    if 'learnedinit' not in setup_name and n_init == 1:
                        continue
                    if normalization in ['Copula', 'mean/var'] and n_init < 2:
                        n_init_ = 2
                    else:
                        n_init_ = n_init
                    filename = "{seed}_50_{task_id}.json"
                    method_name = "%s-rgpe-%s-%s-%s-%s-%s-%d-%s%s-%d" % (
                        method, sampling_strategy, normalization.replace('/', ''),
                        weight_dilution_strategy,
                        'expectation' if use_expectation == 'True' else 'improvement',
                        'global' if use_global_incumbent == 'True' else 'local',
                        num_posterior_samples, alpha, setup_name, n_init_,
                    )
                    output_file = os.path.join(output_directory, benchmark, method_name, filename)
                    relative_output_file_template = os.path.join(benchmark, method_name, filename)
                    command = (
                        "{run_script} --benchmark {benchmark} --method {method} --seed {seed} "
                        "--task {task_id} {setup_string} --weight-dilution-strategy {weight_dilution_strategy} "
                        "--sampling-mode {sampling_strategy} "
                        "--use-expectation {use_expectation} "
                        "--use-global-incumbent {use_global_incumbent} "
                        "--weighting-mode rgpe --n-init {n_init} --output-file {output_file} "
                        "--normalization {normalization} --num-posterior-samples {num_posterior_samples} "
                        "--alpha {alpha}"
                    )

                    template = command.format(**{
                        'method': method,
                        'run_script': run_script,
                        'benchmark': benchmark,
                        'output_file': output_file,
                        'seed': '{seed}',
                        'task_id': '{task_id}',
                        'n_init': n_init_,
                        'setup_string': setup_string.format(
                            learned_initial_design=normalization_to_initial_design[normalization]),
                        'weight_dilution_strategy': weight_dilution_strategy,
                        'sampling_strategy': sampling_strategy,
                        'normalization': normalization,
                        'num_posterior_samples': num_posterior_samples,
                        'use_global_incumbent': use_global_incumbent,
                        'use_expectation': use_expectation,
                        'alpha': alpha,
                    })
                    commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))


    # RGPE
    method = 'rgpe'
    for n_init in (1, ):
        for sampling_strategy in ('bootstrap', ):
            for weight_dilution_strategy in ('None', 'probabilistic-ld'):
                for acquisition_function in ('NoisyEI', 'fullmodelNoisyEI',
                                             'targetEI', 'fullmodeltargetEI',
                                             'CFNEI',
                                             'EI', 'fullmodelEI'):
                    for normalization in ('mean/var', 'Copula', 'None'):
                        for setup_name, setup_string in setups.items():
                            if 'learnedinit' not in setup_name:
                                continue
                            if normalization in ['Copula', 'mean/var'] and n_init < 2:
                                n_init_ = 2
                            else:
                                n_init_ = n_init
                            filename = "{seed}_50_{task_id}.json"
                            method_name = "%s-%s-%s-%s-%s-1000-%s-%d" % (
                                    method, sampling_strategy, normalization.replace('/', ''),
                                    weight_dilution_strategy, acquisition_function, setup_name,
                                    n_init_)
                            output_file = os.path.join(
                                output_directory, benchmark, method_name, filename,
                            )
                            relative_output_file_template = os.path.join(benchmark, method_name, filename)
                            command = (
                                "{run_script} --benchmark {benchmark} --method {method} --seed {seed} "
                                "--task {task_id} {setup_string} --weight-dilution-strategy {weight_dilution_strategy} "
                                "--sampling-mode {sampling_strategy} --normalization {normalization} "
                                "--num-posterior-samples 1000 --n-init {n_init} --output-file {output_file} "
                                "--variance-mode average "
                            )
                            if acquisition_function == 'targetEI':
                                command = '%s --variance-mode target --acquisition-function-name EI' % command
                            elif acquisition_function == 'fullmodeltargetEI':
                                command = '%s --variance-mode target --acquisition-function-name fullmodelEI' % command
                            elif acquisition_function == 'NoisyEI':
                                command = '%s --acquisition-function-name 30 ' \
                                          '--target-model-incumbent True' % command
                            elif acquisition_function == 'fullmodelNoisyEI':
                                command = '%s --acquisition-function-name 30 ' \
                                          '--target-model-incumbent False' % command
                            else:
                                command = '%s --acquisition-function-name %s' % (command, acquisition_function)

                            template = command.format(**{
                                'method': method,
                                'run_script': run_script,
                                'benchmark': benchmark,
                                'output_file': output_file,
                                'seed': '{seed}',
                                'task_id': '{task_id}',
                                'n_init': n_init_,
                                'setup_string': setup_string.format(
                                    learned_initial_design=normalization_to_initial_design[normalization]),
                                'weight_dilution_strategy': weight_dilution_strategy,
                                'sampling_strategy': sampling_strategy,
                                'acquisition_function': acquisition_function,
                                'normalization': normalization,
                            })
                            commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))

    # Old RGPE
    method = 'rgpe'
    for n_init in (1, ):
        for weight_dilution_strategy in ('None', '95'):
            for sampling_strategy in ('correct', ):
                for acquisition_function in ('fullmodelNoisyEI', ):
                    for normalization in ('mean/var', 'Copula', 'None'):
                        for setup_name, setup_string in setups.items():
                            if 'learnedinit' not in setup_name:
                                continue
                            if normalization in ['Copula', 'mean/var'] and n_init < 2:
                                n_init_ = 2
                            else:
                                n_init_ = n_init
                            filename = "{seed}_50_{task_id}.json"
                            method_name = "%s-%s-%s-%s-%s-1000-%s-%d" % (
                                    method, sampling_strategy, normalization.replace('/', ''),
                                    weight_dilution_strategy, acquisition_function, setup_name,
                                    n_init_)
                            output_file = os.path.join(
                                output_directory, benchmark, method_name, filename,
                            )
                            relative_output_file_template = os.path.join(benchmark, method_name, filename)
                            command = (
                                "{run_script} --benchmark {benchmark} --method {method} --seed {seed} "
                                "--task {task_id} {setup_string} --weight-dilution-strategy {weight_dilution_strategy} "
                                "--sampling-mode {sampling_strategy} --normalization {normalization} "
                                "--num-posterior-samples 1000 --n-init {n_init} --output-file {output_file} "
                                "--variance-mode average "
                            )

                            if acquisition_function == 'NoisyEI':
                                command = '%s --acquisition-function-name 30 ' \
                                          '--target-model-incumbent True' % command
                            elif acquisition_function == 'fullmodelNoisyEI':
                                command = '%s --acquisition-function-name 30 ' \
                                          '--target-model-incumbent False' % command
                            else:
                                raise ValueError(acquisition_function)

                            template = command.format(**{
                                'method': method,
                                'run_script': run_script,
                                'benchmark': benchmark,
                                'output_file': output_file,
                                'seed': '{seed}',
                                'task_id': '{task_id}',
                                'n_init': n_init_,
                                'setup_string': setup_string.format(
                                    learned_initial_design=normalization_to_initial_design[normalization]),
                                'weight_dilution_strategy': weight_dilution_strategy,
                                'sampling_strategy': sampling_strategy,
                                'acquisition_function': acquisition_function,
                                'normalization': normalization,
                            })
                            commands.extend(add_seeds_and_tasks(template, n_seeds, n_tasks, relative_output_file_template))

    random.shuffle(commands)
    print(len(commands))
    string = "\n".join(commands)

    commands_file = os.path.join(output_directory, benchmark, 'commands.txt')
    print(commands_file)
    with open(commands_file, 'w') as fh:
        fh.write(string)
