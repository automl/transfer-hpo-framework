# Practical Transfer Learning for Bayesian Optimization

Code accompanying

    Practical Transfer Learning for Bayesian Optimization
    Matthias Feurer, Benjamin Letham, Frank Hutter and Eytan Bakshy
    https://arxiv.org/pdf/1802.02219v3.pdf

All code is developed using Python 3.7 and SMAC3 v0.12.3. The exact versions of the software 
used are given in environment.yaml.

## Guide to the code

### scripts/generate_commands.py

Generates the commands for running experiments. See bottom of this file for usage.

### scripts/run_benchmark.py

Main script. Configures SMAC to use the actual transfer learning methods. Then it applies SMAC 
to the chosen benchmark function and outputs a `.json` file containing the results

### scripts/install.sh

Installation file used to setup the conda environment. We cannot guarantee that this leads to 
the exact same environment that we used for our experiments.

### rgpe/methods

Contains the actual implementations of all methods used throughout the paper:

* ablr.py: Perrone et al., NeurIPS 2019
* GCPplusPrior.py: Salinas et al., ICML 2020
* kl_weighting.py: Ramachandran et al., ECML 2019
* noisy_ei.py: Letham et al., Bayesian Analysis, 2019
* rgpe.py: This paper
* rmogp.py: This paper
* taf.py: Wistuba et al., Machine Learning, 2018
* tstr.py: Wistuba et al., ECML 2016
* warmstarting_ac.py: Lindauer et al., AAAI 2018

### rgpe/test_functions.py

Implementation of all test functions used throughout the paper. Required data is either downloaded 
from the internet (for surrogates based on OpenML data), or needs to be downloaded manually 
(AdaBoost, SVM, LCBench).

To run the LCBench benchmark, close the [LCBench repository](https://github.com/automl/LCBench/) 
and set the paths in the class `NNGrid` to point to where you cloned the repository to, and the 
directories to where you downloaded the LCBench data 
(see [here](https://github.com/automl/LCBench/#downloading-the-data) for downloading the data).

### rgpe/adaboost

AdaBoost data from 
[Schilling et al.](https://github.com/nicoschilling/ECML2016/tree/master/data/adaboost). Please 
download these files from Nico's repository and place them here.

### rgpe/svm

SVM data from [Schilling et al.](https://github.com/nicoschilling/ECML2016/tree/master/data/svm).
Please download the files from Nico's repository and place them here.

### rgpe/utils.py

Helper functions for obtaining Gaussian process objects, conducting Sobol sequence construction 
and computing expected improvement.

## Example calls

### RGPE

```
python scripts/run_benchmark_smac.py --benchmark adaboost --method rgpe --seed 5 --task 20 \
    --empirical-meta-configs --learned-initial-design copula --weight-dilution-strategy probabilistic-ld \
    --sampling-mode bootstrap --normalization Copula --num-posterior-samples 1000 --n-init 1 \
    --output-file results/adaboost/rgpe-bootstrap-Copula-probabilistic-ld-NoisyEI-1000--gpmetadata-learnedinit-1/5_50_20.json \
    --variance-mode average  --acquisition-function-name 30 --target-model-incumbent False
```

### TAF

```
python scripts/run_benchmark_smac.py --benchmark adaboost --method taf --seed 8 --task 47 \
    --empirical-meta-configs --learned-initial-design unscaled --bandwidth 0.1  \
    --weighting-mode tstr --n-init 1 --normalization None --weight_dilution_strategy None \
    --output-file results/adaboost/taf-tstr-None-None-0.100000-gpmetadata-learnedinit-2/8_50_47.json
```

## How to reproduce the experiments

1. Install everything
2. Run only GP(MAP) to obtain data to warmstart transfer learning methods with. To obtain 
   commands for doing so run `python generate_commands.py --benchmark adaboost --setup None`
3. Run (almost) everything else. To obtain the commands for doing so run 
   `python generate_commands.py --benchmark adaboost --setup -gpmetadata-learnedinit`
4. Finally, also run some methods which do not contain a learned initialization:
   `python generate_commands.py --benchmark adaboost --setup -gpmetadata`

