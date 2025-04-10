# RecoveryChaining

Code for the paper [RecoveryChaining: Learning Local Recovery Policies for Robust Manipulation](https://arxiv.org/pdf/2410.13979) (currently under review).

## Installation

This repository is designed to be used with the [Anaconda](https://www.anaconda.com/products/distribution) package manager and has been tested with Python 3.10.

First, install Anaconda or Miniconda. Then, create a new conda environment with the following command:

```conda env create -f environment.yml```

This will create a new conda environment named `recovery_chaining` with all the required dependencies.

Next, activate the environment:

```conda activate recovery_chaining```

Finally, install the package in editable mode. This allows you to make changes to the code and have them reflected immediately without needing to reinstall the package.

```pip install -e .```

## Usage

### Collect failures

```python scripts/learn_recoveries.py failures.learn=True env=pick_place rl=rc_pick_place failures.nfails=100```

This will roll out the nominal controllers for the `pick-place` environment and record 100 failures. The results will be saved in the `belief_srs/outputs` by default. Move the generated files to a data folder (e.g. `data/pick_place`) so that they can be reused to learn recoveries.

### Learn Recoveries

```scripts/learn_recoveries.py recoveries.learn=True env=pick_place rl=rc_pick_place pick_place_data_dir=data/pick_place```

This will train a recovery policy to solve the collected failures using RecoveryChaining.

Run the following script to use Lazy RecoveryChaining:

```scripts/learn_recoveries.py recoveries.learn=True env=pick_place rl=rc_pick_place pick_place_data_dir=data/pick_place rl.use_nominal_precond=True```
