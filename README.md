# belief_srs: Hierarchical reinforcement learning for manipulation

## Features

Provide a summary of what the software does, why it's important and a list of features.

## Installation

Run ```pip install -e .```

## Usage

## Reproduce

### Collect failures

python scripts/learn_recoveries.py failures.learn=True env=shelf rl=drl_shelf tag="failures" output_dir=data/shelf/31-Jan

### Learn Preconditions

python scripts/learn_recoveries.py value_fn.learn=True

### Learn Recoveries

**Recovery Chaining Results**

seeds = 42, 52, 19, 74, 102

scripts/learn_recoveries.py recoveries.learn=True tag=rc rl.algorithm=PPO env=pick_place rl=drl_pick_place seed=42 output_dir=results/rss/pick_place/rc

scripts/learn_recoveries.py recoveries.learn=True tag=rc rl.algorithm=PPO env=shelf rl=drl_shelf seed=42 output_dir=results/rss/shelf/rc

** RL results**

python scripts/learn_recoveries.py recoveries.learn=True tag=rl rl.algorithm=PPO  env=pick_place rl=rl seed=52 output_dir=results/rss/pick_place/rl env.reward_shaping=True

## Evaluation

python scripts/learn_recoveries.py evaluate.evaluate_chain=True env=pick_place rl=drl_pick_place  nevals=50

## Related Links

Add any related links, such as cwiki pages, fileserver folders, etc.

## Contact

Add your contact details

## License
