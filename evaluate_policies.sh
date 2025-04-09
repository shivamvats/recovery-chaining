#!/bin/bash

method=rrl #rrl  #rc rl
env=shelf_clutter #shelf_clutter
#env=pick
seed=92

if [ $method = "rl" ]; then
    reward_shaping=True
    rl_cfg=rl
else
    reward_shaping=False
    rl_cfg=${method}_${env}
fi

echo "Env: ${env}"
echo "Method: ${method}"

# evaluate recoveries
#
python scripts/learn_recoveries.py evaluate.evaluate_chain=True nevals=50 \
evaluate.algo=${method} evaluate.env_name=${env} env=${env} rl=${rl_cfg} \
tag=eval_${method}_${env}_${seed} group=${method}_${env} \
seed=${seed} output_dir=results/ral/${env}/${method} \
env.reward_shaping=${reward_shaping}
