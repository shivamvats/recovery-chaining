#!/bin/bash


method=rc #sc  #rc rl
#env=shelf
env=shelf_clutter
#env=pick_place

if [ $method = "rl" ]; then
    reward_shaping=True
else
    reward_shaping=False
fi

if [ $method = "sc" ]; then
    learn_value_fn=True
else
    learn_value_fn=False
fi

echo "Env: ${env}"
echo "Method: ${method}"
echo "Reward shaping: ${reward_shaping}"

# learn recoveries

for seed in 52 19 74 102 42
#for seed in 19 74 102 42
#for seed in 42
do
    python scripts/learn_recoveries.py recoveries.learn=True \
    tag=${method}_${env}_${seed}_no_keep group=${method}_precond_${env} env=${env} rl=${method}_${env} \
    seed=${seed} output_dir=results/iros/${env}/${method} value_fn.learn=${learn_value_fn} \
    num_cpus=10 rl.algorithm=PPO env.reward_shaping=${reward_shaping} rl.use_nominal_precond=True rl.learn.n_total_steps=200_000 device=cuda #& #500_00\
done

# debug
#for seed in 42
#do
    #python scripts/learn_recoveries.py recoveries.learn=True \
    #group=debug env=${env} rl=${method}_${env} \
    #seed=${seed} value_fn.learn=${learn_value_fn} \
    #num_cpus=10 rl.algorithm=PPO env.reward_shaping=${reward_shaping} rl.learn.n_total_steps=50_000 render=False
#done
