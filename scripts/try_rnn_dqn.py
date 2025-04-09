from matplotlib import pyplot as plt
import numpy as np
from time import sleep, time
import torch
import tqdm
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import (
    Compose,
    ExplorationType,
    GrayScale,
    InitTracker,
    ObservationNorm,
    Resize,
    RewardScaling,
    set_exploration_type,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ConvNet, EGreedyWrapper, LSTMModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate
import wandb

device = torch.device(0) if torch.cuda.device_count() else torch.device("cpu")


def make_policy(env):
    feature = Mod(
        ConvNet(
            num_cells=[32, 32, 64],
            squeeze_output=True,
            aggregator_class=nn.AdaptiveAvgPool2d,
            aggregator_kwargs={"output_size": (1, 1)},
            device=device,
        ),
        in_keys=["pixels"],
        out_keys=["embed"],
    )
    n_cells = feature(env.reset())["embed"].shape[-1]
    lstm = LSTMModule(
        input_size=n_cells,
        hidden_size=128,
        device=device,
        in_key="embed",
        out_key="embed",
    )
    print("in_keys", lstm.in_keys)
    print("out_keys", lstm.out_keys)

    mlp = MLP(
        out_features=env.action_spec.shape[0],
        num_cells=[
            64,
        ],
        device=device,
    )
    mlp[-1].bias.data.fill_(0.0)
    mlp = Mod(mlp, in_keys=["embed"], out_keys=["action_value"])
    qval = QValueModule(action_space=env.action_spec)

    env.append_transform(lstm.make_tensordict_primer())
    policy = Seq(feature, lstm, mlp, qval)
    # policy = EGreedyWrapper(
    # policy, annealing_num_steps=1_000_000, spec=env.action_spec, eps_init=0.2
    # )
    policy = EGreedyWrapper(
        policy,
        annealing_num_steps=100_000,
        spec=env.action_spec,
        eps_init=1,
        eps_end=0.1,
    )
    # policy = Seq(feature, lstm.set_recurrent_mode(True), mlp, qval)

    policy(env.reset())

    return policy


def main():
    run = wandb.init(
        project="belief_srs",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
    )
    env = TransformedEnv(
        GymEnv("CartPole-v1", from_pixels=True, device=device),
        # GymEnv("MountainCar-v0", from_pixels=True, device=device),
        # GymEnv("LunarLander-v2", from_pixels=True, device=device),
        Compose(
            ToTensorImage(),
            GrayScale(),
            Resize(84, 84),
            StepCounter(),
            InitTracker(),
            RewardScaling(loc=0.0, scale=0.1),
            ObservationNorm(standard_normal=True, in_keys=["pixels"]),
        ),
    )
    env.transform[-1].init_stats(1000, reduce_dim=[0, 1, 2], cat_dim=0, keep_dims=[0])
    td = env.reset()
    policy = make_policy(env)
    loss_fn = DQNLoss(policy, action_space=env.action_spec, delay_value=True)
    updater = SoftUpdate(loss_fn, eps=0.95)

    # optim = torch.optim.Adam(policy.parameters(), lr=3e-4)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
    # total_timesteps = 1000
    total_timesteps = 200_000
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=50,
        total_frames=total_timesteps
        # env, policy, frames_per_batch=50, total_frames=1_000_000
    )
    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(20_000), batch_size=8, prefetch=10
    )
    utd = 4

    pbar = tqdm.tqdm(total=total_timesteps)
    longest = 0

    traj_lens, rews = [], []
    for i, data in enumerate(collector):
        if i == 0:
            print(
                "Let us print the first batch of data.\nPay attention to the key names "
                "which will reflect what can be found in this data structure, in particular: "
                "the output of the QValueModule (action_values, action and chosen_action_value),"
                "the 'is_init' key that will tell us if a step is initial or not, and the "
                "recurrent_state keys.\n",
                data,
            )
        pbar.update(data.numel())
        # it is important to pass data that is not flattened
        buffer.extend(data.unsqueeze(0).to_tensordict().cpu())
        rews = []
        total_loss = 0
        for _ in range(utd):
            s = buffer.sample().to(device)
            mean_rew = s["next", "reward"].mean().item()
            # __import__('ipdb').set_trace()
            rews.append(mean_rew)
            loss_vals = loss_fn(s)
            total_loss += loss_vals["loss"].item()
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
        longest = max(longest, data["step_count"].max().item())
        wandb.log(
            {
                "rollout/mean_rew": np.mean(rews),
                "rollout/longest_episode": longest,
                "train/loss": total_loss / utd,
            }
        )
        pbar.set_description(
            f"rews: {np.mean(rews)}, steps: {longest}, loss_val: {loss_vals['loss'].item(): 4.4f}, action_spread: {data['action'].sum(0)}"
        )
        policy.step(data.numel())
        updater.step()

        # if i % 50 == 0:
        # # if i % 5 == 0:
        # with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        # rollout = env.rollout(10000, policy)
        # traj_lens.append(rollout.get(("next", "step_count")).max().item())
        # rews.append(rollout.get(("next", "reward")).sum().item())

        # if traj_lens:
        # plt.plot(traj_lens)
        # plt.plot(rews)
        # plt.xlabel("Test collection")
        # plt.title("Test trajectory lengths")
        # plt.savefig("policy_eval.png")


if __name__ == "__main__":
    main()
