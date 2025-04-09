import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    TensorDictReplayBuffer,
    LazyTensorStorage,
    ListStorage,
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import (
    Compose,
    ExplorationType,
    FlattenObservation,
    GrayScale,
    InitTracker,
    ObservationNorm,
    Resize,
    RewardSum,
    StepCounter,
    RewardScaling,
    set_exploration_type,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder
from torchrl.modules.models.utils import SquashDims
from torchrl.modules import (
    ConvNet,
    EGreedyModule,
    LSTMModule,
    MLP,
    QValueModule,
    QValueActor,
)
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.utils import ValueEstimators
import tqdm
import wandb

logger = logging.getLogger(__name__)


class PPOgentTorchrl:
    def __init__(self, make_env_fn, cfg, wandb_logger):
        self.make_env_fn = make_env_fn
        self.cfg = cfg
        self.wandb_logger = wandb_logger

        self.stoch_policy = None
        self.collector = None
        self.replay_buffer = None

    def learn(self, total_timesteps, callback=None):
        rl_cfg = self.cfg
        device = torch.device(0) if torch.cuda.device_count() else torch.device("cpu")

        def wrap_env(env, is_test=False):
            wrapped_env = TransformedEnv(
                env,
                Compose(
                    ToTensorImage(),
                    GrayScale(),
                    Resize(84, 84),
                    RewardSum(),
                    StepCounter(),
                    # RewardScaling(loc=0.0, scale=0.1),
                    # ObservationNorm(standard_normal=True, in_keys=["pixels"]),
                ),
            )
            if self.cfg.recurrent:
                wrapped_env.append_transform(InitTracker())
            if is_test:
                wrapped_env.insert_transform(
                    0, VideoRecorder(logger=self.wandb_logger, tag="dqn")
                )
            return wrapped_env

        train_env = wrap_env(self.make_env_fn(is_test=False), is_test=False)
        test_env = wrap_env(self.make_env_fn(is_test=True), is_test=True)

        # env.transform[-1].init_stats(1000, reduce_dim=[0, 1, 2], cat_dim=0, keep_dims=[0])

        if not self.is_initialized():
            self._init(train_env)

        collector = self.collector
        rb = self.replay_buffer
        stoch_policy = self.stoch_policy
        loss_fn = self.loss_fn
        optim = self.optim
        advantage_module = self.advantage_module

        # training loop
        num_epochs = rl_cfg.num_epochs
        logger.info(f"  #epochs: {num_epochs}")

        pbar = tqdm.tqdm(total=rl_cfg.total_timesteps)

        if wandb.run is not None:
            wandb.watch(stoch_policy, log="all", log_freq=10000)
            wandb.define_metric("total_timesteps")

        n_timesteps, self._n_updates = 0, 0
        for i, data in enumerate(collector):
            pbar.update(data.numel())
            n_timesteps += data.numel()
            advantage_module(data)
            if self.cfg.recurrent:
                # XXX
                # sequential updates
                # ----------------
                # this adds each trajectory separately
                # benefit: don't need to reset hidden state to 0
                rb.extend(data.unsqueeze(0).to_tensordict().cpu())
            else:
                # random updates
                # we want to uniformly sample transition though (or don't we?)
                rb.extend(data.to_tensordict().cpu())

            mean_rew = data["next", "reward"].mean().item()
            if wandb.run is not None:
                wandb.log(
                    {
                        "total_timesteps": n_timesteps,
                        "train/n_rollouts": i,
                        "rollout/epsilon": self.greedy_module.eps.item(),
                        "rollout/rew_mean": mean_rew,
                    }
                )
            # Get and log training rewards and episode lengths
            episode_rewards = data["next", "episode_reward"][data["next", "done"]]
            if len(episode_rewards) > 0:
                episode_reward_mean = episode_rewards.mean().item()
                episode_length = data["next", "step_count"][data["next", "done"]]
                episode_length_mean = episode_length.sum().item() / len(episode_length)
                if wandb.run is not None:
                    wandb.log(
                        {
                            "rollout/episode_reward": episode_reward_mean,
                            "rollout/episode_length": episode_length_mean,
                        }
                    )
            # if n_timesteps > rl_cfg.learning_starts:
            total_loss = 0
            for _ in range(num_epochs):
                batch = rb.sample().to(device, non_blocking=True)
                loss_vals = loss_fn(batch)

                # update priorities using latest td_error
                rb.update_tensordict_priority(batch)

                total_loss += loss_vals["loss"].item()
                optim.zero_grad()
                loss_vals["loss"].backward()
                nn.utils.clip_grad_norm_(
                    stoch_policy.parameters(), rl_cfg.max_grad_norm
                )
                optim.step()
                if rl_cfg.delay_value:
                    # if n_timesteps % rl_cfg.target_update_interval == 0:
                    # updater.step()
                    updater.step()

                self._n_updates += gradient_steps

                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/q_loss": total_loss / gradient_steps,
                            "train/n_updates": self._n_updates,
                            "train/q_values": (data["action_value"] * data["action"])
                            .sum()
                            .item(),
                        }
                    )

                pbar.set_description(
                    f"loss_val: {total_loss: 4.4f}, action_spread: {data['action'].sum(0)}"
                )
                # stoch_policy.step(data.numel())
                self.greedy_module.step(data.numel())

            if n_timesteps % rl_cfg.eval_freq == 0:
                self.eval(
                    test_env,
                    self.stoch_policy,
                    num_episodes=rl_cfg.eval_episodes,
                    suffix="eval_" + str(n_timesteps),
                )

    def eval(self, env, policy, num_episodes, suffix=None):
        """Evaluate the policy by calculating the mean reward over num_episodes episodes"""

        logger.info(" Evaluating policy...")
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            policy.eval()
            test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
            for i in range(num_episodes):
                td_test = env.reset()
                td_test = env.rollout(
                    max_steps=1000,
                    policy=policy,
                    auto_reset=True,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                reward = td_test["next", "episode_reward"][td_test["next", "done"]]
                test_rewards[i] = reward.sum()
            if wandb.run is not None:
                wandb.log(
                    {
                        "eval/reward_mean": test_rewards.mean(),
                    }
                )
        policy.train()
        env.transform.dump(suffix=suffix)
        return test_rewards.mean()

    def _init(self, env):
        rl_cfg = self.cfg
        device = torch.device(0) if torch.cuda.device_count() else torch.device("cpu")
        env.reset()

        # feature extractor
        def _make_feature_extractor():
            # NatureCNN architecture from stable baselines
            convnet = ConvNet(
                num_cells=[32, 64, 64],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                aggregator_class=SquashDims,
                activation_class=nn.ELU,
                device=device,
            )
            return convnet

        # lstm
        if rl_cfg.recurrent:

            def _make_model():
                feature = _make_feature_extractor()
                n_cells = feature(env.reset()["pixels"]).shape[-1]
                feature = TensorDictModule(
                    feature, in_keys=["pixels"], out_keys=["embed"]
                )
                # lstm = nn.LSTM(
                lstm_mod = LSTMModule(
                    input_size=n_cells,
                    hidden_size=rl_cfg.lstm_hidden_size,
                    num_layers=rl_cfg.lstm_num_layers,
                    device=device,
                    in_key="embed",
                    out_key="embed",
                )
                print(lstm_mod)

                # mlp
                mlp = MLP(
                    # mean and std
                    out_features=2 * env.action_spec.shape[0],
                    num_cells=[128, 128],
                    activation_class=nn.ReLU,
                    device=device,
                )
                return mlp

            # Actor
            actor = _make_model()
            actor = nn.Sequential(actor, NormalParamExtractor())
            policy_module = TensorDictModule(
                actor, in_keys=["embed"], out_keys=["loc", "scale"]
            )
            policy_module = ProbabilisticActor(
                module=policy_module,
                spec=env.action_spec,
                in_keys=["loc", "scale"],
                distribution_class=TanhNormal,
                distribution_kwargs={
                    "min": env.action_spec.space.low,
                    "max": env.action_spec.space.high,
                },
                return_log_prob=True,
                # we'll need the log-prob for the numerator of the importance weights
            )

            # Critic
            critic = _make_model()
            critic = nn.Sequential(critic, nn.LazyLinear(1, device=device))
            critic_module = TensorDictModule(
                critic, in_keys=["embed"], out_keys=["state_value"]
            )

        else:

            def _make_model():
                feature = _make_feature_extractor()
                n_cells = feature(env.reset()["pixels"]).shape[-1]
                feature = TensorDictModule(
                    feature, in_keys=["pixels"], out_keys=["embed"]
                )
                # mlp
                mlp = MLP(
                    # mean and std
                    out_features=2 * env.action_spec.shape[0],
                    num_cells=[128, 128],
                    activation_class=nn.ReLU,
                    device=device,
                )
                return mlp

            # Actor
            actor = _make_model()
            actor = nn.Sequential(actor, NormalParamExtractor())
            policy_module = TensorDictModule(
                actor, in_keys=["embed"], out_keys=["loc", "scale"]
            )
            policy_module = ProbabilisticActor(
                module=policy_module,
                spec=env.action_spec,
                in_keys=["loc", "scale"],
                distribution_class=TanhNormal,
                distribution_kwargs={
                    "min": env.action_spec.space.low,
                    "max": env.action_spec.space.high,
                },
                return_log_prob=True,
                # we'll need the log-prob for the numerator of the importance weights
            )

            # Critic
            critic = _make_model()
            critic = nn.Sequential(critic, nn.LazyLinear(1, device=device))
            critic_module = TensorDictModule(
                critic, in_keys=["embed"], out_keys=["state_value"]
            )

        # replay buffer
        self.collector = SyncDataCollector(
            env,
            policy_module,
            frames_per_batch=rl_cfg.frames_per_batch,
            total_frames=rl_cfg.total_timesteps,
            init_random_frames=None,
        )
        storage = LazyMemmapStorage(
            max_size=rl_cfg["batch_size"]
            # if not rl_cfg.recurrent
            # else rl_cfg["buffer_size"] // rl_cfg["train_freq"],
            scratch_dir=".",
        )
        self.replay_buffer = ReplayBuffer(
            storage=storage,
            sampler=SamplerWithoutReplacement(),
        )
        self.advantage_module = GAE(
            gamma=rl_cfg.gamma,
            lmbda=rl_cfg.lmbda,
            value_network=value_module,
            average_gae=True,
        )

        self.loss_fn = ClipPPOLoss(
            actor=policy_module,
            critic=critic_module,
            clip_epsilon=rl_cfg.clip_epsilon,
            entropy_bonus=bool(rl_cfg.entropy_eps),
            entropy_coef=rl_cfg.entropy_eps,
            # these keys match by default but we set this for completeness
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )

        self.optim = torch.optim.Adam(
            loss_module.parameters(), rl_cfg.learning_rate, eps=1e-5
        )
        # XXX
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            # optim, total_frames // frames_per_batch, 0.0
        # )

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def is_initialized(self):
        return not any(
            [
                self.stoch_policy is None,
                self.collector is None,
                self.replay_buffer is None,
            ]
        )

    def load(self, ckp_path):
        logger.info(f"Loading model from {ckp_path}")
        ckp = torch.load("ckp_path")
        self.stoch_policy.load_state_dict(ckp["stoch_policy_state_dict"])

    def save(self, ckp_path, epoch, loss):
        torch.save(
            {
                "epoch": epoch,
                "stoch_policy_state_dict": self.stoch_policy.state_dict(),
                "loss": loss,
            },
            ckp_path,
        )
