import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy, DiscreteSACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.env.venvs import DummyVectorEnv, SubprocVectorEnv
from belief_srs.utils.layers import NatureCNN, AtariDQN
from belief_srs.utils.tianshou_wandb import WandbLogger


class SACDAgentTianshou:
    def __init__(self, make_env_fn, cfg, wandb_logger):
        self.make_env_fn = make_env_fn
        self.cfg = cfg
        self.wandb_logger = wandb_logger

        self.stoch_policy = None
        self.collector = None
        self.replay_buffer = None
        self.train_env = make_env_fn(test=False)
        self.test_env = make_env_fn(test=True)
        self._init(train_env)

    def learn(self, total_timesteps):
        rl_cfg = self.cfg.rl
        self.train_collector.collect(n_step=rl_cfg.start_timesteps, random=True)

        # logger
        wb_logger = WandbLogger(
            project=cfg.wandb_project,
            name=cfg.tag,
            config=OmegaConf.to_container(cfg, resolve=True),
            group=cfg.group,
            train_interval=1,
            test_interval=1,
            update_interval=1,
        )
        wb_logger.load(SummaryWriter("log"))

        def save_best_fn(policy: BasePolicy) -> None:
            torch.save(policy.state_dict(), "policy.pth")

        # trainer
        result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=rl_cfg.epoch,
            step_per_epoch=rl_cfg.step_per_epoch,
            step_per_collect=rl_cfg.step_per_collect,
            update_per_step=rl_cfg.step_per_collect,
            episode_per_test=rl_cfg.test_num,
            batch_size=rl_cfg.batch_size,
            save_best_fn=save_best_fn,
            logger=wb_logger,
            test_in_train=False,
        ).run()
        pprint.pprint(result)

        # Let's watch its performance!
        policy.eval()
        test_envs.seed(cfg.seed)
        test_collector.reset()
        collector_stats = test_collector.collect(
            n_episode=rl_cfg.test_num, render=cfg.render
        )
        print(collector_stats)

    def eval(self, env, policy, num_episodes, suffix=None):
        pass


    def _init(self, env):
        cfg = self.cfg
        rl_cfg = cfg.rl
        state_shape = env.observation_space[0].shape or env.observation_space[0].n
        action_shape = env.action_space[0].shape or env.action_space[0].n
        print("Observations shape:", state_shape)
        print("Actions shape:", action_shape)
        # seed
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        # model
        # hidden_sizes = rl_cfg.hidden_layers * [rl_cfg.hidden_size]
        hidden_sizes = rl_cfg.hidden_sizes

        if rl_cfg.feature_net == "NatureCNN":
            feature_net_actor = NatureCNN(
                observation_space=env.observation_space[0],
                features_dim=rl_cfg.features_dim,
                normalized_image=True,
                device=cfg.device,
            ).to(cfg.device)
            feature_net_critic = NatureCNN(
                observation_space=env.observation_space[0],
                features_dim=rl_cfg.features_dim,
                normalized_image=True,
                device=cfg.device,
            ).to(cfg.device)

        elif rl_cfg.feature_net == "AtariDQN":
            feature_net_actor = AtariDQN(
                *state_shape,
                action_shape,
                device=cfg.device,
                features_only=True,
                output_dim=rl_cfg.features_dim,
            )
            if rl_cfg.share_features:
                feature_net_critic = feature_net_actor
            else:
                feature_net_critic = AtariDQN(
                    *state_shape,
                    action_shape,
                    device=cfg.device,
                    features_only=True,
                    output_dim=rl_cfg.features_dim,
                )

        else:
            raise NotImplementedError
        actor = Actor(
            feature_net_actor,
            action_shape,
            preprocess_net_output_dim=feature_net_actor.features_dim,
            hidden_sizes=hidden_sizes,
            device=cfg.device,
            softmax_output=False,  # SAC will normalize
        ).to(cfg.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=rl_cfg.actor_lr)

        critic1 = Critic(
            feature_net_critic,
            hidden_sizes=hidden_sizes,
            preprocess_net_output_dim=feature_net_critic.features_dim,
            last_size=action_shape,
            device=cfg.device,
        ).to(cfg.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=rl_cfg.critic_lr)

        critic2 = Critic(
            feature_net_critic,
            hidden_sizes=hidden_sizes,
            preprocess_net_output_dim=feature_net_critic.features_dim,
            last_size=action_shape,
            device=cfg.device,
        ).to(cfg.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=rl_cfg.critic_lr)

        logger.info(f"Actor: {actor}")
        logger.info(f"Critic: {critic1}")

        if rl_cfg.auto_alpha:
            target_entropy = 0.98 * np.log(np.prod(action_shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=cfg.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=rl_cfg.alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)
        else:
            alpha = rl_cfg.alpha

        policy = DiscreteSACPolicy(
            actor=actor,
            actor_optim=actor_optim,
            critic1=critic1,
            critic1_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=rl_cfg.tau,
            gamma=rl_cfg.gamma,
            alpha=alpha,
            estimation_step=rl_cfg.n_step,
            action_space=env.action_space,
            deterministic_eval=False,
        ).to(cfg.device)

        # load a previous policy
        if cfg.resume_path:
            policy.load_state_dict(torch.load(cfg.resume_path, map_location=cfg.device))
            print("Loaded agent from: ", cfg.resume_path)

        # collector
        buffer: VectorReplayBuffer | ReplayBuffer
        buffer = VectorReplayBuffer(rl_cfg.buffer_size, len(train_env))
        self.train_collector = Collector(policy, train_env, buffer, exploration_noise=True)
        self.test_collector = Collector(policy, test_env)
