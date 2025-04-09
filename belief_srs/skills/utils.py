import logging
import gymnasium as gym
import numpy as np
import os
import stable_baselines3.common.callbacks as cb
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import wandb

logger = logging.getLogger(__name__)


class EvalCallback(cb.EvalCallback):
    """
    Adds features on top of sb3 EvalCallback:
        1. Start saving after a `min_train_timesteps`
        2. Save vennorm env along with best_model.
    """

    def __init__(self, min_train_timesteps=20000, **kwargs):
        super().__init__(**kwargs)
        self.min_train_timesteps = min_train_timesteps
        # self.eval_eps = []

    def _on_step(self):
        def on_step():
            continue_training = True

            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # logger.info(f"  n_calls: {self.n_calls}, eval_freq: {self.eval_freq}, eval_eps: {self.eval_eps}")
            # if self.eval_freq > 0 and (self.n_calls // self.eval_freq) not in self.eval_eps:
                # self.eval_eps.append(self.n_calls // self.eval_freq)
                logger.info("Evaluating the model's performance...")
                logger.info("--------------------")
                # Sync training and eval env if there is VecNormalize
                if self.model.get_vec_normalize_env() is not None:
                    try:
                        sync_envs_normalization(self.training_env, self.eval_env)
                    except AttributeError as e:
                        raise AssertionError(
                            "Training and eval env are not wrapped the same way, "
                            "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                            "and warning above."
                        ) from e

                # Reset success rate buffer
                self._is_success_buffer = []

                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )

                if self.log_path is not None:
                    self.evaluations_timesteps.append(self.num_timesteps)
                    self.evaluations_results.append(episode_rewards)
                    self.evaluations_length.append(episode_lengths)

                    kwargs = {}
                    # Save success log if present
                    if len(self._is_success_buffer) > 0:
                        self.evaluations_successes.append(self._is_success_buffer)
                        kwargs = dict(successes=self.evaluations_successes)

                    np.savez(
                        self.log_path,
                        timesteps=self.evaluations_timesteps,
                        results=self.evaluations_results,
                        ep_lengths=self.evaluations_length,
                        **kwargs,
                    )

                mean_reward, std_reward = np.mean(episode_rewards), np.std(
                    episode_rewards
                )
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                    episode_lengths
                )
                self.last_mean_reward = mean_reward

                if self.verbose >= 1:
                    print(
                        f"Eval num_timesteps={self.num_timesteps}, "
                        f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                    )
                    print(
                        f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}"
                    )
                # Add to current Logger
                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record("eval/mean_ep_length", mean_ep_length)

                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    if self.verbose >= 1:
                        print(f"Success rate: {100 * success_rate:.2f}%")
                    self.logger.record("eval/success_rate", success_rate)

                # Dump log so the evaluation results are printed with the correct timestep
                self.logger.record(
                    "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
                )
                self.logger.dump(self.num_timesteps)

                if mean_reward > self.best_mean_reward:
                    if self.verbose >= 1:
                        print("New best mean reward!")
                    if self.best_model_save_path is not None:
                        self.model.save(
                            os.path.join(self.best_model_save_path, "best_eval_model")
                        )
                        # save vecnorm_env
                        if self.model.get_vec_normalize_env() is not None:
                            vec_normalize_path = os.path.join(
                                self.best_model_save_path, "best_eval_vecnorm.pkl"
                            )
                            self.model.get_vec_normalize_env().save(vec_normalize_path)

                    self.best_mean_reward = mean_reward
                    # Trigger callback on new best model, if needed
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()

                # Trigger callback after every evaluation, if needed
                if self.callback is not None:
                    continue_training = continue_training and self._on_event()

            return continue_training

        if self.n_calls >= self.min_train_timesteps:
            continue_training = on_step()
        else:
            continue_training = True

        return continue_training


class WandbCallback(cb.BaseCallback):
    """
    wandb logging
    """

    def __init__(self, best_model_save_path=None, log_path="./policies/", verbose=0):
        super().__init__(verbose)
        self.best_mean_reward = -np.inf
        self.best_model_save_path = best_model_save_path

    def _on_rollout_start(self):
        pass

    def _on_rollout_end(self):
        env = self.training_env
        # potential_fn = self.model.potential_fn
        term_action_ids = env.env_method("get_term_action_ids", indices=0)[0]
        actions = np.array(self.locals["all_actions"])
        values = self.locals["all_values"]
        infos = self.locals["infos"]
        n_fails = np.sum([info["is_failure"] for info in infos])
        # true_states = self.locals["all_true_states"]
        # pot_values = potential_fn(true_states)
        wandb_dict = {
            "rollout/value_mean": np.mean(values),
            "rollout/n_fails": n_fails,
        }
        if "precond_rew" in infos[0]:
            # for skill chaining
            wandb_dict["rollout/precond_rew"] = np.mean([info["precond_rew"] for info in infos])

        if term_action_ids:
            # nom_pot_values = np.array(pot_values)[
            # [action in term_action_id for action in actions]
            # ]
            # num_nom_actions = np.sum([action in term_action_ids for action in actions])
            # nom_advantage = self.model.rollout_buffer.all_nom_advantages
            for nom_id in term_action_ids:
                num_nom_actions = np.sum(actions == nom_id)
                wandb_dict.update(
                    {
                        f"rollout/nominal_action_{nom_id}": num_nom_actions,
                        "timesteps": self.model.num_timesteps,
                    }
                )
        self._save_model()
        wandb.log(wandb_dict)

    def _on_step(self) -> bool:
        return True

    def _save_model(self):
        mean_reward = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])

        if mean_reward > self.best_mean_reward:
            if self.verbose >= 1:
                logger.info(f"New best mean reward! {mean_reward}")

            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                # save vecnorm_env
                if self.model.get_vec_normalize_env() is not None:
                    vec_normalize_path = os.path.join(
                        self.best_model_save_path, "best_vecnorm.pkl"
                    )
                    self.model.get_vec_normalize_env().save(vec_normalize_path)

            self.best_mean_reward = mean_reward

        return True


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
