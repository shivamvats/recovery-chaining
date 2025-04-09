import copy
from gymnasium.core import Env
import logging
import numpy as np
from belief_srs.skills.primitive_skill import PrimitiveSkill
from robosuite.wrappers import Wrapper
import time
import wandb

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


class DiscreteActionWrapper(Wrapper, Env):
    """Discrete actions."""

    def __init__(
        self,
        env,
        cfg,
        nom_reward_model=None,
        target_precondition_id="all",
        evaluate_skill_chain=False,
    ):
        super().__init__(env)
        self.cfg = cfg
        self.nom_reward_model = nom_reward_model
        self.target_precondition_id = target_precondition_id
        self.evaluate_skill_chain = evaluate_skill_chain
        self.actions, self.action_costs = self._compute_actions()
        self.term_actions = []
        self.execute_nominal_actions = True
        self._reset_internal()
        self.horizon = cfg.ll_horizon  # option_time measures ll steps
        self.nom_precond = None

    def reset(self):
        self._reset_internal()
        obs = self.env.reset()
        obs["action_hist"] = np.ones(20) * -1
        return obs
        # self.t = 0
        # self.reset_option_time()
        # return self._add_time(obs)

    # def reset_option_time(self):
    # self.option_t = 0

    def _reset_internal(self):
        self.done = False

    # def _add_time(self, obs):
    # # tracks primitive action history
    # obs["time"] = np.array([self.t])
    # # handle semi-markov nature of options
    # if "option_time" not in obs:
    # obs["option_time"] = np.array([self.option_t])
    # return obs

    def set_execute_nominal_actions(self, value):
        self.execute_nominal_actions = value

    def update_nom_precond(self, precond):
        self.nom_precond = precond

    def step(self, action, obs=None, render=False):
        assert action < len(self.actions)
        assert action >= 0

        if self.done:
            raise ValueError("executing action in terminated hl episode")

        logger.debug(f"  action: {action}")
        # prev_state = self._add_time(self.observe_true_state())
        prev_state = self.observe_true_state()

        prim = self.actions[action]
        term_action_called = action in self.term_actions

        if obs is None:
            obs = self.env._get_observations(force_update=False)

        start_time = time.time()
        start_timestep = obs["time"][0]

        if term_action_called and (not self.execute_nominal_actions):
            # Recovery chaining with reward model
            curr_state = self.observe_true_state()
            nom_action = action - self.term_actions[0]  # subtract offset
            if self.cfg.use_nominal_reward_model:
                rew = self.nom_reward_model([curr_state], nom_action)[0]
            # rew = np.clip(self.potential_fn([curr_state])[0], -3, 1)
            else:
                # critic is in PPO
                rew = 0
            done = True
            info = {"is_success": False, "nominal_action": True}
            # logger.debug(f"  Nominal action {action} called, predicted reward: {rew}")

        else:
            if hasattr(prim, "model"):
                logger.info("  executing learned action")
                obs, rew, done, info = prim.apply(obs, render=render)
            else:
                obs, rew, done, info = prim.apply(self.env, obs, render=render)

            if len(self.term_actions) == 0 and self.cfg.use_nominal_reward_model:
                # skill chaining
                curr_state = self.observe_true_state()
                if self.target_precondition_id == "all":
                    q_values = self.nom_reward_model([curr_state])[0]
                    precond_rew = np.max(q_values)
                else:
                    precond_rew = self.nom_reward_model([curr_state])[0][
                        self.target_precondition_id
                    ]
                # precond lies b/w 0 and 1
                precond_rew = np.clip(precond_rew, 0, 1)
                rew += precond_rew

                info["precond_rew"] = precond_rew
                logger.debug(f"  Precondition reward: {precond_rew}")

        if term_action_called:
            logger.debug(f"  Terminal action {prim} called")
            info["nominal_action"] = True
            done = True
        else:
            info["nominal_action"] = False

        info["timesteps"] = obs["time"][0] - start_timestep

        if done:
            logger.debug("  Env terminated")

        if obs["option_time"].item() >= self.horizon:
        # if obs["time"].item() >= self.horizon:
            done = True
            logger.debug(f"  Timeout")

        logger.debug(
            f"  Timesteps: {info['timesteps']} option_time: {obs['option_time'].item()} done: {done}, start: {start_timestep}, end: {obs['time'][0]}"
        )

        if obs["option_time"].item() >= self.horizon and self.evaluate_skill_chain:
            # evaluate policy by executing nominal chain
            logger.debug("  Evaluating skill chain")
            # pick best chain
            nom_chains = self._nominal_chains
            curr_state = self.observe_true_state()
            preconds = self.nom_reward_model([curr_state])[0]
            precond_sat = np.argmax(preconds)
            logger.debug(f"  Preconds: {preconds}")
            logger.debug(
                f"  Chosen nominal skill: {precond_sat, nom_chains[precond_sat]}"
            )
            # execute nominal chain
            obs, rew, done, info = nom_chains[precond_sat].apply(
                self.env, obs, render=render
            )
            done = True

        self.done = done

        if done or term_action_called:
            # exceeting action timout is not an env termination
            info["is_terminal"] = True
        else:
            info["is_terminal"] = False

        # XXX check effect on performance
        # --------------------------------
        if obs["option_time"].item() >= self.horizon:
            # for SB3
            info["TimeLimit.truncated"] = True
            info["terminal_observation"] = obs
        # else:
            # info["TimeLimit.truncated"] = False

        action_sim_time = time.time() - start_time
        info["action_sim_time"] = action_sim_time

        if self.cfg.reward.potential.enabled:
            curr_state = self.observe_true_state()
            v_prev = self.potential_fn([prev_state])[0]
            # because finite horizon
            v_curr = self.potential_fn([curr_state])[0] if not done else 0
            pot_reward = v_curr - v_prev
            rew += self.cfg.reward.potential.coeff * pot_reward
            # logger.debug(f"  Potential reward: {pot_reward}")
            # print(f"  Potential reward: {np.around(pot_reward, 2)}")
            # print("v_prev: ", v_prev, "v_curr: ", v_curr, "pot_reward: ", pot_reward)

        obs = self._get_observations(force_update=False)
        info["true_state"] = self.observe_true_state(mj_state=True)

        # logger.info(f"  reward: {rew}")
        return obs, rew, self.done, info

    def add_action(self, action, cost=None, term_action=True):
        """
        Args:
            term_action: if True, the episode terminates after calling this
            action.
        """
        self.actions.append(action)
        self.action_costs.append(cost)
        if term_action:
            self.term_actions.append(len(self.actions) - 1)

    def set_nominal_chains_for_eval(self, chains):
        """
        Used only to evaluate recovery policies.
        """
        logger.debug(f"  Setting nominal chains for evaluation: {chains}")

        self._nominal_chains = chains

    def get_term_action_ids(self):
        return copy.deepcopy(self.term_actions)

    @property
    def action_spec(self):
        return len(self.actions)

    @property
    def num_nominal_actions(self):
        return len(self.term_actions)

    @property
    def num_primitive_actions(self):
        return len(self.actions) - len(self.term_actions)

    def _compute_actions(self):
        """
        Compute motion primitives.
        """
        mprims_cfg = self.cfg.mprims
        trans_res = [mprims_cfg.resolution.trans]
        if mprims_cfg.enable_rot_actions:
            rot_res = [mprims_cfg.resolution.rot]
        else:
            rot_res = []

        if mprims_cfg.enable_short_mprims:
            trans_res.append(mprims_cfg.resolution_short.trans)
            if mprims_cfg.enable_rot_actions:
                rot_res.append(mprims_cfg.resolution_short.rot)

        mprims, costs = [], []
        # translation
        if isinstance(mprims_cfg.default_gripper_action, int):
            gripper_actions = [mprims_cfg.default_gripper_action]
        else:
            gripper_actions = mprims_cfg.default_gripper_action

        for gripper_action in gripper_actions:
            for res in trans_res:
                for i in range(3):
                    action = np.zeros(7)
                    action[-1] = gripper_action
                    action[i] = res
                    mprim = PrimitiveSkill(
                        action[:3], action[3:6], action[6], mprims_cfg.steps_per_action
                    )
                    mprims.append(mprim)
                    costs.append(1)

                    action = np.zeros(7)
                    action[-1] = gripper_action
                    action[i] = -res
                    mprim = PrimitiveSkill(
                        action[:3], action[3:6], action[6], mprims_cfg.steps_per_action
                    )
                    mprims.append(mprim)
                    costs.append(1)

            for res in rot_res:
                for i in range(3, 6):
                    action = np.zeros(7)
                    action[-1] = gripper_action
                    action[i] = res
                    mprim = PrimitiveSkill(
                        action[:3], action[3:6], action[6], mprims_cfg.steps_per_action
                    )
                    mprims.append(mprim)
                    costs.append(1)

                    action = np.zeros(7)
                    action[-1] = gripper_action
                    action[i] = -res
                    mprim = PrimitiveSkill(
                        action[:3], action[3:6], action[6], mprims_cfg.steps_per_action
                    )
                    mprims.append(mprim)
                    costs.append(1)

            if mprims_cfg.get("enable_stay_in_place_action", False):
                # stay in place
                action = np.zeros(7)
                action[-1] = mprims_cfg.default_gripper_action
                mprim = PrimitiveSkill(
                    action[:3], action[3:6], action[6], mprims_cfg.steps_per_action
                )
                mprims.append(mprim)
                costs.append(1)

        # gripper
        # close
        if mprims_cfg.enable_grip_actions:
            action = np.zeros(7)
            action[-1] = 1
            mprim = PrimitiveSkill(
                action[:3], action[3:6], action[6], mprims_cfg.steps_per_action
            )
            mprims.append(mprim)
            costs.append(1)
            # open
            action = np.zeros(7)
            action[-1] = -1
            mprim = PrimitiveSkill(
                action[:3], action[3:6], action[6], mprims_cfg.steps_per_action
            )
            mprims.append(mprim)
            costs.append(1)

        return mprims, costs

    # def _get_observations(self, **kwargs):
    # obs = self.env._get_observations(**kwargs)
    # return obs
    # return self._add_time(obs)
