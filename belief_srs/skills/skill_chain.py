import logging

from .robot_skill import RobotSkill

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class SkillChain(RobotSkill):
    def __init__(self, skills):
        super().__init__()
        self.skills = skills

    def _apply(self, env, obs, render=False):
        total_rew = 0
        gamma = 0.99
        interval = 0
        hist = {"state": [], "reward": [], "skill": []}
        for skill in self.skills:
            logger.debug(f"skill: {skill}")
            obs, rew, done, info = skill.apply(env, obs, render)
            hist['state'].extend(info['hist']['state'])
            hist["reward"].extend(info['hist']['reward'])
            hist['skill'].extend(info['hist']['skill'])

            discount = gamma**interval
            total_rew += discount * rew
            interval += info['interval']

            logger.debug(f"  interval: {interval},  discount: {discount}")
            logger.debug(f"    rew: {rew}, discounted_rew: {discount*rew}")
            if done:
                break
        info['hist'] = hist
        return obs, total_rew, done, info
