class SkillChaining():
    def __init__(self, cfg, mode=None):
        self.cfg = cfg
        self.model = None

    def apply(self, obs, render=False):
        pass

    def train_policy(self, demos, train_env_fn, eval_env_fn):
        """
        Fit one-class classifiers to all the skills in the demos.

        Then use the classifiers as target preconditions.
        """
        pass

    def _train_preconditions(self, demos):
        # split states into skill segments

        # fit binary classifiers to each skill
