import logging
import numpy as np
from sklearn import neighbors, preprocessing

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class DensityBasedReward:
    """
    Implements a density-based reward function that is learned from expert
    demonstrations.
    """

    def __init__(self, expert_states, kernel_bandwidth="scott"):
        self.kernel = "gaussian"
        self.kernel_bandwidth = kernel_bandwidth
        self._train(expert_states)

    def __call__(self, states):
        """
        Compute reward from given state.
        """
        X = np.array(states)
        X = self._scaler.transform(X)
        rews = self._density_model.score_samples(X)
        return rews

    def _train(self, expert_states):
        self._scaler = preprocessing.StandardScaler()
        if self.kernel_bandwidth == "classification":
            pass
        elif self.kernel_bandwidth == "auto":
            pos_expert_states = expert_states["pos"]
            neg_expert_states = expert_states["neg"]
            all_states = np.concatenate([pos_expert_states, neg_expert_states])
            self._scaler.fit(all_states)
            pos_expert_states = self._scaler.transform(pos_expert_states)
            neg_expert_states = self._scaler.transform(neg_expert_states)
            bandwidths = np.linspace(0.01, 1, 10)
            ratios = []
            for bandwidth in bandwidths:
                model = self._fit_density(pos_expert_states, bandwidth)
                pos_loglikelihood = model.score(pos_expert_states)
                neg_loglikelihood = model.score(neg_expert_states)
                lik_ratio = pos_loglikelihood - neg_loglikelihood
                ratios.append(lik_ratio)
                logger.debug(f"  bandwidth: {bandwidth}, likelihood ratio: {lik_ratio}")
            print("ratios: ", ratios)

        else:
            all_states = expert_states["pos"]
            self._scaler.fit(all_states)
            expert_states = self._scaler.transform(all_states)
            self._density_model = self._fit_density(all_states, self.kernel_bandwidth)

    def _fit_density(self, transitions, bandwidth):
        density_model = neighbors.KernelDensity(
            kernel=self.kernel,
            bandwidth=bandwidth,
        )
        density_model.fit(transitions)
        return density_model
