import dataclasses
import logging
from typing import Dict, cast

from d3rlpy.base import DeviceArg, LearnableConfig, register_learnable
from d3rlpy.constants import ActionSpace
from d3rlpy.models.builders import create_discrete_q_function
from d3rlpy.models.encoders import EncoderFactory, make_encoder_field
from d3rlpy.models.optimizers import OptimizerFactory, make_optimizer_field
from d3rlpy.models.q_functions import QFunctionFactory, make_q_func_field
from d3rlpy.types import Shape
from d3rlpy.algos.qlearning.base import QLearningAlgoBase
from d3rlpy.algos.qlearning.torch.dqn_impl import DQNLoss, DQNModules
from d3rlpy.models.torch import DiscreteEnsembleQFunctionForwarder
from d3rlpy.torch_utility import Modules, TorchMiniBatch, convert_to_torch, convert_to_torch_recursively
from d3rlpy.dataclass_utils import asdict_as_float
from d3rlpy.algos.qlearning.base import QLearningAlgoImplBase
from d3rlpy.algos.qlearning.torch.utility import DiscreteQFunctionMixin
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import check_non_1d_array
import torch
from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class QEvaluatorImpl(DiscreteQFunctionMixin, QLearningAlgoImplBase):
    _modules: DQNModules
    _gamma: float
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DQNModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        gamma: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._gamma = gamma
        self._q_func_forwarder = q_func_forwarder

    def predict_value_ensemble(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict action-values for each ensemble.
        """
        values = self._q_func_forwarder.compute_expected_q(x, reduction="none")
        flat_action = action.reshape(-1)
        return values[:, torch.arange(0, x.size(0)), flat_action]

    def inner_update(

        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        self._modules.optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_loss(batch, q_tpn)

        loss.loss.backward()
        self._modules.optim.step()

        return asdict_as_float(loss)

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> DQNLoss:
        loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )
        return DQNLoss(loss=loss)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            # SARSA would require to use next_actions

            # DQN uses argmax Q(s', a')
            # next_actions = self._targ_q_func_forwarder.compute_expected_q(
                # batch.next_observations
            # )
            # max_action = next_actions.argmax(dim=1)

            # I use monte carlo
            logger.debug(f"  returns_to_go: {batch.returns_to_go}")
            return batch.returns_to_go

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._q_func_forwarder.compute_expected_q(x).argmax(dim=1)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def update_target(self) -> None:
        hard_sync(self._modules.targ_q_funcs, self._modules.q_funcs)

    @property
    def q_function(self) -> nn.ModuleList:
        return self._modules.q_funcs

    @property
    def q_function_optim(self) -> Optimizer:
        return self._modules.optim


@dataclasses.dataclass()
class QEvaluatorConfig(LearnableConfig):
    r"""Config of Deep offpolicy Q evaluation
    """
    batch_size: int = 32
    learning_rate: float = 6.25e-5
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    gamma: float = 0.99
    n_critics: int = 1

    def create(self, device: DeviceArg = False):
        return QEvaluator(self, device)

    @staticmethod
    def get_type() -> str:
        return "q_eval"

class QEvaluator(QLearningAlgoBase[QEvaluatorImpl, QEvaluatorConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        q_funcs, forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        # targ_q_funcs, targ_forwarder = create_discrete_q_function(
            # observation_shape,
            # action_size,
            # self._config.encoder_factory,
            # self._config.q_func_factory,
            # n_ensembles=self._config.n_critics,
            # device=self._device,
        # )

        optim = self._config.optim_factory.create(
            q_funcs.named_modules(), lr=self._config.learning_rate
        )

        modules = DQNModules(
            q_funcs=q_funcs,
            targ_q_funcs=None,
            optim=optim,
        )

        self._impl = QEvaluatorImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            q_func_forwarder=forwarder,
            modules=modules,
            gamma=self._config.gamma,
            device=self._device,
        )

    def predict_value_ensemble(self, x, action):
        """Returns predicted ensemble action-values

        Args:
            x: Observations
            action: Actions

        Returns:
            Predicted action-values
            Standard deviations of Q values
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        # TODO: support tuple inputs
        torch_x = cast(
            torch.Tensor, convert_to_torch_recursively(x, self._device)
        )

        torch_action = convert_to_torch(action, self._device)

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            if self.get_action_type() == ActionSpace.CONTINUOUS:
                if self._config.action_scaler:
                    torch_action = self._config.action_scaler.transform(
                        torch_action
                    )
            elif self.get_action_type() == ActionSpace.DISCRETE:
                torch_action = torch_action.long()
            else:
                raise ValueError("invalid action type")

            value = self._impl.predict_value_ensemble(torch_x, torch_action)

        return value.cpu().detach().numpy()  # type: ignore

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


