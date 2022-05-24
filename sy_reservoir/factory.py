from .initializer import InitializerBase
from .activator import ActivatorBase
from .network import ReservoirNetwork

import numpy as np


class ReservoirCreator:
    state_initializer: InitializerBase
    input_weight_initializer: InitializerBase
    recurrent_weight_initializer: InitializerBase
    activator: ActivatorBase
    leak_rate_initializer: InitializerBase

    def __init__(
        self,
        state_initializer: InitializerBase,
        input_weight_initializer: InitializerBase,
        recurrent_weight_initializer: InitializerBase,
        activator: ActivatorBase,
        leak_rate_initializer: InitializerBase,
    ) -> None:
        self.state_initializer = state_initializer
        self.input_weight_initializer = input_weight_initializer
        self.recurrent_weight_initializer = recurrent_weight_initializer
        self.activator = activator
        self.leak_rate_initializer = leak_rate_initializer

    def create(self) -> ReservoirNetwork:
        return ReservoirNetwork(
            initial_state=self.state_initializer(),
            input_weight=self.input_weight_initializer(),
            recurrent_weight=self.recurrent_weight_initializer(),
            activator=self.activator,
            leak_rate=self.leak_rate_initializer(),
        )
