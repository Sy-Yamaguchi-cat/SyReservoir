from .activator import ActivatorBase

import numpy as np
from numpy.typing import NDArray
from typing import Union


class ReservoirNetwork:
    def __init__(
        self,
        initial_state: NDArray,
        input_weight: NDArray,
        recurrent_weight: NDArray,
        activator: ActivatorBase,
        leak_rate: Union[int, NDArray],
    ):
        self._initial_state = initial_state
        self._state = np.copy(initial_state)
        self.input_weight = input_weight
        self.recurrent_weight = recurrent_weight
        self.activator = activator
        self.leak_rate = leak_rate

    def __call__(self, u):
        self._state = (
            1 - self.leak_rate
        ) * self._state + self.leak_rate * self.activator(
            self.input_weight @ u + self.recurrent_weight @ self._state
        )
        return self._state

    input_weight: NDArray
    rucurrent_weight: NDArray
    activator: ActivatorBase
    leak_rate: Union[int, NDArray]

    @property
    def state(self) -> NDArray:
        return self._state

    @property
    def initial_state(self) -> NDArray:
        return self._initial_state

    def initialize(self):
        self._state = np.copy(self._initial_state)

    def run(self, inputs: Union[list[NDArray], NDArray]):
        outputs = np.zeros(shape=(len(inputs), *self._state.shape))
        for i, u in enumerate(inputs):
            outputs[i] = self(u)
        return outputs
