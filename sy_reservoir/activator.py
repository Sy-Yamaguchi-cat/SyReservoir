from abc import abstractmethod, ABCMeta
import numpy as np


class ActivatorBase(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, s):
        ...


class TanhActivator(ActivatorBase):
    _coefficient: float

    def __init__(self, coefficient: float = 1.0) -> None:
        self._coefficient = coefficient

    def __call__(self, s):
        return np.tanh(self._coefficient * s)

    @property
    def coefficient(self) -> float:
        return self._coefficient
