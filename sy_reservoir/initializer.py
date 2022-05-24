from abc import ABCMeta, abstractmethod
from typing import Sequence
import numpy as np
from numpy.typing import NDArray


class InitializerBase(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self):
        ...


class ConstantInitializer(InitializerBase):
    _constant: NDArray

    def __init__(self, constant: NDArray) -> None:
        self._constant = constant

    @property
    def constant(self) -> NDArray:
        return self._constant

    def __call__(self) -> NDArray:
        return self.constant


class UniformInitializer(InitializerBase):
    _low: float
    _high: float
    _size: Sequence[int]
    _rng: np.random.Generator

    def __init__(self, low: float, high: float, size: Sequence[int], seed=None) -> None:
        self._rng = np.random.default_rng()
        self._low = low
        self._high = high
        self._size = size
        self._seed = seed

    def __call__(self):
        return self._rng.uniform(low=self.low, high=self.high, size=self.size)

    @property
    def low(self):
        return self.low

    @property
    def high(self):
        return self._high

    @property
    def size(self):
        return self._size


class ZeroInitializer(InitializerBase):
    _size: Sequence[int]

    def __init__(self, size: Sequence[int]) -> None:
        self._size = size

    def __call__(self):
        return np.zeros(shape=self._size)

    @property
    def size(self):
        return self._size


class NormalInitializer(InitializerBase):
    _size: Sequence[int]
    _loc: float
    _scale: float
    _rng: np.random.Generator

    def __init__(self, loc: float, scale: float, size: Sequence[int]) -> None:
        self._rng = np.random.default_rng()
        self._loc = loc
        self._scale = scale
        self._size = size

    def __call__(self):
        return self._rng.normal(self._loc, self._scale, self._size)

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def size(self):
        return self._size
