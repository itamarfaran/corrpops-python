from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from triangle_vector import vector_to_triangle, triangle_to_vector


class Transformer:
    def __init__(self, func: Callable, inv: Callable):
        self.func = func
        self.inv = inv

    def transform(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def inv_transform(self, *args, **kwargs):
        return self.inv(*args, **kwargs)


def _identity(x):
    return x


class BaseLinkFunction(ABC):
    name: str = ""
    null_value: float

    def __init__(self, transformer=Transformer(_identity, _identity)):
        self.transformer = transformer

    @abstractmethod
    def func(self, t, a, d):
        pass

    @abstractmethod
    def reverse(self, data, a, d):
        pass


class MultiplicativeIdentity(BaseLinkFunction):
    name = "multiplicative_identity"
    null_value = 1.0

    def func(self, t, a, d):
        a = self.transformer.transform(a)
        a = np.outer(a, a)
        a[np.diag_indices_from(a)] = 1
        return vector_to_triangle(t, diag_value=1) * a

    def reverse(self, data, a, d):
        a = self.transformer.transform(a)
        a = np.outer(a, a)
        a = triangle_to_vector(a)
        return data / np.outer(len(data), a)


class AdditiveQuotent(BaseLinkFunction):
    name = "additive_quotent"
    null_value = 0.0

    def func(self, t, a, d):
        a = self.transformer.transform(a)
        a = np.tile(a, d).reshape((d, d))
        a = a + a.T
        a[np.diag_indices_from(a)] = 0
        return vector_to_triangle(t, diag_value=1) / (1 + a)

    def reverse(self, data, a, d):
        a = self.transformer.transform(a)
        a = np.tile(a, d).reshape((d, d))
        a = a + a.T
        a = triangle_to_vector(a)
        return data * (1 + np.outer(len(data), a))

