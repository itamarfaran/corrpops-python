from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from linalg.triangle_vector import vector_to_triangle, triangle_to_vector


class Transformer:
    def __init__(
        self,
        func: Callable,
        inverse_func: Callable,
        name: str = None,
    ):
        self.func = func
        self.inverse_func = inverse_func
        self.name_ = name or func.__name__

    def transform(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def inverse_transform(self, *args, **kwargs):
        return self.inverse_func(*args, **kwargs)


class BaseLinkFunction(ABC):
    name_: str
    null_value_: float

    def __init__(self, transformer: Transformer = None):
        self.transformer = transformer or Transformer(lambda x: x, lambda x: x, "")
        self._validate_transformer()

    def _validate_transformer(self):
        transformed = self.transformer.inverse_transform(self.null_value_)
        if not np.allclose(self.null_value_, self.transformer.transform(transformed)):
            raise ValueError(
                "self.transformer.inverse_transform does not "
                "correctly inverse self.transformer.transform"
            )

    @property
    def name(self):
        return (
            f"{self.name_}_{self.transformer.name_}"
            if self.transformer.name_
            else self.name_
        )

    @property
    def null_value(self):
        return self.transformer.inverse_transform(self.null_value_)

    def check_name_equal(self, name: str):
        if name != self.name:
            raise ValueError("link function mismatch")

    def __call__(self, *, t, a, d):
        return self.forward(t=t, a=a, d=d)

    @abstractmethod
    def forward(self, *, t, a, d):
        pass

    @abstractmethod
    def inverse(self, *, data, a, d):
        pass


class MultiplicativeIdentity(BaseLinkFunction):
    name_ = "multiplicative_identity"
    null_value_ = 1.0

    def forward(self, *, t, a, d):
        a = self.transformer.transform(a)
        a = a.reshape((int(a.size / d), d))
        a = a @ a.T
        a[np.diag_indices_from(a)] = 1
        return vector_to_triangle(t, diag_value=1) * a

    def inverse(self, *, data, a, d):
        a = self.transformer.transform(a)
        a = a.reshape((int(a.size / d), d))
        a = a @ a.T
        a = triangle_to_vector(a)
        return data / a


class AdditiveQuotient(BaseLinkFunction):
    name_ = "additive_quotient"
    null_value_ = 0.0

    def forward(self, *, t, a, d):
        a = self.transformer.transform(a)
        a = a.reshape((int(a.size / d), d))
        a = a + a.T
        a[np.diag_indices_from(a)] = 0
        return vector_to_triangle(t, diag_value=1) / (1 + a)

    def inverse(self, *, data, a, d):
        a = self.transformer.transform(a)
        a = a.reshape((int(a.size / d), d))
        a = a + a.T
        a = triangle_to_vector(a)
        return data * (1 + a)
