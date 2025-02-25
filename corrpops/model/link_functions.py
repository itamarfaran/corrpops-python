from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import numpy as np

from linalg.triangle_and_vector import vector_to_triangle, triangle_to_vector


class Transformer:
    def __init__(
        self,
        func: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        inverse_func: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        name: Optional[str] = None,
    ):
        self.func = func
        self.inverse_func = inverse_func
        self.name_ = name or func.__name__

    def transform(self, *args, **kwargs) -> Union[float, np.ndarray]:
        return self.func(*args, **kwargs)

    def inverse_transform(self, *args, **kwargs) -> Union[float, np.ndarray]:
        return self.inverse_func(*args, **kwargs)


class BaseLinkFunction(ABC):
    name_: str
    null_value_: float

    def __init__(self, transformer: Optional[Transformer] = None):
        self.transformer = transformer or Transformer(lambda x: x, lambda x: x)
        self._validate_transformer()

    def _validate_transformer(self):
        transformed = self.transformer.inverse_transform(self.null_value_)
        if not np.allclose(self.null_value_, self.transformer.transform(transformed)):
            raise ValueError(
                "self.transformer.inverse_transform does not "
                "correctly inverse self.transformer.transform"
            )

    @property
    def name(self) -> str:
        return (
            f"{self.name_}_{self.transformer.name_}"
            if self.transformer.name_
            else self.name_
        )

    @property
    def null_value(self) -> float:
        return self.transformer.inverse_transform(self.null_value_)

    def check_name_equal(self, name: str):
        if name != self.name:
            raise ValueError("link function mismatch")

    def __call__(self, t: np.ndarray, a: np.ndarray, d: int) -> np.ndarray:
        return self.forward(t=t, a=a, d=d)

    @abstractmethod
    def forward(self, t: np.ndarray, a: np.ndarray, d: int) -> np.ndarray:
        pass  # pragma: no cover

    @abstractmethod
    def inverse(self, data: np.ndarray, a: np.ndarray, d: int) -> np.ndarray:
        pass  # pragma: no cover


class MultiplicativeIdentity(BaseLinkFunction):
    name_ = "multiplicative_identity"
    null_value_ = 1.0

    def forward(self, t: np.ndarray, a: np.ndarray, d: int) -> np.ndarray:
        a = self.transformer.transform(a)
        a = a.reshape((int(a.size / d), d))
        a = a @ a.T
        a[np.diag_indices_from(a)] = 1
        return vector_to_triangle(t) * a

    def inverse(self, data: np.ndarray, a: np.ndarray, d: int) -> np.ndarray:
        a = self.transformer.transform(a)
        a = a.reshape((int(a.size / d), d))
        a = a @ a.T
        a = triangle_to_vector(a)
        return data / a


class AdditiveQuotient(BaseLinkFunction):
    name_ = "additive_quotient"
    null_value_ = 0.0

    def forward(self, t: np.ndarray, a: np.ndarray, d: int) -> np.ndarray:
        a = self.transformer.transform(a)
        a = a.reshape((int(a.size / d), d))
        a = a + a.T
        a[np.diag_indices_from(a)] = 0
        return vector_to_triangle(t) / (1 + a)

    def inverse(self, data: np.ndarray, a: np.ndarray, d: int) -> np.ndarray:
        a = self.transformer.transform(a)
        a = a.reshape((int(a.size / d), d))
        a = a + a.T
        a = triangle_to_vector(a)
        return data * (1 + a)
