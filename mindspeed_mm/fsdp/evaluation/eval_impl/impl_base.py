from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    @abstractmethod
    def update(self, item: dict[str, Any], prediction: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> dict[str, Any]:
        raise NotImplementedError
