from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ModelAdapter(ABC):
    @abstractmethod
    def preprocess(self, messages: List[dict]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate(self, inputs: Dict[str, Any], generation_config) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode(self, inputs: Dict[str, Any], outputs: Any) -> str:
        raise NotImplementedError
