from abc import ABC, abstractmethod

from mindspeed_mm.fsdp.params.model_args import ModelArguments


class BaseModel(ABC):
    """
    Base Model Abstract Base Class
    All custom models should inherit from this class and implement the required methods.
    """

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        config: ModelArguments,
    ):
        """
        Load model from pretrained weights.
        
        Args:
            config: ModelArguments
        
        Returns:
            Loaded model instance
        """
        pass
    
    @classmethod
    @abstractmethod
    def _from_config(cls, config: ModelArguments) -> "BaseModel":
        """
        Create model instance from configuration without loading pretrained weights.
        Typically used for initialization with meta device or when starting from scratch.
        
        Args:
            config: ModelArguments
        
        Returns:
            Model instance initialized from configuration
        """
        pass