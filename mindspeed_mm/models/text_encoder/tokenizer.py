import importlib
from torch import nn


class Tokenizer:
    """
    Configuration for initializing one or more Tokenizer instances.

    Args:
        config (Optional[dict, list(dict)]): the general config for Tokenizer
        - If `config` is a dictionary, it specifies the parameters for a single Tokenizer instance.
            e.g.
            {
                (1) args for our feautrues
                "backend": type-str, "hf" or "om",
                "autotokenizer_name": type-str, "AutoTokenizer" or other autotokenizer name,

                (2) args for autotokenizer.from_pretrained() of transformers or openmind
                "pretrained_model_name_or_path": type-str, local path or hub path,
                "local_files_only": type-bool,
                ...
            }
        - If `config` is a list of dictionaries, each dictionary in the list will be used to instantiate a separate Tokenizer instance, 
            effectively allowing the creation of multiple tokenizers based on different configurations.
    """

    def __init__(self, config):
        module = importlib.import_module("transformers")
        if isinstance(config, list):
            self.tokenizers = []
            for config_i in config:
                tokenizer_i = self._init_tokenizer(module, config_i)
                self.tokenizers.append(tokenizer_i)
        else:
            self.tokenizers = self._init_tokenizer(module, config)           

    def get_tokenizer(self):
        return self.tokenizers
    
    def _init_tokenizer(self, module, config):
        if not isinstance(config, dict):
            config = config.to_dict()

        # Only huggingface backend is supported currently.
        self.backend = config.pop("hub_backend")
        tokenizer_name = config.pop("autotokenizer_name")
        config["pretrained_model_name_or_path"] = config.pop("from_pretrained")
        tokenizer_cls = getattr(module, tokenizer_name)
        return tokenizer_cls.from_pretrained(**config)