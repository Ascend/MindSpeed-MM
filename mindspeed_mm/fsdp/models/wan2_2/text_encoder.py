"""Wan2.2 text encoder component (UMT5/T5/MT5 wrapper); structure built by ``build_text_encoder``, weights via the unified ``GenerativeBaseModel.setup_weights`` path."""

import logging

import torch
import torch.nn as nn

from mindspeed_mm.fsdp.models.base_model import GenerativeBaseModel
from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.utils.dtype import get_dtype

logger = logging.getLogger(__name__)


class _SimpleTextEncoder(nn.Module, GenerativeBaseModel):
    """Minimal text encoder wrapper loading UMT5/T5/MT5 from transformers and providing encode()."""

    TRANSFORMERS_MAPPING = {
        "T5": "T5EncoderModel",
        "MT5": "MT5EncoderModel",
        "UMT5": "UMT5EncoderModel",
    }

    def __init__(
        self,
        model,
        use_attention_mask=True,
        output_key="last_hidden_state",
        hidden_state_skip_layer=None,
        ucg_rate=None,
        from_pretrained=None,
        load_format=None,
    ):
        super().__init__()
        self.model = model
        self.use_attention_mask = use_attention_mask
        self.output_key = output_key
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.ucg_rate = ucg_rate
        self._from_pretrained = from_pretrained
        self._load_format = load_format

    @classmethod
    def from_pretrained(cls, config):
        raise NotImplementedError(
            "_SimpleTextEncoder is constructed via build_text_encoder()."
        )

    @classmethod
    def _from_config(cls, config):
        raise NotImplementedError(
            "_SimpleTextEncoder is constructed via build_text_encoder()."
        )

    def get_weight_load_module(self):
        # The checkpoint holds the wrapped transformers model's keys, not the
        # wrapper's ``model.``-prefixed ones.
        return self.model

    def resolve_weight_source(self, config):
        from mindspeed_mm.fsdp.checkpoint.convert import resolve_checkpoint_source

        return resolve_checkpoint_source(
            config,
            fallback_from_pretrained=self._from_pretrained,
            fallback_format=self._load_format,
        )

    def post_load(self, loaded: bool):
        # UMT5/T5 checkpoints store the token embedding as ``shared.weight`` and
        # expect it to be tied to ``encoder.embed_tokens.weight``. Re-bind it so
        # the embedding is not left randomly initialised.
        if loaded and hasattr(self.model, "encoder") and hasattr(self.model, "shared"):
            self.model.encoder.embed_tokens.weight = self.model.shared.weight
        logger.info(
            "[WAN-TE] %s ready on %s (weights_loaded=%s)",
            type(self.model).__name__,
            next(self.model.parameters()).device,
            loaded,
        )

    def init_weights(self):
        # Guard for the trainer's meta-init stage: once pretrained weights are
        # loaded, re-initialization must be a no-op.  Only reachable on the
        # random-init fallback path, where we delegate to the wrapped model.
        if getattr(self, "_weights_loaded", False):
            return
        if hasattr(self.model, "init_weights"):
            self.model.init_weights()

    def encode(self, input_ids, attention_mask, **kwargs):
        compute_device = get_device_type()
        # The wrapped T5 may be kept on CPU to save NPU memory. Move inputs to the
        # model's device for the forward pass and bring the outputs back to the
        # compute device so the rest of the pipeline stays on the NPU.
        first_param = next(self.model.parameters())
        model_device = first_param.device
        *BN, L = input_ids.shape
        input_ids = input_ids.to(model_device).view(-1, L)
        attention_mask = attention_mask.to(model_device).view(-1, L)
        model_attention_mask = attention_mask if self.use_attention_mask else None

        output = self.model(
            input_ids=input_ids,
            attention_mask=model_attention_mask,
            output_hidden_states=self.hidden_state_skip_layer is not None,
        )

        emb = output[self.output_key]
        if self.hidden_state_skip_layer is not None:
            emb = emb[-(self.hidden_state_skip_layer + 1)]

        if self.ucg_rate is not None and self.ucg_rate > 0.0:
            def expand_dims_like(x, y):
                while x.dim() != y.dim():
                    x = x.unsqueeze(-1)
                return x
            emb = (
                expand_dims_like(
                    torch.bernoulli(
                        (1.0 - self.ucg_rate) * torch.ones(emb.shape[0], device=emb.device, dtype=emb.dtype)
                    ),
                    emb,
                )
                * emb
            )

        if self.output_key in ["last_hidden_state", "hidden_states"]:
            emb = emb.view(*BN, emb.shape[-2], -1)
        elif self.output_key in ["pooler_output", "text_embeds"]:
            emb = emb.view(*BN, -1)
        else:
            raise NotImplementedError(f"Text encoder output_key: {self.output_key} is not implemented!")

        attention_mask = attention_mask.view(*BN, -1)
        return emb.to(compute_device), attention_mask.to(compute_device)

    def forward(self, input_ids, attention_mask, **kwargs):
        """Expose encode() as the module forward so FSDP2 hooks fire."""
        return self.encode(input_ids, attention_mask, **kwargs)


def build_text_encoder(config: dict):
    """Build a ``_SimpleTextEncoder`` structure on host CPU; weights are deferred to the unified ``GenerativeBaseModel.setup_weights`` path."""
    from transformers import AutoConfig

    cfg = dict(config)  # shallow copy
    backend = cfg.pop("hub_backend", "hf")
    if backend not in ("hf", "huggingface"):
        raise ValueError(
            f"Wan2.2 text_encoder only supports HuggingFace loading, got hub_backend={backend}."
        )
    model_id = cfg.pop("model_id", "UMT5")
    if model_id not in _SimpleTextEncoder.TRANSFORMERS_MAPPING:
        raise ValueError(f"Model ID {model_id} is not supported for text encoder in pure FSDP2 mode")
    automodel_name = _SimpleTextEncoder.TRANSFORMERS_MAPPING[model_id]

    pretrained_path = cfg.pop("from_pretrained", None)
    if not pretrained_path:
        raise ValueError("Wan2.2 text_encoder requires 'from_pretrained' path.")
    torch_dtype = get_dtype(cfg.pop("dtype", "bf16"))

    logger.info("[WAN-TE] Loading %s config from %s", automodel_name, pretrained_path)
    transformer_config = AutoConfig.from_pretrained(pretrained_path)

    logger.info("[WAN-TE] Creating %s on CPU with dtype %s", automodel_name, torch_dtype)
    import transformers
    text_encoder = getattr(transformers, automodel_name)(transformer_config).to(torch_dtype)

    # Weight-source arbitration (load_format / checkpoint_format / hub_backend
    # aliases) is done by resolve_checkpoint_source at setup_weights time; the
    # raw values are only kept as fallbacks here.
    output_key = cfg.pop("output_key", "last_hidden_state")
    hidden_state_skip_layer = cfg.pop("hidden_state_skip_layer", None)
    ucg_rate = cfg.pop("ucg_rate", None)
    if hidden_state_skip_layer and output_key != "hidden_states":
        raise ValueError(
            f"hidden_state_skip_layer={hidden_state_skip_layer} requires output_key='hidden_states', "
            f"got {output_key!r}."
        )
    if ucg_rate is not None and not 0.0 < ucg_rate <= 1.0:
        raise ValueError(f"ucg_rate must be in (0, 1], got {ucg_rate}.")
    wrapper = _SimpleTextEncoder(
        model=text_encoder,
        use_attention_mask=cfg.pop("use_attention_mask", True),
        output_key=output_key,
        hidden_state_skip_layer=hidden_state_skip_layer,
        ucg_rate=ucg_rate,
        from_pretrained=pretrained_path,
        load_format=cfg.pop("load_format", None) or cfg.pop("checkpoint_format", None),
    )
    # By default the frozen text encoder is placed on the compute device along
    # with the other sub-models.  Set ``skip_to_device: true`` in the
    # text_encoder YAML section to keep it on CPU and move inputs/outputs
    # per-step instead (saves NPU memory at the cost of slower host-device
    # copies).
    skip_to_device = cfg.pop("skip_to_device", False)
    if isinstance(skip_to_device, str):
        skip_to_device = skip_to_device.strip().lower() in ("true", "1", "yes", "on")
    wrapper._ms_mm_skip_to_device = bool(skip_to_device)
    return wrapper
