"""Base abstractions for generative-model training pipelines: pure orchestrators (not nn.Module) wiring container sub-models (ae / text_encoder / predictor) for one training step, so FSDP pattern matching is unaffected."""

from dataclasses import dataclass

import torch


@dataclass
class GenerativePipelineOutput:
    """Default pipeline output; the TrainEngine contract only needs ``.loss``."""

    loss: torch.Tensor


def grad_context_for(module):
    """``enable_grad`` when the module holds trainable parameters, else ``no_grad``.

    Lets condition-encoding stages backprop into a sub-model only when it is
    actually trainable, keeping frozen encoders gradient-free.
    """
    trainable = module is not None and any(p.requires_grad for p in module.parameters())
    return torch.enable_grad() if trainable else torch.no_grad()


class GenerativePipelineBase:
    """Stage interface for generative training data flow: ``prepare_conditions`` (grad-aware encoding) -> ``noising`` -> ``predict`` (grad) -> ``compute_loss``; ``forward(container, **batch)`` orchestrates the stages."""

    def prepare_conditions(self, container, **batch) -> dict:
        """Encode conditions without gradients; returns a conditions dict."""
        raise NotImplementedError

    def noising(self, container, conditions, **batch) -> dict:
        """Noise the latents; returns the inputs dict for ``predict``."""
        raise NotImplementedError

    def predict(self, container, **inputs):
        """Run the predictor forward pass (with gradients)."""
        raise NotImplementedError

    def compute_loss(self, container, conditions, inputs, model_output, **batch):
        """Compute the training loss; returns an object with ``.loss``."""
        raise NotImplementedError

    def forward(self, container, **batch):
        conditions = self.prepare_conditions(container, **batch)
        inputs = self.noising(container, conditions, **batch)
        model_output = self.predict(container, **inputs)
        return self.compute_loss(container, conditions, inputs, model_output, **batch)
