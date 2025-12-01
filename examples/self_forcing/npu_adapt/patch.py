import torch
import torch_npu
import torch.nn as nn
from wan.modules import causal_model, model


def global_function_replacement(original_module, function_name):
    def decorator(new_function):
        setattr(original_module, function_name, new_function)
        return new_function
    return decorator


@global_function_replacement(causal_model, 'causal_rope_apply')
def npu_causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1).to(torch.complex64),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1).to(torch.complex64),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1).to(torch.complex64)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


@global_function_replacement(model, 'rope_apply')
def npu_rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1).to(torch.complex64),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1).to(torch.complex64),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1).to(torch.complex64)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


@global_function_replacement(model, 'rope_params')
def npu_rope_params(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float32).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@global_function_replacement(model, 'WanRMSNorm')
class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
