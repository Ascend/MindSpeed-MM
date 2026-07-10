import torch
import torch.nn as nn
import torch_npu


class NpuFusedRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return torch_npu.npu_rms_norm(x.to(self.weight.dtype), self.weight, epsilon=self.eps)[0]
