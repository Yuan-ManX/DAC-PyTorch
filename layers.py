import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange


def WNConv1d(*args, **kwargs):
    """
    权重归一化的 1D 卷积层（Weight Normalized Conv1d）。

    该函数应用权重归一化（Weight Normalization）到标准的 1D 卷积层，以稳定训练过程并加速收敛。

    参数:
        *args: 传递给 nn.Conv1d 的位置参数。
        **kwargs: 传递给 nn.Conv1d 的关键字参数。

    返回:
        nn.Module: 应用了权重归一化的 1D 卷积层。
    """
    # 应用权重归一化到 1D 卷积层
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """
    权重归一化的转置 1D 卷积层（Weight Normalized ConvTranspose1d）。

    该函数应用权重归一化（Weight Normalization）到标准的转置 1D 卷积层，以稳定训练过程并加速收敛。

    参数:
        *args: 传递给 nn.ConvTranspose1d 的位置参数。
        **kwargs: 传递给 nn.ConvTranspose1d 的关键字参数。

    返回:
        nn.Module: 应用了权重归一化的转置 1D 卷积层。
    """
    # 应用权重归一化到转置 1D 卷积层
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    """
    Snake 激活函数。

    Snake 是一种平滑且非线性的激活函数，定义为 x + (1 / alpha) * sin^2(alpha * x)。
    这种激活函数在保持平滑性的同时，允许模型学习更复杂的模式。

    参数:
        x (torch.Tensor): 输入张量。
        alpha (torch.Tensor): 控制函数形状的参数张量。

    返回:
        torch.Tensor: 经过 Snake 激活函数处理后的张量。
    """
    # 获取输入张量的形状
    shape = x.shape
    # 重塑张量为 (batch_size, channels, -1)
    x = x.reshape(shape[0], shape[1], -1)
    # 应用 Snake 激活函数
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    # 重塑回原始形状
    x = x.reshape(shape)
    # 返回激活后的张量
    return x


class Snake1d(nn.Module):
    """
    Snake 激活函数的 1D 实现（Snake1d）。

    该模块实现了 Snake 激活函数，并将其应用于输入张量的每个通道。
    """
    def __init__(self, channels):
        """
        初始化 Snake1d 模块。

        参数:
            channels (int): 输入张量的通道数。
        """
        super().__init__()
        # 定义参数 alpha，形状为 (1, channels, 1)
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        """
        前向传播函数，执行 Snake 激活函数的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过 Snake 激活函数处理后的张量。
        """
        return snake(x, self.alpha)
