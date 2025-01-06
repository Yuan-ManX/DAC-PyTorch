import math
from typing import List
from typing import Union

import numpy as np
import torch
from torch import nn
from audiotools import AudioSignal
from audiotools.ml import BaseModel

from base import CodecMixin
from layers import Snake1d
from layers import WNConv1d
from layers import WNConvTranspose1d
from quantize import ResidualVectorQuantize


def init_weights(m):
    """
    初始化模型权重。

    该函数用于初始化模型的卷积层权重和偏置。

    参数:
        m (nn.Module): 要初始化的模型模块。
    """
    if isinstance(m, nn.Conv1d):
        # 使用截断正态分布初始化卷积层的权重，标准差为 0.02
        nn.init.trunc_normal_(m.weight, std=0.02)
        # 将偏置初始化为常数 0
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    """
    残差单元（ResidualUnit）。

    该模块实现了带有残差连接和膨胀卷积的残差单元。
    每个残差单元包含两个卷积层和一个残差连接。
    """
    def __init__(self, dim: int = 16, dilation: int = 1):
        """
        初始化残差单元。

        参数:
            dim (int, 可选): 通道维度，默认为 16。
            dilation (int, 可选): 膨胀因子，默认为 1。
        """
        super().__init__()
        # 计算填充量，以确保输入和输出具有相同的时间步长
        pad = ((7 - 1) * dilation) // 2
        # 定义残差单元的卷积块
        self.block = nn.Sequential(
            Snake1d(dim),  # 应用 Snake 激活函数
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),  # 应用权重归一化的 1D 卷积层
            Snake1d(dim),  # 再次应用 Snake 激活函数
            WNConv1d(dim, dim, kernel_size=1),  # 应用 1x1 卷积层
        )

    def forward(self, x):
        """
        前向传播函数，执行残差单元的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 残差单元的输出。
        """
        # 应用卷积块
        y = self.block(x)
        # 计算填充量，以确保输入和输出具有相同的时间步长
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            # 对输入进行裁剪
            x = x[..., pad:-pad]
        # 应用残差连接
        return x + y


class EncoderBlock(nn.Module):
    """
    编码器块（EncoderBlock）。

    该模块实现了编码器中的一个块，包含多个残差单元和一个下采样卷积层。
    """
    def __init__(self, dim: int = 16, stride: int = 1):
        """
        初始化编码器块。

        参数:
            dim (int, 可选): 通道维度，默认为 16。
            stride (int, 可选): 下采样步幅，默认为 1。
        """
        super().__init__()
        # 定义编码器块的卷积块
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),  # 第一个残差单元，膨胀因子为 1
            ResidualUnit(dim // 2, dilation=3),  # 第二个残差单元，膨胀因子为 3
            ResidualUnit(dim // 2, dilation=9),  # 第三个残差单元，膨胀因子为 9
            Snake1d(dim // 2),  # 应用 Snake 激活函数
            # 应用权重归一化的下采样卷积层
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        """
        前向传播函数，执行编码器块的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 编码器块的输出。
        """
        return self.block(x)


class Encoder(nn.Module):
    """
    编码器（Encoder）。

    该模块实现了音频信号的编码过程，将输入信号通过多个卷积层和残差单元逐步下采样，
    最终生成潜在空间表示。
    """
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        """
        初始化编码器。

        参数:
            d_model (int, 可选): 初始通道数，默认为 64。
            strides (list, 可选): 下采样步幅列表，默认为 [2, 4, 8, 8]。
            d_latent (int, 可选): 潜在空间维度，默认为 64。
        """
        super().__init__()
        # Create first convolution
        # 创建第一个卷积层，输入通道数为 1，输出通道数为 d_model
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        # 创建多个 EncoderBlock，每个块在下采样时通道数翻倍
        for stride in strides:
            d_model *= 2  # 通道数翻倍
            self.block += [EncoderBlock(d_model, stride=stride)]  # 添加 EncoderBlock

        # Create last convolution
        # 创建最后一个卷积层，将通道数转换为潜在空间维度
        self.block += [
            Snake1d(d_model),  # 应用 Snake 激活函数
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),  # 应用权重归一化的 1D 卷积层
        ]

        # Wrap black into nn.Sequential
        # 将所有层包装成 nn.Sequential
        self.block = nn.Sequential(*self.block)
        # 记录最终的通道数
        self.enc_dim = d_model

    def forward(self, x):
        """
        前向传播函数，执行编码器的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 编码器的输出。
        """
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    解码器块（DecoderBlock）。

    该模块实现了解码器中的一个块，包括上采样卷积层和多个残差单元。
    """
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        """
        初始化解码器块。

        参数:
            input_dim (int, 可选): 输入通道数，默认为 16。
            output_dim (int, 可选): 输出通道数，默认为 8。
            stride (int, 可选): 上采样步幅，默认为 1。
        """
        super().__init__()
        # 定义解码器块的卷积块
        self.block = nn.Sequential(
            Snake1d(input_dim), # 应用 Snake 激活函数
            # 应用权重归一化的转置 1D 卷积层
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),  # 应用残差单元，膨胀因子为 1
            ResidualUnit(output_dim, dilation=3),  # 应用残差单元，膨胀因子为 3
            ResidualUnit(output_dim, dilation=9),  # 应用残差单元，膨胀因子为 9
        )

    def forward(self, x):
        """
        前向传播函数，执行解码器块的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 解码器块的输出。
        """
        return self.block(x)


class Decoder(nn.Module):
    """
    解码器（Decoder）。

    该模块实现了音频信号解码过程，将潜在空间表示逐步上采样并转换为原始信号。
    """
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        """
        初始化解码器。

        参数:
            input_channel (int): 输入通道数。
            channels (int): 通道数。
            rates (list): 上采样步幅列表。
            d_out (int, 可选): 输出通道数，默认为 1。
        """
        super().__init__()

        # Add first conv layer
        # 添加第一个卷积层
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        # 添加上采样和 MRF 块
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i  # 计算输入通道数
            output_dim = channels // 2 ** (i + 1)  # 计算输出通道数
            layers += [DecoderBlock(input_dim, output_dim, stride)]  # 添加 DecoderBlock

        # Add final conv layer
        # 添加最后一个卷积层
        layers += [
            Snake1d(output_dim),  # 应用 Snake 激活函数
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),  # 应用权重归一化的 1D 卷积层
            nn.Tanh(),  # 应用 Tanh 激活函数
        ]

        # 将所有层包装成 nn.Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数，执行解码器的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 解码器的输出。
        """
        return self.model(x)


class DAC(BaseModel, CodecMixin):
    """
    离散音频编解码器（Discrete Audio Codec, DAC）。

    该类实现了音频信号的编码和解码过程，包括卷积编码器、向量量化器和解码器。
    """
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
    ):
        """
        初始化 DAC 模型。

        参数:
            encoder_dim (int, 可选): 编码器初始通道数，默认为 64。
            encoder_rates (List[int], 可选): 编码器下采样步幅列表，默认为 [2, 4, 8, 8]。
            latent_dim (int, 可选): 潜在空间维度，如果为 None，则根据编码器参数自动计算。
            decoder_dim (int, 可选): 解码器初始通道数，默认为 1536。
            decoder_rates (List[int], 可选): 解码器上采样步幅列表，默认为 [8, 8, 4, 2]。
            n_codebooks (int, 可选): 向量量化器的码本数量，默认为 9。
            codebook_size (int, 可选): 每个码本的大小，默认为 1024。
            codebook_dim (Union[int, List[int]], 可选): 码本的维度，如果为列表，则每个码本有不同的维度，默认为 8。
            quantizer_dropout (bool, 可选): 是否使用量化器 dropout，默认为 False。
            sample_rate (int, 可选): 采样率，默认为 44100 Hz。
        """
        super().__init__()

        # 记录编码器初始通道数
        self.encoder_dim = encoder_dim
        # 记录编码器下采样步幅列表
        self.encoder_rates = encoder_rates
        # 记录解码器初始通道数
        self.decoder_dim = decoder_dim
        # 记录解码器上采样步幅列表
        self.decoder_rates = decoder_rates
        # 记录采样率
        self.sample_rate = sample_rate

        if latent_dim is None:
            # 如果潜在空间维度未指定，则根据编码器下采样步幅自动计算
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        # 记录潜在空间维度
        self.latent_dim = latent_dim

        # 计算跳步长度，为编码器下采样步幅的乘积
        self.hop_length = np.prod(encoder_rates)
        # 初始化编码器
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        # 记录码本数量
        self.n_codebooks = n_codebooks
        # 记录每个码本的大小
        self.codebook_size = codebook_size
        # 记录码本的维度
        self.codebook_dim = codebook_dim
        # 初始化向量量化器
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,  # 输入维度
            n_codebooks=n_codebooks,  # 码本数量
            codebook_size=codebook_size,  # 每个码本的大小
            codebook_dim=codebook_dim,  # 码本的维度
            quantizer_dropout=quantizer_dropout,  # 是否使用量化器 dropout
        )

        # 初始化解码器
        self.decoder = Decoder(
            latent_dim,  # 输入潜在空间维度
            decoder_dim,  # 解码器初始通道数
            decoder_rates,  # 解码器上采样步幅列表
        )
        # 记录采样率
        self.sample_rate = sample_rate
        # 应用初始化权重函数
        self.apply(init_weights)
        # 计算延迟
        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        """
        预处理音频数据。

        参数:
            audio_data (torch.Tensor): 输入音频数据。
            sample_rate (int, 可选): 音频数据的采样率。如果为 None，则使用模型采样率。

        返回:
            torch.Tensor: 预处理后的音频数据。
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        # 获取音频数据的长度
        length = audio_data.shape[-1]
        # 计算右侧填充量
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        # 对音频数据进行填充
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        # 返回预处理后的音频数据
        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
    ):
        """
        编码给定的音频数据并返回量化的潜在编码。

        参数:
            audio_data (Tensor[B x 1 x T]): 要编码的音频数据。
            n_quantizers (int, 可选): 要使用的量化器数量，默认为 None。
                                       如果为 None，则使用所有量化器。

        返回:
            dict: 一个包含以下键的字典:
                "z" (Tensor[B x D x T]): 输入的量化连续表示。
                "codes" (Tensor[B x N x T]): 每个码本的码本索引
                                             (输入的量化离散表示)。
                "latents" (Tensor[B x N*D x T]): 投影后的潜在变量
                                                (量化前的连续表示)。
                "vq/commitment_loss" (Tensor[1]): 训练编码器以预测更接近码本条目的向量的承诺损失。
                "vq/codebook_loss" (Tensor[1]): 更新码本的码本损失。
                "length" (int): 输入音频中的样本数量。
        """
        # 对音频数据进行编码
        z = self.encoder(audio_data)
        # 对编码后的数据进行量化
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )
        # 返回编码结果
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        """
        解码给定的潜在编码并返回音频数据。

        参数:
            z (Tensor[B x D x T]): 输入的量化连续表示。

        返回:
            dict: 一个包含以下键的字典:
                "audio" (Tensor[B x 1 x length]): 解码后的音频数据。
        """
        # 对潜在编码进行解码
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """
        模型前向传播。

        参数:
            audio_data (Tensor[B x 1 x T]): 要编码的音频数据。
            sample_rate (int, 可选): 音频数据的采样率（Hz），默认为 None。
                                      如果为 None，则默认为 `self.sample_rate`。
            n_quantizers (int, 可选): 要使用的量化器数量，默认为 None。
                                       如果为 None，则使用所有量化器。

        返回:
            dict: 一个包含以下键的字典:
                "z" (Tensor[B x D x T]): 输入的量化连续表示。
                "codes" (Tensor[B x N x T]): 每个码本的码本索引
                                             (输入的量化离散表示)。
                "latents" (Tensor[B x N*D x T]): 投影后的潜在变量
                                                (量化前的连续表示)。
                "vq/commitment_loss" (Tensor[1]): 训练编码器以预测更接近码本条目的向量的承诺损失。
                "vq/codebook_loss" (Tensor[1]): 更新码本的码本损失。
                "length" (int): 输入音频中的样本数量。
                "audio" (Tensor[B x 1 x length]): 解码后的音频数据。
        """
        # 获取音频数据的长度
        length = audio_data.shape[-1]
        # 对音频数据进行预处理
        audio_data = self.preprocess(audio_data, sample_rate)
        # 对音频数据进行编码
        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )

        # 对编码后的数据进行解码
        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }


if __name__ == "__main__":

    import numpy as np
    from functools import partial

    # 实例化 DAC 模型，并将其移动到 CPU
    model = DAC().to("cpu")

    # 遍历模型中的所有子模块
    for n, m in model.named_modules():
        o = m.extra_repr()
        # 计算子模块中所有参数的总数量
        p = sum([np.prod(p.size()) for p in m.parameters()])
        # 定义一个 lambda 函数，用于格式化参数数量（以百万为单位）
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    # 定义音频信号的长度（这里假设为 88200 * 2 个样本）
    length = 88200 * 2
    # 生成一个形状为 (1, 1, length) 的随机张量作为输入音频数据，并将其移动到模型的设备上
    x = torch.randn(1, 1, length).to(model.device)
    # 设置输入张量需要梯度，并保留梯度信息
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    # 进行前向传播
    out = model(x)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    # 创建一个与输出张量形状相同的梯度张量，并将中间位置的值设为 1
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    # 进行后向传播，计算输入张量的梯度
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    # 创建一个长度为 60 秒的随机音频信号，采样率为 44100 Hz
    x = AudioSignal(torch.randn(1, 1, 44100 * 60), 44100)
    # 对音频信号进行压缩和解压缩，并显示进度信息
    model.decompress(model.compress(x, verbose=True), verbose=True)
