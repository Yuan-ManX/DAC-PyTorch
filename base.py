import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn
import tqdm
from audiotools import AudioSignal


SUPPORTED_VERSIONS = ["1.0.0"]


@dataclass
class DACFile:
    """
    DAC 文件类，用于处理 `.dac` 文件。

    该类包含音频编码数据及其元数据，并提供保存和加载 `.dac` 文件的方法。
    """
    codes: torch.Tensor

    # Metadata
    chunk_length: int
    original_length: int
    input_db: float
    channels: int
    sample_rate: int
    padding: bool
    dac_version: str

    def save(self, path):
        """
        将 DACFile 对象保存为 `.dac` 文件。

        参数:
            path (str): 保存文件的路径。

        返回:
            str: 保存后的文件路径。
        """
        # 构建要保存的工件字典，包括编码数据和元数据
        artifacts = {
            "codes": self.codes.numpy().astype(np.uint16), # 将编码数据转换为 uint16 类型的 numpy 数组
            "metadata": {
                "input_db": self.input_db.numpy().astype(np.float32), # 将音量数据转换为 float32 类型的 numpy 数组
                "original_length": self.original_length,
                "sample_rate": self.sample_rate,
                "chunk_length": self.chunk_length,
                "channels": self.channels,
                "padding": self.padding,
                "dac_version": SUPPORTED_VERSIONS[-1],
            },
        }
        path = Path(path).with_suffix(".dac")
        with open(path, "wb") as f:
            # 使用 numpy 的保存功能将工件保存到文件中
            np.save(f, artifacts)
        return path

    @classmethod
    def load(cls, path):
        """
        从 `.dac` 文件加载 DACFile 对象。

        参数:
            path (str): 要加载的文件路径。

        返回:
            DACFile: 加载的 DACFile 对象。

        异常:
            RuntimeError: 如果文件的 DAC 版本不被支持。
        """
        # 从文件中加载工件，允许使用 pickle
        artifacts = np.load(path, allow_pickle=True)[()]
        # 将编码数据从 numpy 数组转换为 torch 张量，并转换为整数类型
        codes = torch.from_numpy(artifacts["codes"].astype(int))
        if artifacts["metadata"].get("dac_version", None) not in SUPPORTED_VERSIONS:
            raise RuntimeError(
                f"Given file {path} can't be loaded with this version of descript-audio-codec."
            )
        return cls(codes=codes, **artifacts["metadata"])


class CodecMixin:
    """
    编解码器混合类（CodecMixin）。

    该类为卷积编码器和解码器提供了混合功能，包括处理填充、计算延迟和输出长度等。
    """
    @property
    def padding(self):
        """
        获取填充属性。

        如果没有设置 _padding 属性，则默认设置为 True。

        返回:
            bool: 填充标志。
        """
        if not hasattr(self, "_padding"):
            self._padding = True
        return self._padding

    @padding.setter
    def padding(self, value):
        """
        设置填充属性。

        参数:
            value (bool): 填充标志。

        异常:
            AssertionError: 如果 value 不是布尔类型，则抛出断言错误。
        """
        assert isinstance(value, bool)

        # 查找所有卷积层（Conv1d 和 ConvTranspose1d）
        layers = [
            l for l in self.modules() if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d))
        ]

        for layer in layers:
            if value:
                # 如果启用填充，并且层有 original_padding 属性，则恢复原始填充
                if hasattr(layer, "original_padding"):
                    layer.padding = layer.original_padding
            else:
                # 如果禁用了填充，则保存原始填充并设置为无填充
                layer.original_padding = layer.padding
                layer.padding = tuple(0 for _ in range(len(layer.padding)))

        # 设置填充标志
        self._padding = value

    def get_delay(self):
        """
        计算编解码器的延迟。

        延迟被定义为输入长度与输出长度之差的一半。

        返回:
            int: 延迟长度。
        """
        # 使用任何数字都可以，因为延迟与输入长度无关，这里使用 0 作为示例输入长度
        # Any number works here, delay is invariant to input length
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        # 查找所有卷积层（Conv1d 和 ConvTranspose1d）
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        # 从后向前遍历层，计算延迟
        for layer in reversed(layers):
            d = layer.dilation[0]    # 获取膨胀因子
            k = layer.kernel_size[0] # 获取卷积核大小
            s = layer.stride[0]      # 获取步幅

            if isinstance(layer, nn.ConvTranspose1d):
                # 对于转置卷积，计算输出长度
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                # 对于普通卷积，计算输出长度
                L = (L - 1) * s + d * (k - 1) + 1

            # 向上取整
            L = math.ceil(L)

        # 最终的输入长度
        l_in = L

        # 计算延迟
        return (l_in - l_out) // 2

    def get_output_length(self, input_length):
        """
        计算编解码器的输出长度。

        参数:
            input_length (int): 输入长度。

        返回:
            int: 输出长度。
        """
        L = input_length
        # Calculate output length
        # 计算输出长度
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]    # 获取膨胀因子
                k = layer.kernel_size[0] # 获取卷积核大小
                s = layer.stride[0]      # 获取步幅

                if isinstance(layer, nn.Conv1d):
                    # 对于普通卷积，计算输出长度
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    # 对于转置卷积，计算输出长度
                    L = (L - 1) * s + d * (k - 1) + 1

                # 向下取整
                L = math.floor(L)
        # 返回输出长度
        return L

    @torch.no_grad()
    def compress(
        self,
        audio_path_or_signal: Union[str, Path, AudioSignal],
        win_duration: float = 1.0,
        verbose: bool = False,
        normalize_db: float = -16,
        n_quantizers: int = None,
    ) -> DACFile:
        """
        将音频信号从文件或 AudioSignal 对象处理为离散编码。

        该函数以短窗口处理信号，使用恒定的 GPU 内存。

        参数:
            audio_path_or_signal (Union[str, Path, AudioSignal]): 要重建的音频信号，可以是文件路径或 AudioSignal 对象。
            win_duration (float, 可选): 窗口持续时间（秒），默认 5.0 秒。
            verbose (bool, 可选): 是否显示进度信息，默认 False。
            normalize_db (float, 可选): 归一化音量（分贝），默认 -16 dB。

        返回:
            DACFile: 包含压缩编码和元数据的对象，用于解压缩。
        """
        # 获取音频信号
        audio_signal = audio_path_or_signal
        if isinstance(audio_signal, (str, Path)):
            # 如果输入是字符串或 Path 对象，则使用 ffmpeg 从文件中加载音频信号
            audio_signal = AudioSignal.load_from_file_with_ffmpeg(str(audio_signal))

        self.eval()
        # 保存原始填充设置
        original_padding = self.padding
        original_device = audio_signal.device

        # 克隆音频信号以避免修改原始数据
        audio_signal = audio_signal.clone()
        # 获取响度计算函数
        original_sr = audio_signal.sample_rate

        # 获取重采样函数
        resample_fn = audio_signal.resample
        loudness_fn = audio_signal.loudness

        # If audio is > 10 minutes long, use the ffmpeg versions
        # 如果音频时长超过 10 分钟，则使用 ffmpeg 版本的重采样和响度计算函数
        if audio_signal.signal_duration >= 10 * 60 * 60:
            resample_fn = audio_signal.ffmpeg_resample
            loudness_fn = audio_signal.ffmpeg_loudness

        # 获取原始信号长度
        original_length = audio_signal.signal_length
        # 将音频信号重采样到模型的采样率
        resample_fn(self.sample_rate)
        # 计算音频信号的响度
        input_db = loudness_fn()

        if normalize_db is not None:
            # 对音频信号进行归一化
            audio_signal.normalize(normalize_db)
        # 确保音频信号的最大值不超过 1.0
        audio_signal.ensure_max_of_audio()

        # 获取音频数据的形状 (批次, 通道, 时间)
        nb, nac, nt = audio_signal.audio_data.shape
        # 重塑音频数据为 (批次*通道, 1, 时间)
        audio_signal.audio_data = audio_signal.audio_data.reshape(nb * nac, 1, nt)
        # 确定窗口持续时间
        win_duration = (
            audio_signal.signal_duration if win_duration is None else win_duration
        )

        if audio_signal.signal_duration <= win_duration:
            # Unchunked compression (used if signal length < win duration)
            # 如果信号长度小于窗口持续时间，则使用无分块压缩
            self.padding = True
            n_samples = nt # 设置样本数为时间步长
            hop = nt # 设置步幅为时间步长
        else:
            # Chunked inference
            # 如果信号长度大于窗口持续时间，则使用分块推理
            self.padding = False
            # Zero-pad signal on either side by the delay
            # 在信号的两侧进行零填充，填充量为延迟
            audio_signal.zero_pad(self.delay, self.delay)
            # 计算每个窗口的样本数
            n_samples = int(win_duration * self.sample_rate)
            # Round n_samples to nearest hop length multiple
            # 将 n_samples 向上取整到最近的 hop_length 的倍数
            n_samples = int(math.ceil(n_samples / self.hop_length) * self.hop_length)
            # 计算步幅
            hop = self.get_output_length(n_samples)

        # 初始化编码列表
        codes = []
        range_fn = range if not verbose else tqdm.trange

        for i in range_fn(0, nt, hop):
            # 获取当前窗口的音频数据
            x = audio_signal[..., i : i + n_samples]
            # 对音频数据进行填充
            x = x.zero_pad(0, max(0, n_samples - x.shape[-1]))

            audio_data = x.audio_data.to(self.device)
            # 预处理音频数据
            audio_data = self.preprocess(audio_data, self.sample_rate)
            # 编码音频数据
            _, c, _, _, _ = self.encode(audio_data, n_quantizers)
            # 将编码添加到列表中
            codes.append(c.to(original_device))
            # 获取当前块的编码长度
            chunk_length = c.shape[-1]

        # 连接所有编码块
        codes = torch.cat(codes, dim=-1)

        dac_file = DACFile(
            codes=codes,
            chunk_length=chunk_length,
            original_length=original_length,
            input_db=input_db,
            channels=nac,
            sample_rate=original_sr,
            padding=self.padding,
            dac_version=SUPPORTED_VERSIONS[-1],
        )

        if n_quantizers is not None:
            # 如果指定了量化器的数量，则截取编码
            codes = codes[:, :n_quantizers, :]

        # 恢复原始填充设置
        self.padding = original_padding
        return dac_file

    @torch.no_grad()
    def decompress(
        self,
        obj: Union[str, Path, DACFile],
        verbose: bool = False,
    ) -> AudioSignal:
        """
        从给定的 .dac 文件重建音频。

        参数:
            obj (Union[str, Path, DACFile]): .dac 文件的位置或相应的 DACFile 对象。
            verbose (bool, 可选): 如果为 True，则打印进度信息，默认 False。

        返回:
            AudioSignal: 重建的音频对象。
        """
        self.eval()
        if isinstance(obj, (str, Path)):
            # 如果输入是字符串或 Path 对象，则加载 DACFile 对象
            obj = DACFile.load(obj)

        original_padding = self.padding
        # 设置填充为 DACFile 对象的填充设置
        self.padding = obj.padding

        range_fn = range if not verbose else tqdm.trange
        # 获取编码
        codes = obj.codes
        # 获取编码设备
        original_device = codes.device
        # 获取编码块长度
        chunk_length = obj.chunk_length
        # 初始化重建列表
        recons = []

        for i in range_fn(0, codes.shape[-1], chunk_length):
            c = codes[..., i : i + chunk_length].to(self.device)
            # 从编码中解码出潜在变量
            z = self.quantizer.from_codes(c)[0]
            # 解码潜在变量
            r = self.decode(z)
            # 将重建结果添加到列表中
            recons.append(r.to(original_device))

        # 连接所有重建块
        recons = torch.cat(recons, dim=-1)
        # 创建 AudioSignal 对象
        recons = AudioSignal(recons, self.sample_rate)

        # 获取重采样函数
        resample_fn = recons.resample
        # 获取响度计算函数
        loudness_fn = recons.loudness

        # If audio is > 10 minutes long, use the ffmpeg versions
        # 如果音频时长超过 10 分钟，则使用 ffmpeg 版本的重采样和响度计算函数
        if recons.signal_duration >= 10 * 60 * 60:
            resample_fn = recons.ffmpeg_resample
            loudness_fn = recons.ffmpeg_loudness

        # 对重建音频进行归一化
        recons.normalize(obj.input_db)
        # 将重建音频重采样到原始采样率
        resample_fn(obj.sample_rate)
        # 截取重建音频到原始长度
        recons = recons[..., : obj.original_length]
        # 计算重建音频的响度
        loudness_fn()
        # 重塑音频数据为 (批次, 通道, 时间)
        recons.audio_data = recons.audio_data.reshape(
            -1, obj.channels, obj.original_length
        )

        # 恢复原始填充设置
        self.padding = original_padding
        # 返回重建的 AudioSignal 对象
        return recons
