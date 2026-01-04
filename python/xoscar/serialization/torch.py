# Copyright 2024 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Tuple

import numpy as np

from ..utils import lazy_import
from .core import Serializer, buffered

# 延迟导入PyTorch，避免强制依赖
torch = lazy_import("torch")


class TorchTensorSerializer(Serializer):
    @buffered
    def serial(self, obj: "torch.Tensor", context: Dict):  # type: ignore
        # 对于CPU张量，直接使用内存视图
        if obj.device.type == "cpu":
            # 确保张量是连续的以避免复制
            if not obj.is_contiguous():
                obj = obj.contiguous()
            # 获取内存视图并构建头部信息
            header = {
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
                "device": obj.device.type,
                "requires_grad": obj.requires_grad,
                "strides": tuple(obj.stride()),
            }
            # 将张量数据转换为uint8视图以获取原始字节
            data = obj.view(torch.uint8).cpu().numpy()
            return (header,), [memoryview(data)], True
        elif obj.device.type == "cuda":
            # 对于CUDA张量，使用__cuda_array_interface__
            if not (
                obj.is_contiguous()
                or obj.is_contiguous(memory_format=torch.channels_last)
            ):
                obj = obj.contiguous()

            # 获取CUDA数组接口信息
            cuda_interface = obj.__cuda_array_interface__
            header = {
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
                "device": obj.device.type,
                "device_index": obj.device.index,
                "requires_grad": obj.requires_grad,
                "strides": tuple(obj.stride()),
                "cuda_array_interface": cuda_interface,
            }

            # 创建缓冲区视图（零拷贝）
            buffer = obj.data_ptr()  # 获取原始设备指针
            # 通过numpy创建设备缓冲区视图（实际不复制数据）
            buffer_view = np.ndarray(
                shape=(obj.nbytes,),
                dtype=np.uint8,
                buffer=None,
                offset=buffer,
                strides=(1,),
            )
            return (header,), [buffer_view], True
        else:
            # 不支持的设备类型，回退到常规序列化
            raise NotImplementedError(f"Unsupported device type: {obj.device.type}")

    def deserial(self, serialized: Tuple, context: Dict, subs: List[Any]):
        header = serialized[0]
        device = header["device"]
        data_buffer = subs[0]

        # 从缓冲区重建张量
        if device == "cpu":
            # 从内存视图创建numpy数组，再转换为PyTorch张量
            np_array = np.frombuffer(
                data_buffer, dtype=np.dtype(header["dtype"].split(".")[-1])
            )
            tensor = torch.from_numpy(np_array).view(header["shape"])
        elif device == "cuda":
            # 从CUDA数组接口信息重建
            np_array = np.frombuffer(data_buffer, dtype=np.uint8)
            # 将数据转移到目标设备
            tensor = (
                torch.from_numpy(np_array)
                .view(torch.dtype(header["dtype"]), *header["shape"])
                .to(device=f"cuda:{header['device_index']}")
            )
        else:
            raise NotImplementedError(f"Unsupported device type: {device}")

        # 恢复requires_grad属性
        tensor.requires_grad = header["requires_grad"]
        return tensor


# 仅当PyTorch可用时注册序列化器
if torch is not None:
    TorchTensorSerializer.register("torch.Tensor")
