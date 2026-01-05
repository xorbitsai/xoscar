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

# lazy import PyTorch to avoid enforced dependency
torch = lazy_import("torch")


class TorchTensorSerializer(Serializer):
    @buffered
    def serial(self, obj: "torch.Tensor", context: Dict):  # type: ignore
        # for cpu tensor, use memory viewpoint
        if obj.device.type == "cpu":
            # make sure tensor is contiguous
            if not obj.is_contiguous():
                obj = obj.contiguous()
            # get memory viewpoint and collect header information
            header = {
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
                "device": obj.device.type,
                "requires_grad": obj.requires_grad,
                "strides": tuple(obj.stride()),
            }
            # convert tensor data into uint8 viewpoint, to get original bytes
            data = obj.view(torch.uint8).cpu().numpy()
            return (header,), [memoryview(data)], True
        elif obj.device.type == "cuda":
            # for CUDA， use __cuda_array_interface__
            if not (
                obj.is_contiguous()
                or obj.is_contiguous(memory_format=torch.channels_last)
            ):
                obj = obj.contiguous()

            # get cuda array interface information
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

            # create buffer view, zero copy, get device pointer
            buffer = obj.data_ptr()
            # instead of actual copy, create device buffer viewpoint from numpy
            buffer_view = np.ndarray(
                shape=(obj.nbytes,),
                dtype=np.uint8,
                buffer=None,
                offset=buffer,
                strides=(1,),
            )
            return (header,), [buffer_view], True
        else:
            # for unsupported device
            raise NotImplementedError(f"Unsupported device type: {obj.device.type}")

    def deserial(self, serialized: Tuple, context: Dict, subs: List[Any]):
        header = serialized[0]
        device = header["device"]
        data_buffer = subs[0]

        # 从缓冲区重建张量
        if device == "cpu":
            # create numpy array from memory viewpoint, then convert to PyTorch tensor
            np_array = np.frombuffer(
                data_buffer, dtype=np.dtype(header["dtype"].split(".")[-1])
            )
            tensor = torch.from_numpy(np_array).view(header["shape"])
        elif device == "cuda":
            np_array = np.frombuffer(data_buffer, dtype=np.uint8)
            # move data into cuda device
            tensor = (
                torch.from_numpy(np_array)
                .view(torch.dtype(header["dtype"]), *header["shape"])
                .to(device=f"cuda: {header['device_index']}")
            )
        else:
            raise NotImplementedError(f"Unsupported device type: {device}")

        # recover requires_grad attributes
        tensor.requires_grad = header["requires_grad"]
        return tensor


# only when torch is available, we register module
if torch is not None:
    TorchTensorSerializer.register("torch.Tensor")
