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
cupy = lazy_import("cupy")


def rmm_to_torch(buf):
    cupy_arr = cupy.asarray(buf)  # zero-copy
    torch_tensor = torch.utils.dlpack.from_dlpack(cupy_arr.toDlpack())  # zero-copy
    return torch_tensor


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
            header = {
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
                "device": obj.device.type,
                "device_index": obj.device.index,
                "requires_grad": obj.requires_grad,
                "strides": tuple(obj.stride()),
            }

            # ---- Core idea: expose raw CUDA memory as a uint8 buffer (zero-copy) ----
            # Get the underlying untyped storage that actually owns the CUDA memory
            storage = obj.untyped_storage()

            # Create a uint8 CUDA tensor that will act as a byte-view of the same memory
            # This does NOT allocate new GPU memory; it only creates a new Tensor wrapper
            buffer = torch.empty(
                (storage.nbytes(),),
                dtype=torch.uint8,
                device=obj.device,
            )

            # Make the buffer tensor share the same storage (zero-copy)
            buffer.set_(storage)

            # Return: (metadata,), [raw CUDA buffer], mark as buffered
            return (header,), [buffer], True
        else:
            # for unsupported device
            raise NotImplementedError(f"Unsupported device type: {obj.device.type}")

    def deserial(self, serialized: Tuple, context: Dict, subs: List[Any]):
        header = serialized[0]
        device = header["device"]
        data_buffer = subs[0]

        if device == "cpu":
            # create numpy array from memory viewpoint, then convert to PyTorch tensor
            np_array = np.frombuffer(
                data_buffer, dtype=np.dtype(header["dtype"].split(".")[-1])
            )
            tensor = torch.from_numpy(np_array).view(header["shape"])
        elif device == "cuda":
            # Unpack metadata
            (header,) = serialized

            # Raw CUDA buffer (uint8 tensor)
            buffer = subs[0]
            if not isinstance(buffer, torch.Tensor):
                buffer = rmm_to_torch(buffer)
            assert buffer.is_cuda, "buffer must be a CUDA tensor"
            assert buffer.dtype == torch.uint8, "buffer must be uint8"

            # Get the shared CUDA storage
            storage = buffer.untyped_storage()

            # Resolve original dtype
            dtype_name = header["dtype"].split(".")[-1]
            dtype = getattr(torch, dtype_name)

            # Create an empty tensor wrapper with the correct dtype
            # This does NOT allocate new GPU memory for data
            tensor = torch.empty(0, device=buffer.device, dtype=dtype)

            # Bind the tensor to the same storage with original shape/stride
            # Pure zero-copy: only a new Tensor view, no data movement
            tensor.set_(
                storage,
                storage_offset=0,
                size=tuple(header["shape"]),
                stride=tuple(header["strides"]),
            )

            # Restore requires_grad if needed
            if header.get("requires_grad"):
                tensor.requires_grad_(True)

            return tensor
        else:
            raise NotImplementedError(f"Unsupported device type: {device}")

        # recover requires_grad attributes
        tensor.requires_grad = header["requires_grad"]
        return tensor


# only when torch is available, we register module
if torch is not None:
    TorchTensorSerializer.register("torch.Tensor")
