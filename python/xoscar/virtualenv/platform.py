# Copyright 2022-2025 XProbe Inc.
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

from typing import Optional


def check_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except (ImportError, AttributeError):
        return False


def get_cuda_version() -> Optional[str]:
    try:
        import torch

        return torch.version.cuda  # e.g. '12.1'
    except (ImportError, AttributeError):
        return None


def get_cuda_arch() -> Optional[str]:
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}"  # e.g. 'sm_80'
    except (ImportError, AttributeError, AssertionError):
        # If no cuda available,
        # AssertionError("Torch not compiled with CUDA enabled")
        # will be raised
        return None


def check_npu_available() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False
