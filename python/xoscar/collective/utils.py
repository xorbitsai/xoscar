# Copyright 2022-2023 XProbe Inc.
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
import os

import numpy as np

from ..utils import lazy_import

cupy = lazy_import("cupy")


def convert_data_to_np_array(data):
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.frombuffer(data, dtype="u1")


def convert_data_to_cp_array(data):
    if isinstance(data, cupy.ndarray):
        return data
    else:
        return cupy.frombuffer(data, dtype="u1")


def get_rank_address_via_env(env_key: str, err_message: str) -> str:
    address = os.environ.get(env_key, None)
    if address is None:
        raise RuntimeError(err_message)
    return address
