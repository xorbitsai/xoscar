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
from enum import IntEnum
from typing import Dict, Type

import numpy as np

from . import xoscar_pygloo as xp

ReduceOpMappingGloo: Dict["CollectiveReduceOp", "xp.ReduceOp"] = {}
AllReduceAlgorithmMappingGloo: Dict["AllReduceAlgorithm", "xp.AllreduceAlgorithm"] = {}


def _register_reduce_op(reduce_op):
    for op_type in reduce_op:
        ReduceOpMappingGloo[op_type] = xp.ReduceOp(op_type)
    return reduce_op


def _register_allreduce_algo(algorithms):
    for algo in algorithms:
        AllReduceAlgorithmMappingGloo[algo] = xp.AllreduceAlgorithm(algo)
    return algorithms


@_register_reduce_op
class CollectiveReduceOp(IntEnum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    UNUSED = 7


@_register_allreduce_algo
class AllReduceAlgorithm(IntEnum):
    UNSPECIFIED = 0
    RING = 1
    BCUBE = 2


TypeMappingGloo: Dict[Type[np.dtype], "xp.GlooDataType_t"] = {
    np.int8: xp.GlooDataType_t.glooInt8,
    np.uint8: xp.GlooDataType_t.glooUint8,
    np.int32: xp.GlooDataType_t.glooInt32,
    np.uint32: xp.GlooDataType_t.glooUint32,
    np.int64: xp.GlooDataType_t.glooInt64,
    np.uint64: xp.GlooDataType_t.glooUint64,
    np.float16: xp.GlooDataType_t.glooFloat16,
    np.float32: xp.GlooDataType_t.glooFloat32,
    np.float64: xp.GlooDataType_t.glooFloat64,
}

# Some static variables
INVOKE_ERROR_MESSAGE = "Collective-related functions must be called in a process that is involved in collection communication."
RANK_ADDRESS_ENV_KEY = "COLLECTIVE_RANK_ADDRESS"
RENDEZVOUS_MASTER_IP_ENV_KEY = "COLLECTIVE_MASTER_IP"
RENDEZVOUS_MASTER_PORT_ENV_KEY = "COLLECTIVE_MASTER_PORT"
