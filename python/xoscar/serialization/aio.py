# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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

import asyncio
import struct
from typing import Any, BinaryIO, Union

import cloudpickle

from ..utils import lazy_import
from ._security import (
    AUTH_SERIALIZATION_VERSION,
    MAC_SIZE,
    MAX_MANIFEST_SIZE,
    get_auth_key,
    pack_manifest,
    sign_buffers,
    unpack_manifest,
    verify_buffers,
)
from .core import deserialize, serialize_with_spawn

rmm = lazy_import("rmm")

DEFAULT_SERIALIZATION_VERSION = 1
DEFAULT_SPAWN_THRESHOLD = 100
BUFFER_SIZES_NAME = "buf_sizes"


class AioSerializer:
    def __init__(self, obj: Any, compress=0):
        self._obj = obj
        self._compress = compress

    async def _get_buffers(self):
        headers, buffers = await serialize_with_spawn(
            self._obj, spawn_threshold=DEFAULT_SPAWN_THRESHOLD
        )

        def _is_cuda_buffer(buf: Union["rmm.DeviceBuffer", BinaryIO]):  # type: ignore
            return hasattr(buf, "__cuda_array_interface__")

        is_cuda_buffers = [_is_cuda_buffer(buf) for buf in buffers]
        headers[0]["is_cuda_buffers"] = is_cuda_buffers
        buffer_sizes = [
            buf.nbytes if hasattr(buf, "nbytes") else len(buf) for buf in buffers
        ]
        headers[0][BUFFER_SIZES_NAME] = buffer_sizes
        header = cloudpickle.dumps(headers)
        auth_key = get_auth_key()
        if auth_key is None:
            # Preserve the v1 format, including custom serializer headers, for
            # existing deployments whose network peers are already trusted.
            prefix = struct.pack(
                "<BQH", DEFAULT_SERIALIZATION_VERSION, len(header), self._compress
            )
            return [prefix, header, *buffers]

        # The manifest contains only lengths and device flags. It lets receivers
        # read and authenticate the ENTIRE frame before unpickling even a header.
        manifest = pack_manifest(len(header), buffer_sizes, is_cuda_buffers)
        prefix = struct.pack(
            "<BQH", AUTH_SERIALIZATION_VERSION, len(manifest), self._compress
        )
        manifest_mac = sign_buffers(auth_key, (prefix, manifest))
        frame_mac = sign_buffers(auth_key, (prefix, manifest, header, *buffers))
        return [prefix, manifest_mac, frame_mac, manifest, header, *buffers]

    async def run(self):
        return await self._get_buffers()


MALFORMED_MSG = "Received malformed data, please check Xoscar version on both side."


def get_header_length(header_bytes: bytes):
    if len(header_bytes) != 11:
        raise EOFError(MALFORMED_MSG)
    version, header_length, _ = struct.unpack("<BQH", header_bytes)
    if version not in (DEFAULT_SERIALIZATION_VERSION, AUTH_SERIALIZATION_VERSION):
        raise EOFError(MALFORMED_MSG)
    return version, header_length


class AioDeserializer:
    def __init__(self, file, buffer_reader=None):
        self._file = file
        self._buffer_reader = buffer_reader
        self._buffers = None
        self.frame_mac = None
        self._frame_overhead = 11
        self._header_length = 0

    async def _readexactly(self, n: int):
        read = (
            self._file.readexactly
            if hasattr(self._file, "readexactly")
            else self._file.read
        )
        data = await read(n)
        if len(data) != n:
            raise EOFError("Received truncated frame")
        return data

    async def _read_buffers(self, header):
        if self._buffer_reader is not None:
            return await self._buffer_reader(header, self._file)
        return [
            await self._readexactly(size) for size in header[0].pop(BUFFER_SIZES_NAME)
        ]

    async def _get_obj_header_bytes(self):
        try:
            prefix = bytes(await self._readexactly(11))
        except ConnectionResetError:
            raise EOFError("Server may be closed") from None
        version, length = get_header_length(prefix)
        auth_key = get_auth_key()
        if version == DEFAULT_SERIALIZATION_VERSION:
            if auth_key is not None:
                raise EOFError(
                    "Rejected unauthenticated frame: XOSCAR_AUTH_KEY is configured"
                )
            self._frame_overhead = 11
            self._header_length = length
            return bytes(await self._readexactly(length))

        if auth_key is None:
            raise EOFError(
                "Received authenticated frame but XOSCAR_AUTH_KEY is not configured"
            )
        if length > MAX_MANIFEST_SIZE:
            raise EOFError("Authenticated frame manifest exceeds the size limit")
        manifest_mac = bytes(await self._readexactly(MAC_SIZE))
        frame_mac = bytes(await self._readexactly(MAC_SIZE))
        manifest = bytes(await self._readexactly(length))
        # Verify before trusting sizes or allocating payload/device buffers.
        if not verify_buffers(auth_key, (prefix, manifest), manifest_mac):
            raise EOFError("Frame manifest authentication failed")
        header_length, sizes, cuda_flags = unpack_manifest(manifest)
        header_bytes = bytes(await self._readexactly(header_length))
        buffers = await self._read_buffers(
            [{BUFFER_SIZES_NAME: sizes, "is_cuda_buffers": cuda_flags}]
        )
        if not verify_buffers(
            auth_key, (prefix, manifest, header_bytes, *buffers), frame_mac
        ):
            raise EOFError("Frame payload authentication failed")
        self._buffers = buffers
        self.frame_mac = frame_mac
        self._frame_overhead = 11 + 2 * MAC_SIZE + len(manifest)
        self._header_length = header_length
        return header_bytes

    async def get_header(self):
        # In keyed mode, _get_obj_header_bytes verifies all raw bytes before
        # custom Python objects in the header or payload can be reconstructed.
        return cloudpickle.loads(await self._get_obj_header_bytes())

    async def get_buffers(self, header):
        if self._buffers is not None:
            buffers, self._buffers = self._buffers, None
            header[0].pop(BUFFER_SIZES_NAME)
            return buffers
        return await self._read_buffers(header)

    async def _get_obj(self):
        header = await self.get_header()
        buffers = await self.get_buffers(header)
        num_objs = header[0].get("_N", 0)
        if num_objs <= DEFAULT_SPAWN_THRESHOLD:
            return deserialize(header, buffers)
        return await asyncio.to_thread(deserialize, header, buffers)

    async def run(self):
        return await self._get_obj()

    async def get_size(self):
        header = await self.get_header()
        buffer_sizes = header[0].pop(BUFFER_SIZES_NAME)
        return self._frame_overhead + self._header_length + sum(buffer_sizes)
