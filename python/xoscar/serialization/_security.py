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

"""Authentication of actor frames, enabled explicitly with XOSCAR_AUTH_KEY."""

import hashlib
import hmac
import os
import struct

import numpy as np

AUTH_SERIALIZATION_VERSION = 2
MAC_SIZE = hashlib.sha256().digest_size
MAX_MANIFEST_SIZE = 16 * 1024**2
MAX_HEADER_SIZE = 64 * 1024**2
_MANIFEST_HEADER = struct.Struct("<QQ")
_BUFFER_INFO = struct.Struct("<QB")


def get_auth_key():
    """Return the shared cluster key, or None for the legacy trusted-network mode."""
    key = os.environ.get("XOSCAR_AUTH_KEY")
    return key.encode("utf-8") if key else None


def byte_view(buffer):
    """View contiguous host memory in physical byte order without copying."""
    view = memoryview(buffer)
    if not view.contiguous:
        raise ValueError("Authentication requires contiguous buffers")
    if not view.nbytes:
        return memoryview(b"")
    if not view.c_contiguous:
        # memoryview.cast and hashlib only accept C-contiguous views. Flatten
        # Fortran-contiguous storage without changing its physical byte order.
        view = memoryview(np.asarray(view).ravel(order="K"))
    return view.cast("B")


def _host_buffer(buffer):
    if hasattr(buffer, "__cuda_array_interface__"):
        # Hash device data on the host. Authenticated UCX therefore has a host
        # staging cost, unlike the unauthenticated zero-copy transport.
        if hasattr(buffer, "copy_to_host"):
            buffer = buffer.copy_to_host()
        elif hasattr(buffer, "get"):
            buffer = buffer.get(order="A")
        else:  # pragma: no cover
            from numba import cuda

            buffer = cuda.as_cuda_array(buffer).copy_to_host()
    return byte_view(buffer)


def sign_buffers(key: bytes, buffers) -> bytes:
    """Authenticate all buffers in wire order without concatenating the payload."""
    mac = hmac.new(key, digestmod=hashlib.sha256)
    for buffer in buffers:
        mac.update(_host_buffer(buffer))
    return mac.digest()


def verify_buffers(key: bytes, buffers, mac: bytes) -> bool:
    return hmac.compare_digest(sign_buffers(key, buffers), mac)


def sign(key: bytes, data: bytes) -> bytes:
    return sign_buffers(key, (data,))


def verify(key: bytes, data: bytes, mac: bytes) -> bool:
    return verify_buffers(key, (data,), mac)


def pack_manifest(header_length, buffer_sizes, is_cuda_buffers):
    manifest = bytearray(_MANIFEST_HEADER.pack(header_length, len(buffer_sizes)))
    for size, is_cuda in zip(buffer_sizes, is_cuda_buffers):
        manifest.extend(_BUFFER_INFO.pack(size, is_cuda))
    if len(manifest) > MAX_MANIFEST_SIZE or header_length > MAX_HEADER_SIZE:
        raise ValueError("Authenticated frame metadata exceeds the size limit")
    return bytes(manifest)


def unpack_manifest(manifest):
    """Decode authenticated, fixed-width framing metadata without unpickling."""
    if len(manifest) < _MANIFEST_HEADER.size:
        raise EOFError("Malformed authenticated frame manifest")
    header_length, count = _MANIFEST_HEADER.unpack_from(manifest)
    if (
        header_length > MAX_HEADER_SIZE
        or len(manifest) != _MANIFEST_HEADER.size + count * _BUFFER_INFO.size
    ):
        raise EOFError("Malformed authenticated frame manifest")
    sizes, cuda_flags = [], []
    for size, flag in _BUFFER_INFO.iter_unpack(manifest[_MANIFEST_HEADER.size :]):
        if flag not in (0, 1):
            raise EOFError("Malformed authenticated frame device flag")
        sizes.append(size)
        cuda_flags.append(bool(flag))
    return header_length, sizes, cuda_flags
