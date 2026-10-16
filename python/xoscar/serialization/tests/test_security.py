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

import asyncio
import io
import struct
from collections import deque

import cloudpickle
import numpy as np
import pytest

from ...backends.communication import SocketChannel
from ...backends.communication.errors import ChannelClosed
from ...backends.communication.ucx import UCXChannel
from .. import aio
from .._security import (
    AUTH_SERIALIZATION_VERSION,
    MAC_SIZE,
    MAX_MANIFEST_SIZE,
    pack_manifest,
    sign_buffers,
    unpack_manifest,
)


class CustomList(list):
    pass


class _FakeReader:
    def __init__(self, data):
        self._b = io.BytesIO(data)

    async def readexactly(self, size):
        return self._b.read(size)


async def _frame(obj):
    return await aio.AioSerializer(obj).run()


async def _unframe(buffers):
    return await aio.AioDeserializer(
        _FakeReader(b"".join(bytes(buf) for buf in buffers))
    ).run()


@pytest.mark.parametrize("key", [None, "cluster-secret"])
@pytest.mark.parametrize(
    "obj",
    [
        None,
        {"hello": [1, 2, 3]},
        np.arange(8),
        np.empty(0),
        CustomList([1, 2]),
        ValueError("boom"),
    ],
)
async def test_transport_roundtrip(monkeypatch, key, obj):
    if key is None:
        monkeypatch.delenv("XOSCAR_AUTH_KEY", raising=False)
    else:
        monkeypatch.setenv("XOSCAR_AUTH_KEY", key)
    buffers = await _frame(obj)
    expected_version = AUTH_SERIALIZATION_VERSION if key else 1
    assert buffers[0][0] == expected_version
    result = await _unframe(buffers)
    assert type(result) is type(obj)
    if isinstance(obj, np.ndarray):
        np.testing.assert_array_equal(result, obj)
    else:
        assert repr(result) == repr(obj)
    reader = _FakeReader(b"".join(bytes(buf) for buf in buffers))
    assert await aio.AioDeserializer(reader).get_size() == sum(map(len, buffers))


async def test_legacy_wire_format_is_unchanged(monkeypatch):
    monkeypatch.delenv("XOSCAR_AUTH_KEY", raising=False)
    buffers = await _frame({"x": "y"})
    assert len(buffers[0]) == 11
    assert buffers[0] == struct.pack("<BQH", 1, len(buffers[1]), 0)
    header = cloudpickle.loads(buffers[1])
    assert header[0][aio.BUFFER_SIZES_NAME] == [len(buf) for buf in buffers[2:]]


@pytest.mark.parametrize("receiver_key", [None, "wrong-key"])
async def test_wrong_or_missing_key(monkeypatch, receiver_key):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "sender-key")
    buffers = await _frame({"q": 1})
    if receiver_key is None:
        monkeypatch.delenv("XOSCAR_AUTH_KEY")
    else:
        monkeypatch.setenv("XOSCAR_AUTH_KEY", receiver_key)
    with pytest.raises(EOFError):
        await _unframe(buffers)


async def test_reject_legacy_frame_in_keyed_mode(monkeypatch):
    monkeypatch.delenv("XOSCAR_AUTH_KEY", raising=False)
    buffers = await _frame(np.arange(8))
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    with pytest.raises(EOFError, match="unauthenticated"):
        await _unframe(buffers)


@pytest.mark.parametrize("part", [0, 1, 2, 3, 4, 5, 6])
async def test_tampered_frame_never_unpickles(monkeypatch, part):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    # Two independent buffers exercise authentication of every payload buffer.
    buffers = await _frame([np.arange(4), np.arange(6)])
    assert len(buffers) == 7
    buffers = list(buffers)
    tampered = bytearray(buffers[part])
    tampered[-1] ^= 1
    buffers[part] = tampered

    def forbidden_loads(*args, **kwargs):
        pytest.fail("Unpickling ran before complete frame authentication")

    monkeypatch.setattr(aio.cloudpickle, "loads", forbidden_loads)
    with pytest.raises(EOFError):
        await _unframe(buffers)


async def test_reusing_signed_header_does_not_authenticate_changed_payload(monkeypatch):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    buffers = await _frame(np.arange(8))
    # Preserve all signed metadata and replace the same-sized body, the bypass
    # identified in the original review.
    buffers[-1] = bytes(len(buffers[-1]))
    with pytest.raises(EOFError, match="payload authentication"):
        await _unframe(buffers)


async def test_valid_header_mac_does_not_allow_unverified_header_code(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    marker = tmp_path / "executed"

    class HeaderPayload:
        def __reduce__(self):
            return (eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",))

    header = cloudpickle.dumps(HeaderPayload())
    manifest = pack_manifest(len(header), [], [])
    prefix = struct.pack("<BQH", AUTH_SERIALIZATION_VERSION, len(manifest), 0)
    manifest_mac = sign_buffers(b"cluster-secret", (prefix, manifest))
    with pytest.raises(EOFError, match="payload authentication"):
        await _unframe([prefix, manifest_mac, bytes(MAC_SIZE), manifest, header])
    assert not marker.exists()


async def test_payload_gadget_cannot_reuse_valid_signed_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    buffers = await _frame(ValueError("x" * 2000))
    marker = tmp_path / "payload-executed"

    class Payload:
        def __reduce__(self):
            return (eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",))

    payload = cloudpickle.dumps(Payload())
    assert len(payload) <= len(buffers[-1])
    buffers[-1] = payload.ljust(len(buffers[-1]), b"\x00")
    with pytest.raises(EOFError, match="payload authentication"):
        await _unframe(buffers)
    assert not marker.exists()


@pytest.mark.parametrize("length", [0, 1, 10, 11, 42, 80])
async def test_truncated_authenticated_frame(monkeypatch, length):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    buffers = await _frame(np.arange(8))
    data = b"".join(bytes(buf) for buf in buffers)
    with pytest.raises(EOFError):
        await _unframe([data[:length]])


async def test_manifest_size_checked_before_read_or_allocation(monkeypatch):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    prefix = struct.pack("<BQH", AUTH_SERIALIZATION_VERSION, MAX_MANIFEST_SIZE + 1, 0)
    with pytest.raises(EOFError, match="size limit"):
        await _unframe([prefix])


@pytest.mark.parametrize("method", ["copy_to_host", "get"])
def test_authentication_hashes_device_contents(method):
    class DeviceBuffer:
        __cuda_array_interface__ = {}

    buffer = DeviceBuffer()
    setattr(buffer, method, lambda **kwargs: np.frombuffer(b"device-bytes", dtype="u1"))
    assert sign_buffers(b"key", [buffer]) == sign_buffers(b"key", [b"device-bytes"])


@pytest.mark.parametrize("manifest", [b"", bytes(15), struct.pack("<QQ", 1, 10)])
def test_malformed_manifest(manifest):
    with pytest.raises(EOFError, match="Malformed"):
        unpack_manifest(manifest)


class _FakeUCXEndpoint:
    def __init__(self, buffers=()):
        self.buffers = deque(bytes(buf) for buf in buffers)
        self.aborted = False

    def set_close_callback(self, callback):
        self.callback = callback

    async def recv(self, buffer):
        data = self.buffers.popleft()
        assert len(data) == memoryview(buffer).nbytes
        memoryview(buffer).cast("B")[:] = data

    async def send(self, buffer):
        # UCX tagged sends preserve message boundaries, including zero-byte
        # messages. Silently dropping them here would hide framing defects.
        self.buffers.append(memoryview(buffer).tobytes(order="A"))

    def abort(self):
        self.aborted = True


@pytest.mark.parametrize("key", [None, "cluster-secret"])
async def test_ucx_roundtrip_without_device_runtime(monkeypatch, key):
    if key is None:
        monkeypatch.delenv("XOSCAR_AUTH_KEY", raising=False)
    else:
        monkeypatch.setenv("XOSCAR_AUTH_KEY", key)
    endpoint = _FakeUCXEndpoint()
    channel = UCXChannel(endpoint)
    data = [np.arange(8), CustomList([1, 2]), np.empty(0)]
    await channel.send(data)
    result = await channel.recv()
    np.testing.assert_array_equal(result[0], data[0])
    assert type(result[1]) is CustomList
    np.testing.assert_array_equal(result[2], data[2])
    assert not endpoint.buffers


@pytest.mark.parametrize("part", [3, 4, 5])
async def test_ucx_rejects_tampered_payload(monkeypatch, part):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    buffers = await _frame(np.arange(8))
    tampered = bytearray(buffers[part])
    tampered[-1] ^= 1
    buffers[part] = tampered
    endpoint = _FakeUCXEndpoint(buffers)
    with pytest.raises(ChannelClosed, match="authentication"):
        await UCXChannel(endpoint).recv()
    assert endpoint.aborted


async def test_ucx_raw_buffer_authentication(monkeypatch):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    endpoint = _FakeUCXEndpoint()
    channel = UCXChannel(endpoint)
    await channel.send_buffers([b"payload"], meta={"copy": True})
    assert await channel.recv() == {"copy": True}
    received = bytearray(7)
    await channel.recv_buffers([received])
    assert received == b"payload"


@pytest.mark.parametrize("key", [None, "cluster-secret"])
async def test_ucx_raw_empty_buffers_keep_message_boundaries(monkeypatch, key):
    if key is None:
        monkeypatch.delenv("XOSCAR_AUTH_KEY", raising=False)
    else:
        monkeypatch.setenv("XOSCAR_AUTH_KEY", key)
    endpoint = _FakeUCXEndpoint()
    channel = UCXChannel(endpoint)
    await channel.send_buffers([b"", b"payload", b""], meta={"copy": True})
    await channel.recv()
    targets = [bytearray(), bytearray(7), bytearray()]
    await channel.recv_buffers(targets)
    assert targets[1] == b"payload"
    assert not endpoint.buffers


@pytest.mark.parametrize("order", ["C", "F"])
async def test_ucx_raw_buffers_preserve_contiguous_memory_order(monkeypatch, order):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    source = np.array([[1, 2, 3], [4, 5, 6]], dtype="i4", order=order)
    target = np.zeros_like(source, order=order)
    endpoint = _FakeUCXEndpoint()
    channel = UCXChannel(endpoint)
    await channel.send_buffers([source], meta={"copy": True})
    await channel.recv()
    await channel.recv_buffers([target])
    np.testing.assert_array_equal(target, source)


@pytest.mark.parametrize("order", ["C", "F"])
def test_authentication_hashes_physical_array_order(order):
    source = np.array([[1, 2, 3], [4, 5, 6]], dtype="i4", order=order)
    assert sign_buffers(b"key", [source]) == sign_buffers(
        b"key", [source.tobytes(order="A")]
    )

    class DeviceArray:
        __cuda_array_interface__ = {}

        def get(self, order="C"):
            return source.copy(order=order)

    assert sign_buffers(b"key", [DeviceArray()]) == sign_buffers(
        b"key", [source.tobytes(order="A")]
    )


async def test_ucx_rejects_tampered_raw_buffers(monkeypatch):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    endpoint = _FakeUCXEndpoint()
    channel = UCXChannel(endpoint)
    await channel.send_buffers([b"payload"], meta={"copy": True})
    assert await channel.recv() == {"copy": True}
    endpoint.buffers[0] = b"changed"
    received = bytearray(7)
    with pytest.raises(ChannelClosed, match="Raw buffer authentication"):
        await channel.recv_buffers([received])
    assert received == bytearray(7)
    assert endpoint.aborted


async def test_ucx_raw_buffers_are_bound_to_control_message(monkeypatch):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    first = _FakeUCXEndpoint()
    channel = UCXChannel(first)
    await channel.send_buffers([b"payload"], meta={"copy": "first"})
    await channel.recv()
    first_mac = first.buffers[-1]

    second = _FakeUCXEndpoint()
    channel = UCXChannel(second)
    await channel.send_buffers([b"payload"], meta={"copy": "second"})
    await channel.recv()
    second.buffers[-1] = first_mac
    with pytest.raises(ChannelClosed, match="Raw buffer authentication"):
        await channel.recv_buffers([bytearray(7)])


async def test_socket_multiple_authenticated_messages(monkeypatch):
    monkeypatch.setenv("XOSCAR_AUTH_KEY", "cluster-secret")
    received = asyncio.get_running_loop().create_future()

    async def echo(reader, writer):
        channel = SocketChannel(reader, writer)
        try:
            for _ in range(2):
                await channel.send(await channel.recv())
            received.set_result(True)
        except Exception as exc:
            received.set_exception(exc)
        finally:
            await channel.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    address = server.sockets[0].getsockname()
    reader, writer = await asyncio.open_connection(*address)
    client = SocketChannel(reader, writer)
    try:
        for data in [CustomList([1, 2]), {"hello": np.arange(8)}]:
            await client.send(data)
            result = await asyncio.wait_for(client.recv(), 5)
            assert type(result) is type(data)
            if isinstance(data, dict):
                np.testing.assert_array_equal(result["hello"], data["hello"])
            else:
                assert result == data
        await asyncio.wait_for(received, 5)
    finally:
        await client.close()
        server.close()
        await server.wait_closed()
