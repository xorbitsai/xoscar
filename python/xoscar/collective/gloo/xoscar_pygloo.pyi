import datetime
from ctypes import c_void_p
from enum import IntEnum
from typing import Callable, List, Optional

import xoscar_pygloo

class ReduceOp(IntEnum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    UNUSED = 7

class GlooDataType(IntEnum):
    glooInt8 = 0
    glooUint8 = 1
    glooInt32 = 2
    glooUint32 = 3
    glooInt64 = 4
    glooUint64 = 5
    glooFloat16 = 6
    glooFloat32 = 7
    glooFloat64 = 8

class AllreduceAlgorithm(IntEnum):
    UNSPECIFIED = 0
    RING = 1
    BCUBE = 2

def transport_tcp_available() -> bool: ...
def transport_uv_available() -> bool: ...

class Context:
    rank: Optional[int] = None
    size: Optional[int] = None
    base: Optional[int] = 2
    def getDevice(self) -> int: ...
    def createUnboundBuffer(self, ptr: c_void_p, size: int): ...
    def nextSlot(self, numToSkip: int) -> int: ...
    def closeConnections(self) -> None: ...
    def setTimeout(self, timeout: datetime.timedelta) -> None: ...
    def getTimeout(self) -> datetime.timedelta: ...

def allreduce(
    context: Optional[Context] = None,
    sendbuf: Optional[int] = None,
    recvbuf: Optional[int] = None,
    size: Optional[int] = None,
    datatype: Optional[GlooDataType] = None,
    reduceop: Optional[ReduceOp] = ReduceOp.SUM,
    algorithm: Optional[AllreduceAlgorithm] = AllreduceAlgorithm.RING,
    tag: int = 0,
) -> None: ...
def allgather(
    context: Optional[Context] = None,
    sendbuf: Optional[int] = None,
    recvbuf: Optional[int] = None,
    size: Optional[int] = None,
    datatype: Optional[GlooDataType] = None,
    tag: Optional[int] = 0,
) -> None: ...
def all_to_all(
    context: Optional[Context] = None,
    sendbuf: Optional[int] = None,
    recvbuf: Optional[int] = None,
    size: Optional[int] = None,
    datatype: Optional[GlooDataType] = None,
    tag: Optional[int] = 0,
) -> None: ...
def allgatherv(
    context: Optional[Context] = None,
    sendbuf: Optional[int] = None,
    recvbuf: Optional[int] = None,
    size: Optional[int] = None,
    datatype: Optional[GlooDataType] = None,
    tag: Optional[int] = 0,
) -> None: ...
def reduce(
    context: Optional[Context] = None,
    sendbuf: Optional[int] = None,
    recvbuf: Optional[int] = None,
    size: Optional[int] = None,
    datatype: Optional[GlooDataType] = None,
    reduceop: Optional[ReduceOp] = ReduceOp.SUM,
    root: Optional[int] = 0,
    tag: Optional[int] = 0,
) -> None: ...
def scatter(
    context: Optional[Context] = None,
    sendbuf: Optional[int] = None,
    recvbuf: Optional[int] = None,
    size: Optional[int] = None,
    datatype: Optional[GlooDataType] = None,
    root: Optional[int] = 0,
    tag: Optional[int] = 0,
) -> None: ...
def gather(
    context: Optional[Context] = None,
    sendbuf: Optional[int] = None,
    recvbuf: Optional[int] = None,
    size: Optional[int] = None,
    datatype: Optional[GlooDataType] = None,
    root: Optional[int] = 0,
    tag: Optional[int] = 0,
) -> None: ...
def send(
    context: Optional[Context] = None,
    sendbuf: Optional[int] = None,
    size: Optional[int] = None,
    datatype: Optional[GlooDataType] = None,
    peer: Optional[int] = None,
    tag: Optional[int] = 0,
) -> None: ...
def recv(
    context: Optional[Context] = None,
    recvbuf: Optional[int] = None,
    size: Optional[int] = None,
    datatype: Optional[GlooDataType] = None,
    peer: Optional[int] = None,
    tag: Optional[int] = 0,
) -> None: ...
def broadcast(
    context: Optional[Context] = None,
    sendbuf: Optional[int] = None,
    recvbuf: Optional[int] = None,
    size: Optional[int] = None,
    datatype: Optional[GlooDataType] = None,
    root: Optional[int] = 0,
    tag: Optional[int] = 0,
) -> None: ...
def reduce_scatter(
    context: Optional[Context] = None,
    sendbuf: Optional[int] = None,
    recvbuf: Optional[int] = None,
    size: Optional[int] = None,
    recvElems: Optional[List[int]] = None,
    datatype: Optional[GlooDataType] = None,
    reduceop: Optional[ReduceOp] = ReduceOp.SUM,
) -> None: ...
def barrier(context: Optional[Context] = None, tag: Optional[int] = 0) -> None: ...

class rendezvous:
    class Store:
        def set(self, key: str, data: List[str]) -> None: ...
        def get(self, key: str) -> str: ...

    class FileStore(Store):
        def __init__(self, path: str) -> None: ...

    class HashStore(Store):
        def __init__(self) -> None: ...

    class PrefixStore(Store):
        def __init__(self, prefix: str, store: rendezvous.Store) -> None: ...

    class CustomStore(Store):
        def __init__(self, real_store_py_object: object) -> None: ...
        def delKeys(self, keys: List[str]) -> None: ...

    class Context(xoscar_pygloo.Context):
        def connectFullMesh(
            self, store: rendezvous.Store, dev: transport.Device
        ) -> None: ...

class transport:
    class uv:
        pass

    class tcp:
        class Device(transport.Device):
            def __init__(self, attr: transport.tcp.attr) -> None: ...

        class Context(xoscar_pygloo.Context):
            def __init__(
                self, device: transport.tcp.Device, rank: int, size: int
            ) -> None: ...

        class attr:
            hostname: str
            iface: str
            ai_family: int
            ai_socktype: int
            ai_protocol: int
            ai_addr: object
            ai_addrlen: int
            def __init__(self, string: Optional[str] = None) -> None: ...

        def CreateDevice(self, src: transport.tcp.attr) -> transport.Device: ...

    class Device:
        def __str__(self) -> str: ...
        def getPCIBusID(self) -> Callable[[], str]: ...
        def getInterfaceSpeed(self) -> int: ...
        def hasGPUDirect(self) -> bool: ...
        def createContext(self, rank: int, size: int): ...
