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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Type

from ...utils import classproperty, implements


@dataclass
class ChannelType:
    local = 0  # for local communication
    ipc = 1  # inproc
    remote = 2  # remote


class Channel(ABC):
    """
    Channel is used to do data exchange between server and client.
    """

    __slots__ = "local_address", "dest_address", "compression"

    name: str | None = None

    def __init__(
        self,
        local_address: str | None = None,
        dest_address: str | None = None,
        compression: str | None = None,
    ):
        self.local_address = local_address
        self.dest_address = dest_address
        self.compression = compression

    @abstractmethod
    async def send(self, message: Any):
        """
        Send data to dest. There should be only one send for one recv, otherwise recv messages
        may overlap.

        Parameters
        ----------
        message:
            data that sent to dest.
        """

    @abstractmethod
    async def recv(self):
        """
        Receive data that sent from dest.
        """

    @abstractmethod
    async def close(self):
        """
        Close channel.
        """

    @property
    @abstractmethod
    def closed(self) -> bool:
        """
        This channel is closed or not.

        Returns
        -------
        closed:
            If the channel is closed.
        """

    @property
    @abstractmethod
    def type(self) -> int:
        """
        Channel is used for, can be dummy, ipc or remote.

        Returns
        -------
        channel_type: int
            type that can be dummy, ipc or remote.
        """

    @property
    def info(self) -> dict:
        return {
            "name": self.name,
            "compression": self.compression,
            "type": self.type,
            "local_address": self.local_address,
            "dest_address": self.dest_address,
        }


class Server(ABC):
    __slots__ = "address", "channel_handler"

    scheme: str | None = None

    def __init__(
        self,
        address: str,
        channel_handler: Callable[[Channel], Coroutine] | None = None,
    ):
        self.address = address
        self.channel_handler = channel_handler

    @classproperty
    @abstractmethod
    def client_type(self) -> Type["Client"]:
        """
        Return the corresponding client type.

        Returns
        -------
        client_type: type
            client type.
        """

    @property
    @abstractmethod
    def channel_type(self) -> int:
        """
        Channel type, can be dummy, ipc or remote.

        Returns
        -------
        channel_type: int
            type that can be dummy, ipc or remote.
        """

    @staticmethod
    @abstractmethod
    async def create(config: dict) -> "Server":
        """
        Create a server instance according to configuration.

        Parameters
        ----------
        config: dict
            configuration about creating a channel.

        Returns
        -------
        server: Server
            a server that waiting for connections from clients.
        """

    @abstractmethod
    async def start(self):
        """
        Used for listening to port or similar stuff.
        """

    @abstractmethod
    async def join(self, timeout=None):
        """
        Wait forever until timeout.
        """

    @abstractmethod
    async def on_connected(self, *args, **kwargs):
        """
        Return a channel when new client connected.

        Returns
        -------
        channel: Channel
            channel for communication
        """

    @abstractmethod
    async def stop(self):
        """
        Stop the server.
        """

    @property
    @abstractmethod
    def stopped(self) -> bool:
        """
        If this server is stopped or not.

        Returns
        -------
        if_stopped: bool
           This server is stopped or not.
        """

    @property
    def info(self) -> dict:
        return {
            "name": self.scheme,
            "address": self.address,
            "channel_type": self.channel_type,
        }

    @classmethod
    def parse_config(cls, config: dict) -> dict:
        # skip parsing config by default
        return dict()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class Client(ABC):
    __slots__ = "local_address", "dest_address", "channel"

    scheme: str | None = None

    def __init__(
        self, local_address: str | None, dest_address: str | None, channel: Channel
    ):
        self.local_address = local_address
        self.dest_address = dest_address
        self.channel = channel

    @property
    def channel_type(self) -> int:
        """
        Channel type, can be dummy, ipc or remote.

        Returns
        -------
        channel_type: int
            type that can be dummy, ipc or remote.
        """
        return self.channel.type

    @staticmethod
    @abstractmethod
    async def connect(
        dest_address: str, local_address: str | None = None, **kwargs
    ) -> "Client":
        """
        Create a client that is able to connect to some server.

        Parameters
        ----------
        dest_address: str
            Destination server address that to connect to.
        local_address: str
            local address.

        Returns
        -------
        client: Client
            Client that holds a channel to communicate.
        """

    @classmethod
    def parse_config(cls, config: dict) -> dict:
        # skip parsing config by default
        return dict()

    @implements(Channel.send)
    async def send(self, message):
        return await self.channel.send(message)

    @implements(Channel.recv)
    async def recv(self):
        return await self.channel.recv()

    async def close(self):
        """
        Close connection.
        """
        await self.channel.close()

    @property
    def closed(self) -> bool:
        """
        This client is closed or not.

        Returns
        -------
        closed: bool
            If the client is closed.
        """
        return self.channel.closed

    @property
    def info(self) -> dict:
        return {
            "local_address": self.local_address,
            "dest_address": self.dest_address,
            "channel_name": self.channel.name,
            "channel_type": self.channel_type,
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
