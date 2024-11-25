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

from typing import Dict, List, Optional, Type

from .communication import Client, get_client_type


class Router:
    """
    Router provides mapping from external address to internal address.
    """

    __slots__ = (
        "_curr_external_addresses",
        "_local_mapping",
        "_mapping",
        "_comm_config",
    )

    _instance: "Router" | None = None

    @staticmethod
    def set_instance(router: Optional["Router"]):
        # Default router is set when an actor pool started
        Router._instance = router

    @staticmethod
    def get_instance() -> "Router" | None:
        return Router._instance

    @staticmethod
    def get_instance_or_empty() -> "Router":
        return Router._instance or Router(list(), None)

    def __init__(
        self,
        external_addresses: list[str],
        local_address: str | None,
        mapping: dict[str, str] | None = None,
        comm_config: dict | None = None,
    ):
        self._curr_external_addresses = external_addresses
        self._local_mapping = dict()
        for addr in self._curr_external_addresses:
            self._local_mapping[addr] = local_address
        if mapping is None:
            mapping = dict()
        self._mapping = mapping
        self._comm_config = comm_config or dict()

    def set_mapping(self, mapping: dict[str, str]):
        self._mapping = mapping

    def get_config(self):
        return self._comm_config

    def add_router(self, router: "Router"):
        self._curr_external_addresses.extend(router._curr_external_addresses)
        self._local_mapping.update(router._local_mapping)
        self._mapping.update(router._mapping)
        self._comm_config.update(router._comm_config)

    def remove_router(self, router: "Router"):
        for external_address in router._curr_external_addresses:
            try:
                self._curr_external_addresses.remove(external_address)
            except ValueError:
                pass
        for addr in router._local_mapping:
            self._local_mapping.pop(addr, None)
        for addr in router._mapping:
            self._mapping.pop(addr, None)

    @property
    def external_address(self):
        if self._curr_external_addresses:
            return self._curr_external_addresses[0]

    def get_internal_address(self, external_address: str) -> str | None:
        try:
            # local address, use dummy address
            return self._local_mapping[external_address]
        except KeyError:
            # try to lookup inner address from address mapping
            return self._mapping.get(external_address)

    async def get_client(
        self,
        external_address: str,
        **kw,
    ) -> Client:
        address = self.get_internal_address(external_address)
        if address is None:
            # no inner address, just use external address
            address = external_address
        client_type: Type[Client] = get_client_type(address)
        client = await self._create_client(client_type, address, **kw)
        return client

    async def _create_client(
        self, client_type: Type[Client], address: str, **kw
    ) -> Client:
        config = client_type.parse_config(self._comm_config)
        if config:
            kw["config"] = config
        local_address = (
            self._curr_external_addresses[0] if self._curr_external_addresses else None
        )
        return await client_type.connect(address, local_address=local_address, **kw)

    def _get_client_type_to_addresses(
        self, external_address: str
    ) -> Dict[Type[Client], str]:
        client_type_to_addresses = dict()
        client_type_to_addresses[get_client_type(external_address)] = external_address
        if external_address in self._curr_external_addresses:  # pragma: no cover
            # local address, use dummy address
            addr = self._local_mapping.get(external_address)
            client_type = get_client_type(addr)  # type: ignore
            client_type_to_addresses[client_type] = addr  # type: ignore
        if external_address in self._mapping:
            # try to lookup inner address from address mapping
            addr = self._mapping.get(external_address)
            client_type = get_client_type(addr)  # type: ignore
            client_type_to_addresses[client_type] = addr  # type: ignore
        return client_type_to_addresses

    def get_all_client_types(self, external_address: str) -> List[Type[Client]]:
        return list(self._get_client_type_to_addresses(external_address))

    async def get_client_via_type(
        self,
        external_address: str,
        client_type: Type[Client],
        **kw,
    ) -> Client:
        client_type_to_addresses = self._get_client_type_to_addresses(external_address)
        if client_type not in client_type_to_addresses:  # pragma: no cover
            raise ValueError(
                f"Client type({client_type}) is not supported for {external_address}"
            )
        address = client_type_to_addresses[client_type]
        client = await self._create_client(client_type, address, **kw)
        return client
