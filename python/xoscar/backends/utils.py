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

from __future__ import annotations


def get_proxy(proxy_map: dict[str, str], from_addr: str) -> str | None:
    host = from_addr.split(":", 1)[0]

    if addr := proxy_map.get(from_addr):
        return addr
    elif addr := proxy_map.get(host):
        # host match
        return addr
    elif addr := proxy_map.get("*"):
        # wildcard that matches all addresses
        return addr
    else:
        return None


def get_proxies(proxy_map: dict[str, str], from_addr: str) -> list[str] | None:
    """
    Get all proxies

    e.g. Proxy mapping {'a': 'b', 'b': 'c'}
    get_proxies('a') will return ['b', 'c']
    """
    proxies: list[str] = []
    while True:
        proxy = get_proxy(proxy_map, from_addr)
        if not proxies and not proxy:
            return None
        elif not proxy:
            return proxies
        else:
            proxies.append(proxy)
            from_addr = proxy
