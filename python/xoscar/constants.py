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
from pathlib import Path

XOSCAR_TEMP_DIR = Path(os.getenv("XOSCAR_DIR", Path.home())) / ".xoscar"

# unix socket.
XOSCAR_UNIX_SOCKET_DIR = XOSCAR_TEMP_DIR / "socket"

XOSCAR_CONNECT_TIMEOUT = 8

XOSCAR_IDLE_TIMEOUT = 30
