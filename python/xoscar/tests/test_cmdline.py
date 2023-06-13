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
import subprocess
import sys

import psutil
import pytest

from .. import Actor, create_actor
from ..utils import get_next_port

cmd = [sys.executable, "-m", "xoscar.worker"]


class Mock(Actor):
    def f(self):
        return "mock"


def _stop_proc(proc):
    if proc is not None:
        subprocs = psutil.Process(proc.pid).children(recursive=True)
        for ps_proc in subprocs + [proc]:
            try:
                ps_proc.kill()
            except psutil.NoSuchProcess:
                pass


params = [
    ["--n-cpu", "1"],
    ["--n-cpu", "1", "--labels", "label"],
    ["--n-cpu", "1", "--envs", "a=1"],
    ["--n-cpu", "1", "--auto-recover", "1"],
    ["--n-cpu", "1", "--start-method", "spawn"],
]


@pytest.mark.parametrize("args", params)
@pytest.mark.asyncio
async def test_cmdline(args):
    proc = None
    try:
        port = get_next_port()
        endpoint = f"127.0.0.1:{port}"
        cmd.extend(["-e", endpoint])
        cmd.extend(args)
        proc = subprocess.Popen(cmd)
        retry_nums = 5
        for trial in range(retry_nums):
            await asyncio.sleep(5)
            try:
                actor_ref = await create_actor(Mock, address=endpoint, uid="mock")
                assert (await actor_ref.f()) == "mock"
                break
            except:
                if trial == retry_nums - 1:
                    raise
                else:
                    _stop_proc(proc)
    except:
        _stop_proc(proc)
        raise
