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

import argparse
import asyncio
import logging
import multiprocessing
import sys
from typing import List, Optional

from .api import create_actor_pool
from .utils import ensure_coverage, get_next_port

logger = logging.getLogger(__name__)
ensure_coverage()


class CommandRunner:
    def __init__(self):
        self._description = "Xoscar command line"
        self._args = None
        self._endpoint = None
        self._pool = None

    def config_args(self, parser):
        parser.add_argument("-e", "--endpoint", help="endpoint of the xoscar")
        parser.add_argument("--n-cpu", help="num of CPU to use")
        parser.add_argument("--cuda-devices", help="cuda devices to use")
        parser.add_argument("--labels", help="label for each pool, separated by commas")
        parser.add_argument(
            "--envs",
            help="environments for each pool,"
            " separated by commas between variables, "
            "separated by semicolons between pools",
        )
        parser.add_argument("--ports", help="port for each pool, separated by commas")
        parser.add_argument(
            "--address-schemes",
            help="address scheme for each pool, "
            "separated by commas, only ucx is supported now.",
        )
        parser.add_argument(
            "--start-method", help="method to start subprocess, forkserver or spawn"
        )
        parser.add_argument(
            "--auto-recover",
            help="if recover automatically when actors down, 1 for True, 0 for False",
        )
        parser.add_argument("--modules", help="modules to import after creating pools")
        parser.add_argument(
            "--use-uvloop", help="use uvloop, 'auto' by default. Use 'no' to disable"
        )

    def parse_args(self, parser, argv):
        kwargs = dict()
        self._args = args = parser.parse_args(argv)

        if args.endpoint is not None:
            kwargs["address"] = self._endpoint = args.endpoint
        else:
            default_host = (
                "0.0.0.0" if not sys.platform.startswith("win") else "127.0.0.1"
            )
            port = str(get_next_port())
            kwargs["address"] = self._endpoint = ":".join([default_host, port])
        if args.n_cpu is not None:
            n_cpu = int(args.n_cpu)
        else:
            n_cpu = multiprocessing.cpu_count()
        if args.cuda_devices is not None:  # pragma: no cover
            cuda_devices = [int(i) for i in args.cuda_devices.split(",")]
        else:
            cuda_devices = []
        kwargs["n_process"] = n_process = n_cpu + len(cuda_devices)
        if args.labels is not None:
            kwargs["labels"] = ["main"] + args.labels.split(",")
        if args.ports is not None:  # pragma: no cover
            kwargs["ports"] = [int(i) for i in args.ports.split(",")]
        if args.envs is not None:
            envs = [
                dict(env.split("=") for env in e.split(","))
                for e in args.envs.split(";")
            ]
        else:
            envs = [[]] * n_process
        if cuda_devices:  # pragma: no cover
            for env in envs[:n_cpu]:
                env.update({"CUDA_VISIBLE_DEVICES": -1})
            for i, env in enumerate(envs[n_cpu:]):
                env.update({"CUDA_VISIBLE_DEVICES": cuda_devices[i]})
        if args.envs or cuda_devices:
            kwargs["envs"] = envs
        if args.address_schemes is not None:
            kwargs["external_address_schemes"] = args.address_schemes.split(",")
        if args.start_method is not None:
            kwargs["subprocess_start_method"] = args.start_method
        if args.auto_recover is not None:
            kwargs["auto_recover"] = bool(int(args.auto_recover))
        if args.modules is not None:  # pragma: no cover
            kwargs["modules"] = args.modules.split(",")
        if args.use_uvloop is not None:
            if args.use_uvloop == "no":
                kwargs["use_uvloop"] = False
        else:
            args.use_uvloop = "auto"
        return kwargs

    def create_loop(self):  # pragma: no cover
        use_uvloop = self._args.use_uvloop
        if use_uvloop and use_uvloop in ("0", "no"):
            loop = asyncio.get_event_loop()
        else:
            try:
                import uvloop

                loop = uvloop.new_event_loop()
                asyncio.set_event_loop(loop)
            except ImportError:
                if use_uvloop == "auto":
                    loop = asyncio.get_event_loop()
                else:  # pragma: no cover
                    raise
        return loop

    async def _main(self, **kwargs):  # pragma: no cover
        try:
            self._pool = pool = await create_actor_pool(**kwargs)
            await pool.join()
        except asyncio.CancelledError:
            if self._pool:  # pragma: no branch
                await self._pool.stop()

    def run(self, argv: Optional[List[str]] = None):  # pragma: no cover
        parser = argparse.ArgumentParser(description=self._description)
        self.config_args(parser)
        create_pool_kwargs = self.parse_args(parser, argv)

        loop = self.create_loop()
        task = loop.create_task(self._main(**create_pool_kwargs))
        try:
            logger.warning("Xoscar worker started at %s", self._endpoint)
            loop.run_until_complete(task)
        except KeyboardInterrupt:
            task.cancel()
            loop.run_until_complete(task)
            # avoid displaying exception-unhandled warnings
            task.exception()


if __name__ == "__main__":  # pragma: no cover
    runner = CommandRunner()
    runner.run()
