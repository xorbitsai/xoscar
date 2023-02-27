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
import subprocess

import pytest

from .backends.ray.communication import RayServer
from .backends.router import Router
from .utils import lazy_import

ray = lazy_import("ray")


@pytest.fixture(scope="module")
def ray_start_regular_shared(request):  # pragma: no cover
    yield from _ray_start_regular(request)


@pytest.fixture
def ray_start_regular(request):  # pragma: no cover
    yield from _ray_start_regular(request)


def _ray_start_regular(request):  # pragma: no cover
    param = getattr(request, "param", {})
    if not param.get("enable", True):
        yield
    elif ray and ray.is_initialized():
        yield
    else:
        num_cpus = param.get("num_cpus", 64)
        total_memory_mb = num_cpus * 2 * 1024**2
        try:
            try:
                job_config = ray.job_config.JobConfig(total_memory_mb=total_memory_mb)
            except TypeError:
                job_config = None
            yield ray.init(num_cpus=num_cpus, job_config=job_config)
        finally:
            ray.shutdown()
            Router.set_instance(None)
            RayServer.clear()


@pytest.fixture(scope="module")
def ray_large_cluster_shared(request):  # pragma: no cover
    yield from _ray_large_cluster(request)


@pytest.fixture
def ray_large_cluster(request):  # pragma: no cover
    yield from _ray_large_cluster(request)


def _ray_large_cluster(request):  # pragma: no cover
    param = getattr(request, "param", {})
    num_nodes = param.get("num_nodes", 3)
    num_cpus = param.get("num_cpus", 16)
    from ray.cluster_utils import Cluster

    cluster = Cluster()
    remote_nodes = []
    for i in range(num_nodes):
        remote_nodes.append(
            cluster.add_node(num_cpus=num_cpus, memory=num_cpus * 2 * 1024**3)
        )
        if len(remote_nodes) == 1:
            try:
                job_config = ray.job_config.JobConfig(
                    total_memory_mb=num_nodes * 32 * 1024**3
                )
            except TypeError:
                job_config = None
            ray.init(address=cluster.address, job_config=job_config)
    try:
        yield cluster
    finally:
        Router.set_instance(None)
        RayServer.clear()
        ray.shutdown()
        cluster.shutdown()
        if "COV_CORE_SOURCE" in os.environ:
            # Remove this when https://github.com/ray-project/ray/issues/16802 got fixed
            subprocess.check_call(["ray", "stop", "--force"])
