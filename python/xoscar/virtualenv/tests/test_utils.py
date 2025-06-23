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

import logging
from io import StringIO
from unittest.mock import patch

import pytest

from .. import utils


@pytest.fixture
def patched_logger():
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    mock_logger = logging.getLogger("pytest_logger")
    mock_logger.setLevel(logging.INFO)
    mock_logger.addHandler(handler)

    with patch.object(utils, "logger", mock_logger):
        yield stream

    mock_logger.removeHandler(handler)


def test_stdout_logging(patched_logger):
    stream = patched_logger
    with utils.run_subprocess_with_logger(["echo", "hello pytest"]) as p:
        pass
    stream.seek(0)
    logs = stream.read()
    assert p.returncode == 0
    assert "hello pytest" in logs


def test_stderr_logging(patched_logger):
    stream = patched_logger
    with utils.run_subprocess_with_logger(["ls", "non_existent_file"]) as p:
        pass
    stream.seek(0)
    logs = stream.read()
    assert p.returncode != 0
    assert "non_existent_file" in logs
