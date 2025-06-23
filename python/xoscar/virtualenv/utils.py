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
import re
import subprocess
import sys
import threading
from contextlib import contextmanager
from typing import BinaryIO, Callable, Iterator, List, Optional, TextIO, Union

logger = logging.getLogger(__name__)

ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def clean_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return ansi_escape.sub("", text)


def stream_reader(
    stream: BinaryIO, log_func: Callable[[str], None], output_stream: TextIO
) -> None:
    """
    Read from the stream, write to logger, and also write to the terminal.
    """
    for line in iter(stream.readline, b""):
        decoded = line.decode(errors="replace")
        output_stream.write(decoded)
        output_stream.flush()
        log_func(clean_ansi(decoded.rstrip("\n")))


@contextmanager
def run_subprocess_with_logger(
    cmd: Union[str, List[str]], cwd: Optional[str] = None, env: Optional[dict] = None
) -> Iterator[subprocess.Popen]:
    """
    Run a subprocess, redirect stdout to logger.info and stderr to logger.error.
    Returns the Popen object as a context manager.

    :param cmd: Command to execute
    :param kwargs: Additional arguments passed to subprocess.Popen
    :yield: The subprocess.Popen object
    """

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        bufsize=1,
    )

    threads = [
        threading.Thread(
            target=stream_reader, args=(process.stdout, logger.info, sys.stdout)
        ),
        threading.Thread(
            target=stream_reader, args=(process.stderr, logger.error, sys.stderr)
        ),
    ]
    for t in threads:
        t.start()

    try:
        yield process
    finally:
        process.wait()
        for t in threads:
            t.join()
