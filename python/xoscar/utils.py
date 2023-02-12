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

from types import TracebackType
from typing import (
    Optional,
    Tuple,
    Type,
)

def wrap_exception(
    exc: Exception,
    bases: Tuple[Type] = None,
    wrap_name: str = None,
    message: str = None,
    traceback: Optional[TracebackType] = None,
    attr_dict: dict = None,
) -> Exception:
    """Generate an exception wraps the cause exception."""

    def __init__(self):
        pass

    def __getattr__(self, item):
        return getattr(exc, item)

    def __str__(self):
        return message or super(type(self), self).__str__()

    traceback = traceback or exc.__traceback__
    bases = bases or ()
    attr_dict = attr_dict or {}
    attr_dict.update(
        {
            "__init__": __init__,
            "__getattr__": __getattr__,
            "__str__": __str__,
            "__wrapname__": wrap_name,
            "__wrapped__": exc,
            "__module__": type(exc).__module__,
            "__cause__": exc.__cause__,
            "__context__": exc.__context__,
            "__suppress_context__": exc.__suppress_context__,
            "args": exc.args,
        }
    )
    new_exc_type = type(type(exc).__name__, bases + (type(exc),), attr_dict)
    return new_exc_type().with_traceback(traceback)
