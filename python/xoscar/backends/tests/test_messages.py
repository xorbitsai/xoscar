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

from ..._utils import create_actor_ref
from ...serialization import deserialize, serialize
from ..message import ForwardMessage, SendMessage, new_message_id


def test_serial_forward_message():
    send_message = SendMessage(
        message_id=new_message_id(),
        actor_ref=create_actor_ref("127.0.0.1:1111", "MyActor"),
        content="sth",
    )
    forward_message = ForwardMessage(
        message_id=new_message_id(),
        address="127.0.0.1:1111",
        forward_from=["127.0.0.1:1112"],
        raw_message=send_message,
    )

    forward_message2 = deserialize(*serialize(forward_message))
    assert id(forward_message) != id(forward_message2)
    assert forward_message.message_id == forward_message2.message_id
    assert forward_message.address == forward_message2.address
    assert forward_message.forward_from == forward_message2.forward_from
    assert id(forward_message.raw_message) != id(forward_message2.raw_message)
    assert (
        forward_message.raw_message.actor_ref == forward_message2.raw_message.actor_ref
    )
    assert forward_message.raw_message.content == forward_message2.raw_message.content
