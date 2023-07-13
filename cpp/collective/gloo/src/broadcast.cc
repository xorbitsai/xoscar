/* Copyright 2022-2023 XProbe Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <collective.h>
#include <gloo/broadcast.h>
#include <gloo/reduce.h>

namespace xoscar {

template <typename T>
void broadcast(const std::shared_ptr<gloo::Context> &context,
               intptr_t sendbuf,
               intptr_t recvbuf,
               size_t size,
               int root,
               uint32_t tag) {
    // Configure BroadcastOptions struct and call broadcast function
    gloo::BroadcastOptions opts_(context);

    if (context->rank == root) {
        T *input_ptr = reinterpret_cast<T *>(sendbuf);
        opts_.setInput(input_ptr, size);
    }
    T *output_ptr = reinterpret_cast<T *>(recvbuf);
    opts_.setOutput(output_ptr, size);

    opts_.setRoot(root);
    opts_.setTag(tag);

    gloo::broadcast(opts_);
}

void broadcast_wrapper(const std::shared_ptr<gloo::Context> &context,
                       intptr_t sendbuf,
                       intptr_t recvbuf,
                       size_t size,
                       glooDataType_t datatype,
                       int root,
                       uint32_t tag) {
    switch (datatype) {
        case glooDataType_t::glooInt8:
            broadcast<int8_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooUint8:
            broadcast<uint8_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooInt32:
            broadcast<int32_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooUint32:
            broadcast<uint32_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooInt64:
            broadcast<int64_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooUint64:
            broadcast<uint64_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooFloat16:
            broadcast<gloo::float16>(
                context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooFloat32:
            broadcast<float_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooFloat64:
            broadcast<double_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        default:
            throw std::runtime_error("Unhandled dataType");
    }
}
}  // namespace xoscar
