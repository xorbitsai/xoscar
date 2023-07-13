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
#include <gloo/gather.h>

namespace xoscar {

template <typename T>
void gather(const std::shared_ptr<gloo::Context> &context,
            intptr_t sendbuf,
            intptr_t recvbuf,
            size_t size,
            int root,
            uint32_t tag) {
    // Configure GatherOptions struct
    gloo::GatherOptions opts_(context);

    T *input_ptr = reinterpret_cast<T *>(sendbuf);
    opts_.setInput(input_ptr, size);

    if (root == context->rank) {
        T *output_ptr = reinterpret_cast<T *>(recvbuf);
        opts_.setOutput(output_ptr, context->size * size);
    }
    opts_.setRoot(root);
    opts_.setTag(tag);

    gloo::gather(opts_);
}

void gather_wrapper(const std::shared_ptr<gloo::Context> &context,
                    intptr_t sendbuf,
                    intptr_t recvbuf,
                    size_t size,
                    glooDataType_t datatype,
                    int root,
                    uint32_t tag) {
    switch (datatype) {
        case glooDataType_t::glooInt8:
            gather<int8_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooUint8:
            gather<uint8_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooInt32:
            gather<int32_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooUint32:
            gather<uint32_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooInt64:
            gather<int64_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooUint64:
            gather<uint64_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooFloat16:
            gather<gloo::float16>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooFloat32:
            gather<float_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooFloat64:
            gather<double_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        default:
            throw std::runtime_error("Unhandled dataType");
    }
}
}  // namespace xoscar
