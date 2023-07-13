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
#include <gloo/scatter.h>

namespace xoscar {

template <typename T>
void scatter(const std::shared_ptr<gloo::Context> &context,
             std::vector<intptr_t> sendbuf,
             intptr_t recvbuf,
             size_t size,
             int root,
             uint32_t tag) {
    std::vector<T *> input_ptr;
    for (size_t i = 0; i < sendbuf.size(); ++i)
        input_ptr.emplace_back(reinterpret_cast<T *>(sendbuf[i]));

    T *output_ptr = reinterpret_cast<T *>(recvbuf);

    // Configure ScatterOptions struct
    gloo::ScatterOptions opts_(context);
    opts_.setInputs(input_ptr, size);
    opts_.setOutput(output_ptr, size);
    opts_.setTag(tag);
    opts_.setRoot(root);

    gloo::scatter(opts_);
}

void scatter_wrapper(const std::shared_ptr<gloo::Context> &context,
                     std::vector<intptr_t> sendbuf,
                     intptr_t recvbuf,
                     size_t size,
                     glooDataType_t datatype,
                     int root,
                     uint32_t tag) {
    switch (datatype) {
        case glooDataType_t::glooInt8:
            scatter<int8_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooUint8:
            scatter<uint8_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooInt32:
            scatter<int32_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooUint32:
            scatter<uint32_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooInt64:
            scatter<int64_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooUint64:
            scatter<uint64_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooFloat16:
            scatter<gloo::float16>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooFloat32:
            scatter<float_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        case glooDataType_t::glooFloat64:
            scatter<double_t>(context, sendbuf, recvbuf, size, root, tag);
            break;
        default:
            throw std::runtime_error("Unhandled dataType");
    }
}
}  // namespace xoscar
