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
#include <gloo/alltoall.h>
#include <gloo/alltoallv.h>
#include <gloo/context.h>

namespace xoscar {

template <typename T>
void all_to_all(const std::shared_ptr<gloo::Context> &context,
                intptr_t sendbuf,
                intptr_t recvbuf,
                size_t size,
                uint32_t tag) {
    T *input_ptr = reinterpret_cast<T *>(sendbuf);
    T *output_ptr = reinterpret_cast<T *>(recvbuf);

    // Configure AlltoallOptions struct and call alltoall function
    gloo::AlltoallOptions opts_(context);
    opts_.setInput(input_ptr, size);
    opts_.setOutput(output_ptr, size);
    opts_.setTag(tag);

    gloo::alltoall(opts_);
}

void all_to_all_wrapper(const std::shared_ptr<gloo::Context> &context,
                        intptr_t sendbuf,
                        intptr_t recvbuf,
                        size_t size,
                        glooDataType_t datatype,
                        uint32_t tag) {
    switch (datatype) {
        case glooDataType_t::glooInt8:
            all_to_all<int8_t>(context, sendbuf, recvbuf, size, tag);
            break;
        case glooDataType_t::glooUint8:
            all_to_all<uint8_t>(context, sendbuf, recvbuf, size, tag);
            break;
        case glooDataType_t::glooInt32:
            all_to_all<int32_t>(context, sendbuf, recvbuf, size, tag);
            break;
        case glooDataType_t::glooUint32:
            all_to_all<uint32_t>(context, sendbuf, recvbuf, size, tag);
            break;
        case glooDataType_t::glooInt64:
            all_to_all<int64_t>(context, sendbuf, recvbuf, size, tag);
            break;
        case glooDataType_t::glooUint64:
            all_to_all<uint64_t>(context, sendbuf, recvbuf, size, tag);
            break;
        case glooDataType_t::glooFloat16:
            all_to_all<gloo::float16>(context, sendbuf, recvbuf, size, tag);
            break;
        case glooDataType_t::glooFloat32:
            all_to_all<float_t>(context, sendbuf, recvbuf, size, tag);
            break;
        case glooDataType_t::glooFloat64:
            all_to_all<double_t>(context, sendbuf, recvbuf, size, tag);
            break;
        default:
            throw std::runtime_error("Unhandled dataType");
    }
}

}  // namespace xoscar
