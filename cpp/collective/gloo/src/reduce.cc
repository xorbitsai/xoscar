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
#include <gloo/reduce.h>

namespace xoscar {

template <typename T>
void reduce(const std::shared_ptr<gloo::Context> &context,
            intptr_t sendbuf,
            intptr_t recvbuf,
            size_t size,
            ReduceOp reduceop,
            int root,
            uint32_t tag) {
    T *input_ptr = reinterpret_cast<T *>(sendbuf);

    T *output_ptr;
    if (context->rank == root)
        output_ptr = reinterpret_cast<T *>(recvbuf);
    else
        output_ptr = new T[size];

    // Configure reduceOptions struct
    gloo::ReduceOptions opts_(context);
    opts_.setInput(input_ptr, size);
    opts_.setOutput(output_ptr, size);
    gloo::ReduceOptions::Func fn = toFunction<T>(reduceop);
    opts_.setReduceFunction(fn);
    opts_.setRoot(root);
    opts_.setTag(tag);

    gloo::reduce(opts_);

    if (context->rank != root)
        delete output_ptr;
}

void reduce_wrapper(const std::shared_ptr<gloo::Context> &context,
                    intptr_t sendbuf,
                    intptr_t recvbuf,
                    size_t size,
                    glooDataType_t datatype,
                    ReduceOp reduceop,
                    int root,
                    uint32_t tag) {
    switch (datatype) {
        case glooDataType_t::glooInt8:
            reduce<int8_t>(
                context, sendbuf, recvbuf, size, reduceop, root, tag);
            break;
        case glooDataType_t::glooUint8:
            reduce<uint8_t>(
                context, sendbuf, recvbuf, size, reduceop, root, tag);
            break;
        case glooDataType_t::glooInt32:
            reduce<int32_t>(
                context, sendbuf, recvbuf, size, reduceop, root, tag);
            break;
        case glooDataType_t::glooUint32:
            reduce<uint32_t>(
                context, sendbuf, recvbuf, size, reduceop, root, tag);
            break;
        case glooDataType_t::glooInt64:
            reduce<int64_t>(
                context, sendbuf, recvbuf, size, reduceop, root, tag);
            break;
        case glooDataType_t::glooUint64:
            reduce<uint64_t>(
                context, sendbuf, recvbuf, size, reduceop, root, tag);
            break;
        case glooDataType_t::glooFloat16:
            reduce<gloo::float16>(
                context, sendbuf, recvbuf, size, reduceop, root, tag);
            break;
        case glooDataType_t::glooFloat32:
            reduce<float_t>(
                context, sendbuf, recvbuf, size, reduceop, root, tag);
            break;
        case glooDataType_t::glooFloat64:
            reduce<double_t>(
                context, sendbuf, recvbuf, size, reduceop, root, tag);
            break;
        default:
            throw std::runtime_error("Unhandled dataType");
    }
}
}  // namespace xoscar
