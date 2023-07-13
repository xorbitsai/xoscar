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
#include <gloo/allreduce.h>
#include <gloo/reduce.h>

namespace xoscar {

template <typename T>
void allreduce(const std::shared_ptr<gloo::Context> &context,
               intptr_t sendbuf,
               intptr_t recvbuf,
               size_t size,
               ReduceOp reduceop,
               gloo::AllreduceOptions::Algorithm algorithm,
               uint32_t tag) {
    std::vector<T *> input_ptr{reinterpret_cast<T *>(sendbuf)};
    std::vector<T *> output_ptr{reinterpret_cast<T *>(recvbuf)};

    // Configure AllreduceOptions struct and call allreduce function
    gloo::AllreduceOptions opts_(context);
    opts_.setInputs(input_ptr, size);
    opts_.setOutputs(output_ptr, size);
    opts_.setAlgorithm(algorithm);
    gloo::ReduceOptions::Func fn = toFunction<T>(reduceop);
    opts_.setReduceFunction(fn);
    opts_.setTag(tag);

    gloo::allreduce(opts_);
}

void allreduce_wrapper(const std::shared_ptr<gloo::Context> &context,
                       intptr_t sendbuf,
                       intptr_t recvbuf,
                       size_t size,
                       glooDataType_t datatype,
                       ReduceOp reduceop,
                       gloo::AllreduceOptions::Algorithm algorithm,
                       uint32_t tag) {
    switch (datatype) {
        case glooDataType_t::glooInt8:
            allreduce<int8_t>(
                context, sendbuf, recvbuf, size, reduceop, algorithm, tag);
            break;
        case glooDataType_t::glooUint8:
            allreduce<uint8_t>(
                context, sendbuf, recvbuf, size, reduceop, algorithm, tag);
            break;
        case glooDataType_t::glooInt32:
            allreduce<int32_t>(
                context, sendbuf, recvbuf, size, reduceop, algorithm, tag);
            break;
        case glooDataType_t::glooUint32:
            allreduce<uint32_t>(
                context, sendbuf, recvbuf, size, reduceop, algorithm, tag);
            break;
        case glooDataType_t::glooInt64:
            allreduce<int64_t>(
                context, sendbuf, recvbuf, size, reduceop, algorithm, tag);
            break;
        case glooDataType_t::glooUint64:
            allreduce<uint64_t>(
                context, sendbuf, recvbuf, size, reduceop, algorithm, tag);
            break;
        case glooDataType_t::glooFloat16:
            allreduce<gloo::float16>(
                context, sendbuf, recvbuf, size, reduceop, algorithm, tag);
            break;
        case glooDataType_t::glooFloat32:
            allreduce<float_t>(
                context, sendbuf, recvbuf, size, reduceop, algorithm, tag);
            break;
        case glooDataType_t::glooFloat64:
            allreduce<double_t>(
                context, sendbuf, recvbuf, size, reduceop, algorithm, tag);
            break;
        default:
            throw std::runtime_error("Unhandled dataType");
    }
}
}  // namespace xoscar
