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
#pragma once

#include <gloo/allreduce.h>
#include <gloo/context.h>
#include <gloo/math.h>
#include <gloo/types.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace xoscar {

enum class ReduceOp : std::uint8_t {
    SUM = 0,
    PRODUCT,
    MIN,
    MAX,
    BAND,  // Bitwise AND
    BOR,   // Bitwise OR
    BXOR,  // Bitwise XOR
    UNUSED,
};

typedef void (*ReduceFunc)(void *, const void *, const void *, size_t);

template <typename T>
ReduceFunc toFunction(const ReduceOp &r) {
    switch (r) {
        case ReduceOp::SUM:
            return ReduceFunc(&gloo::sum<T>);
        case ReduceOp::PRODUCT:
            return ReduceFunc(&gloo::product<T>);
        case ReduceOp::MIN:
            return ReduceFunc(&gloo::min<T>);
        case ReduceOp::MAX:
            return ReduceFunc(&gloo::max<T>);
        case ReduceOp::BAND:
            throw std::runtime_error(
                "Cannot use ReduceOp.BAND with non-integral dtype");
            break;
        case ReduceOp::BOR:
            throw std::runtime_error(
                "Cannot use ReduceOp.BOR with non-integral dtype");
            break;
        case ReduceOp::BXOR:
            throw std::runtime_error(
                "Cannot use ReduceOp.BXOR with non-integral dtype");
            break;
        case ReduceOp::UNUSED:
            break;
    }

    throw std::runtime_error("Unhandled ReduceOp");
}

enum class glooDataType_t : std::uint8_t {
    glooInt8 = 0,
    glooUint8,
    glooInt32,
    glooUint32,
    glooInt64,
    glooUint64,
    glooFloat16,
    glooFloat32,
    glooFloat64,
};

void allreduce_wrapper(const std::shared_ptr<gloo::Context> &context,
                       intptr_t sendbuf,
                       intptr_t recvbuf,
                       size_t size,
                       glooDataType_t datatype,
                       ReduceOp reduceop = ReduceOp::SUM,
                       gloo::AllreduceOptions::Algorithm algorithm
                       = gloo::AllreduceOptions::Algorithm::RING,
                       uint32_t tag = 0);

void allgather_wrapper(const std::shared_ptr<gloo::Context> &context,
                       intptr_t sendbuf,
                       intptr_t recvbuf,
                       size_t size,
                       glooDataType_t datatype,
                       uint32_t tag = 0);

void allgatherv_wrapper(const std::shared_ptr<gloo::Context> &context,
                        intptr_t sendbuf,
                        intptr_t recvbuf,
                        size_t size,
                        glooDataType_t datatype,
                        uint32_t tag = 0);

void reduce_wrapper(const std::shared_ptr<gloo::Context> &context,
                    intptr_t sendbuf,
                    intptr_t recvbuf,
                    size_t size,
                    glooDataType_t datatype,
                    ReduceOp reduceop = xoscar::ReduceOp::SUM,
                    int root = 0,
                    uint32_t tag = 0);

void scatter_wrapper(const std::shared_ptr<gloo::Context> &context,
                     std::vector<intptr_t> sendbuf,
                     intptr_t recvbuf,
                     size_t size,
                     glooDataType_t datatype,
                     int root = 0,
                     uint32_t tag = 0);

void gather_wrapper(const std::shared_ptr<gloo::Context> &context,
                    intptr_t sendbuf,
                    intptr_t recvbuf,
                    size_t size,
                    glooDataType_t datatype,
                    int root = 0,
                    uint32_t tag = 0);

void send_wrapper(const std::shared_ptr<gloo::Context> &context,
                  intptr_t sendbuf,
                  size_t size,
                  glooDataType_t datatype,
                  int peer,
                  uint32_t tag = 0);

void recv_wrapper(const std::shared_ptr<gloo::Context> &context,
                  intptr_t recvbuf,
                  size_t size,
                  glooDataType_t datatype,
                  int peer,
                  uint32_t tag = 0);

void broadcast_wrapper(const std::shared_ptr<gloo::Context> &context,
                       intptr_t sendbuf,
                       intptr_t recvbuf,
                       size_t size,
                       glooDataType_t datatype,
                       int root = 0,
                       uint32_t tag = 0);

void reduce_scatter_wrapper(const std::shared_ptr<gloo::Context> &context,
                            intptr_t sendbuf,
                            intptr_t recvbuf,
                            size_t size,
                            std::vector<int> recvElems,
                            glooDataType_t datatype,
                            ReduceOp reduceop = xoscar::ReduceOp::SUM);

void barrier(const std::shared_ptr<gloo::Context> &context, uint32_t tag = 0);

void all_to_all_wrapper(const std::shared_ptr<gloo::Context> &context,
                        intptr_t sendbuf,
                        intptr_t recvbuf,
                        size_t size,
                        glooDataType_t datatype,
                        uint32_t tag);
}  // namespace xoscar
