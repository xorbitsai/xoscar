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
#include <config.h>
#include <gloo/context.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rendezvous.h>
#include <sstream>

namespace xoscar {
bool transport_tcp_available() { return GLOO_HAVE_TRANSPORT_TCP; }

bool transport_uv_available() { return GLOO_HAVE_TRANSPORT_UV; }
}  // namespace xoscar
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MODULE(xoscar_pygloo, m) {
    m.doc() = "binding gloo from c to python";  // optional module docstring

    m.def("transport_tcp_available",
          &xoscar::transport_tcp_available,
          "transport_tcp_available");

    m.def("transport_uv_available",
          &xoscar::transport_uv_available,
          "transport_uv_available");
    pybind11::bind_vector<std::vector<std::string>>(m, "StringVector");

    pybind11::enum_<xoscar::ReduceOp>(m, "ReduceOp", pybind11::arithmetic())
        .value("SUM", xoscar::ReduceOp::SUM)
        .value("PRODUCT", xoscar::ReduceOp::PRODUCT)
        .value("MIN", xoscar::ReduceOp::MIN)
        .value("MAX", xoscar::ReduceOp::MAX)
        .value("BAND", xoscar::ReduceOp::BAND)
        .value("BOR", xoscar::ReduceOp::BOR)
        .value("BXOR", xoscar::ReduceOp::BXOR)
        .value("UNUSED", xoscar::ReduceOp::UNUSED)
        .export_values();

    pybind11::enum_<gloo::detail::AllreduceOptionsImpl::Algorithm>(
        m, "AllreduceAlgorithm", pybind11::arithmetic())
        .value("SUM",
               gloo::detail::AllreduceOptionsImpl::Algorithm::UNSPECIFIED)
        .value("RING", gloo::detail::AllreduceOptionsImpl::Algorithm::RING)
        .value("BCUBE", gloo::detail::AllreduceOptionsImpl::Algorithm::BCUBE)
        .export_values();

    pybind11::enum_<xoscar::glooDataType_t>(
        m, "GlooDataType_t", pybind11::arithmetic())
        .value("glooInt8", xoscar::glooDataType_t::glooInt8)
        .value("glooUint8", xoscar::glooDataType_t::glooUint8)
        .value("glooInt32", xoscar::glooDataType_t::glooInt32)
        .value("glooUint32", xoscar::glooDataType_t::glooUint32)
        .value("glooInt64", xoscar::glooDataType_t::glooInt64)
        .value("glooUint64", xoscar::glooDataType_t::glooUint64)
        .value("glooFloat16", xoscar::glooDataType_t::glooFloat16)
        .value("glooFloat32", xoscar::glooDataType_t::glooFloat32)
        .value("glooFloat64", xoscar::glooDataType_t::glooFloat64)
        .export_values();

    m.def("allreduce",
          &xoscar::allreduce_wrapper,
          pybind11::arg("context") = nullptr,
          pybind11::arg("sendbuf") = nullptr,
          pybind11::arg("recvbuf") = nullptr,
          pybind11::arg("size") = nullptr,
          pybind11::arg("datatype") = nullptr,
          pybind11::arg("reduceop") = xoscar::ReduceOp::SUM,
          pybind11::arg("algorithm") = gloo::AllreduceOptions::Algorithm::RING,
          pybind11::arg("tag") = 0);

    m.def("allgather",
          &xoscar::allgather_wrapper,
          pybind11::arg("context") = nullptr,
          pybind11::arg("sendbuf") = nullptr,
          pybind11::arg("recvbuf") = nullptr,
          pybind11::arg("size") = nullptr,
          pybind11::arg("datatype") = nullptr,
          pybind11::arg("tag") = 0);
    m.def("allgatherv",
          &xoscar::allgatherv_wrapper,
          pybind11::arg("context") = nullptr,
          pybind11::arg("sendbuf") = nullptr,
          pybind11::arg("recvbuf") = nullptr,
          pybind11::arg("size") = nullptr,
          pybind11::arg("datatype") = nullptr,
          pybind11::arg("tag") = 0);

    m.def("reduce",
          &xoscar::reduce_wrapper,
          pybind11::arg("context") = nullptr,
          pybind11::arg("sendbuf") = nullptr,
          pybind11::arg("recvbuf") = nullptr,
          pybind11::arg("size") = nullptr,
          pybind11::arg("datatype") = nullptr,
          pybind11::arg("reduceop") = xoscar::ReduceOp::SUM,
          pybind11::arg("root") = 0,
          pybind11::arg("tag") = 0);

    m.def("scatter",
          &xoscar::scatter_wrapper,
          pybind11::arg("context") = nullptr,
          pybind11::arg("sendbuf") = nullptr,
          pybind11::arg("recvbuf") = nullptr,
          pybind11::arg("size") = nullptr,
          pybind11::arg("datatype") = nullptr,
          pybind11::arg("root") = 0,
          pybind11::arg("tag") = 0);

    m.def("gather",
          &xoscar::gather_wrapper,
          pybind11::arg("context") = nullptr,
          pybind11::arg("sendbuf") = nullptr,
          pybind11::arg("recvbuf") = nullptr,
          pybind11::arg("size") = nullptr,
          pybind11::arg("datatype") = nullptr,
          pybind11::arg("root") = 0,
          pybind11::arg("tag") = 0);

    m.def("send",
          &xoscar::send_wrapper,
          pybind11::arg("context") = nullptr,
          pybind11::arg("sendbuf") = nullptr,
          pybind11::arg("size") = nullptr,
          pybind11::arg("datatype") = nullptr,
          pybind11::arg("peer") = nullptr,
          pybind11::arg("tag") = 0);
    m.def("recv",
          &xoscar::recv_wrapper,
          pybind11::arg("context") = nullptr,
          pybind11::arg("recvbuf") = nullptr,
          pybind11::arg("size") = nullptr,
          pybind11::arg("datatype") = nullptr,
          pybind11::arg("peer") = nullptr,
          pybind11::arg("tag") = 0);

    m.def("broadcast",
          &xoscar::broadcast_wrapper,
          pybind11::arg("context") = nullptr,
          pybind11::arg("sendbuf") = nullptr,
          pybind11::arg("recvbuf") = nullptr,
          pybind11::arg("size") = nullptr,
          pybind11::arg("datatype") = nullptr,
          pybind11::arg("root") = 0,
          pybind11::arg("tag") = 0);
#ifdef __linux__
    m.def("reduce_scatter",
          &xoscar::reduce_scatter_wrapper,
          pybind11::arg("context") = nullptr,
          pybind11::arg("sendbuf") = nullptr,
          pybind11::arg("recvbuf") = nullptr,
          pybind11::arg("size") = nullptr,
          pybind11::arg("recvElems") = nullptr,
          pybind11::arg("datatype") = nullptr,
          pybind11::arg("reduceop") = xoscar::ReduceOp::SUM);
#endif
    m.def("all_to_all",
          &xoscar::all_to_all_wrapper,
          pybind11::arg("context") = nullptr,
          pybind11::arg("sendbuf") = nullptr,
          pybind11::arg("recvbuf") = nullptr,
          pybind11::arg("size") = nullptr,
          pybind11::arg("datatype") = nullptr,
          pybind11::arg("tag") = 0);

    m.def("barrier",
          &xoscar::barrier,
          pybind11::arg("context") = nullptr,
          pybind11::arg("tag") = 0);

    pybind11::class_<gloo::Context, std::shared_ptr<gloo::Context>>(m,
                                                                    "Context")
        .def(pybind11::init<int, int, int>(),
             pybind11::arg("rank") = nullptr,
             pybind11::arg("size") = nullptr,
             pybind11::arg("base") = 2)
        .def("getDevice", &gloo::Context::getDevice)
        .def_readonly("rank", &gloo::Context::rank)
        .def_readonly("size", &gloo::Context::size)
        .def_readwrite("base", &gloo::Context::base)
        // .def("getPair", &gloo::Context::getPair)
        .def("createUnboundBuffer", &gloo::Context::createUnboundBuffer)
        .def("nextSlot", &gloo::Context::nextSlot)
        .def("closeConnections", &gloo::Context::closeConnections)
        .def("setTimeout", &gloo::Context::setTimeout)
        .def("getTimeout", &gloo::Context::getTimeout);

    xoscar::transport::def_transport_module(m);
    xoscar::rendezvous::def_rendezvous_module(m);
}
