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

#include <chrono>
#include <transport.h>

namespace xoscar {
namespace transport {

#if GLOO_HAVE_TRANSPORT_TCP
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

void def_transport_tcp_module(pybind11::module &m) {
    pybind11::module tcp = m.def_submodule("tcp", "This is a tcp module");

    tcp.def("CreateDevice", &gloo::transport::tcp::CreateDevice);

    pybind11::class_<gloo::transport::tcp::attr>(tcp, "attr")
        .def(pybind11::init<>())
        .def(pybind11::init<const char *>())
        .def_readwrite("hostname", &gloo::transport::tcp::attr::hostname)
        .def_readwrite("iface", &gloo::transport::tcp::attr::iface)
        .def_readwrite("ai_family", &gloo::transport::tcp::attr::ai_family)
        .def_readwrite("hostname", &gloo::transport::tcp::attr::hostname)
        .def_readwrite("ai_socktype", &gloo::transport::tcp::attr::ai_socktype)
        .def_readwrite("ai_protocol", &gloo::transport::tcp::attr::ai_protocol)
        .def_readwrite("ai_addr", &gloo::transport::tcp::attr::ai_addr)
        .def_readwrite("ai_addrlen", &gloo::transport::tcp::attr::ai_addrlen);

    pybind11::class_<gloo::transport::tcp::Context,
                     std::shared_ptr<gloo::transport::tcp::Context>>(tcp,
                                                                     "Context")
        .def(pybind11::init<std::shared_ptr<gloo::transport::tcp::Device>,
                            int,
                            int>())
        // .def("createPair", &gloo::transport::tcp::Context::createPair)
        .def("createUnboundBuffer",
             &gloo::transport::tcp::Context::createUnboundBuffer);

    pybind11::class_<gloo::transport::tcp::Device,
                     std::shared_ptr<gloo::transport::tcp::Device>,
                     gloo::transport::Device>(tcp, "Device")
        .def(pybind11::init<const struct gloo::transport::tcp::attr &>());
}
#else
void def_transport_tcp_module(pybind11::module &m) {
    pybind11::module tcp = m.def_submodule("tcp", "This is a tcp module");
}
#endif

#if GLOO_HAVE_TRANSPORT_UV
void def_transport_uv_module(pybind11::module &m) {
    pybind11::module uv = m.def_submodule("uv", "This is a uv module");

    uv.def("CreateDevice", &gloo::transport::uv::CreateDevice, "CreateDevice");

    pybind11::class_<gloo::transport::uv::attr>(uv, "attr")
        .def(pybind11::init<>())
        .def(pybind11::init<const char *>())
        .def_readwrite("hostname", &gloo::transport::uv::attr::hostname)
        .def_readwrite("iface", &gloo::transport::uv::attr::iface)
        .def_readwrite("ai_family", &gloo::transport::uv::attr::ai_family)
        .def_readwrite("ai_socktype", &gloo::transport::uv::attr::ai_socktype)
        .def_readwrite("ai_protocol", &gloo::transport::uv::attr::ai_protocol)
        .def_readwrite("ai_addr", &gloo::transport::uv::attr::ai_addr)
        .def_readwrite("ai_addrlen", &gloo::transport::uv::attr::ai_addrlen);

    pybind11::class_<gloo::transport::uv::Context,
                     std::shared_ptr<gloo::transport::uv::Context>>(uv,
                                                                    "Context")
        .def(pybind11::
                 init<std::shared_ptr<gloo::transport::uv::Device>, int, int>())
        .def("createUnboundBuffer",
             &gloo::transport::uv::Context::createUnboundBuffer);

    pybind11::class_<gloo::transport::uv::Device,
                     std::shared_ptr<gloo::transport::uv::Device>,
                     gloo::transport::Device>(uv, "Device")
        .def(pybind11::init<const struct gloo::transport::uv::attr &>());
}
#else
void def_transport_uv_module(pybind11::module &m) {
    pybind11::module uv = m.def_submodule("uv", "This is a uv module");
}
#endif

void def_transport_module(pybind11::module &m) {
    pybind11::module transport
        = m.def_submodule("transport", "This is a transport module");

    pybind11::class_<gloo::transport::Device,
                     std::shared_ptr<gloo::transport::Device>,
                     xoscar::transport::PyDevice>(
        transport, "Device", pybind11::module_local())
        .def("str", &gloo::transport::Device::str)
        .def("getPCIBusID", &gloo::transport::Device::getPCIBusID)
        .def("getInterfaceSpeed", &gloo::transport::Device::getInterfaceSpeed)
        .def("hasGPUDirect", &gloo::transport::Device::hasGPUDirect)
        .def("createContext", &gloo::transport::Device::createContext);

    def_transport_uv_module(transport);
    def_transport_tcp_module(transport);
}
}  // namespace transport
}  // namespace xoscar
