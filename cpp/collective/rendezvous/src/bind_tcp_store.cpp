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

#include "tcp_store.hpp"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace xoscar {
PYBIND11_MODULE(xoscar_store, m) {
    py::class_<TCPStoreOptions>(m, "TCPStoreOptions")
        .def(py::init())
        .def_readwrite("port", &TCPStoreOptions::port)
        .def_readwrite("isServer", &TCPStoreOptions::isServer)
        .def_readwrite("numWorkers", &TCPStoreOptions::numWorkers)
        .def_readwrite("waitWorkers", &TCPStoreOptions::waitWorkers)
        .def_readwrite("timeout", &TCPStoreOptions::timeout)
        .def_readwrite("multiTenant", &TCPStoreOptions::multiTenant);

    py::class_<Store>(m, "Store");

    py::class_<TCPStore, Store>(m, "TCPStore")
        .def(py::init<std::string, const TCPStoreOptions &>())
        .def("wait",
             py::overload_cast<const std::vector<std::string> &>(
                 &TCPStore::wait))
        .def("wait",
             py::overload_cast<const std::vector<std::string> &,
                               const std::chrono::milliseconds &>(
                 &TCPStore::wait))
        .def("set",
             [](TCPStore &self, const std::string &key, py::bytes &bytes) {
                 const py::buffer_info info(py::buffer(bytes).request());
                 const char *data = reinterpret_cast<const char *>(info.ptr);
                 auto length = static_cast<size_t>(info.size);
                 self.set(key, std::vector<uint8_t>(data, data + length));
             })
        .def("get", [](TCPStore &self, const std::string &key) {
            auto result = self.get(key);
            const std::string str_result(result.begin(), result.end());
            return py::bytes(str_result);
        });
}
}  // namespace xoscar
