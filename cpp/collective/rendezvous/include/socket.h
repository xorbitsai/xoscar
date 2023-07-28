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

#include "exception.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>

#define XOSCAR_WARNING(...)

namespace xoscar {
namespace detail {

class SocketOptions {
public:
    SocketOptions &prefer_ipv6(bool value) noexcept {
        prefer_ipv6_ = value;

        return *this;
    }

    bool prefer_ipv6() const noexcept { return prefer_ipv6_; }

    SocketOptions &connect_timeout(std::chrono::seconds value) noexcept {
        connect_timeout_ = value;

        return *this;
    }

    std::chrono::seconds connect_timeout() const noexcept {
        return connect_timeout_;
    }

private:
    bool prefer_ipv6_ = true;
    std::chrono::seconds connect_timeout_{30};
};

class SocketImpl;

class Socket {
public:
    // This function initializes the underlying socket library and must be
    // called before any other socket function.
    static void initialize();

    static Socket listen(std::uint16_t port, const SocketOptions &opts = {});

    static Socket connect(const std::string &host,
                          std::uint16_t port,
                          const SocketOptions &opts = {});

    Socket() noexcept = default;

    Socket(const Socket &other) = delete;

    Socket &operator=(const Socket &other) = delete;

    Socket(Socket &&other) noexcept;

    Socket &operator=(Socket &&other) noexcept;

    ~Socket();

    Socket accept() const;

    int handle() const noexcept;

    std::uint16_t port() const;

private:
    explicit Socket(std::unique_ptr<SocketImpl> &&impl) noexcept;

    std::unique_ptr<SocketImpl> impl_;
};

}  // namespace detail

class SocketError : public XoscarError {
public:
    using XoscarError::XoscarError;

    SocketError(const SocketError &) = default;

    SocketError &operator=(const SocketError &) = default;

    SocketError(SocketError &&) = default;

    SocketError &operator=(SocketError &&) = default;

    ~SocketError() override;
};

}  // namespace xoscar
