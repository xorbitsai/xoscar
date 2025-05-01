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
#include "socket.h"

#include "error.h"
#include "exception.h"
#include "fmt/chrono.h"
#include "fmt/ranges.h"

#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#    include <call_once.h>
#    include <mutex>
#    include <winsock2.h>
#    include <ws2tcpip.h>
#else
#    include <fcntl.h>
#    include <netdb.h>
#    include <netinet/tcp.h>
#    include <poll.h>
#    include <sys/socket.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

namespace xoscar {
namespace detail {
namespace {
#ifdef _WIN32

// Since Winsock uses the name `WSAPoll` instead of `poll`, we alias it here
// to avoid #ifdefs in the source code.
const auto pollFd = ::WSAPoll;

// Winsock's `getsockopt()` and `setsockopt()` functions expect option values to
// be passed as `char*` instead of `void*`. We wrap them here to avoid redundant
// casts in the source code.
int getSocketOption(
    SOCKET s, int level, int optname, void *optval, int *optlen) {
    return ::getsockopt(s, level, optname, static_cast<char *>(optval), optlen);
}

int setSocketOption(
    SOCKET s, int level, int optname, const void *optval, int optlen) {
    return ::setsockopt(
        s, level, optname, static_cast<const char *>(optval), optlen);
}

// Winsock has its own error codes which differ from Berkeley's. Fortunately the
// C++ Standard Library on Windows can map them to standard error codes.
inline std::error_code getSocketError() noexcept {
    return std::error_code{::WSAGetLastError(), std::system_category()};
}

inline void setSocketError(int val) noexcept { ::WSASetLastError(val); }

#else

const auto pollFd = ::poll;

const auto getSocketOption = ::getsockopt;
const auto setSocketOption = ::setsockopt;

inline std::error_code getSocketError() noexcept { return lastError(); }

inline void setSocketError(int val) noexcept { errno = val; }

#endif

// Suspends the current thread for the specified duration.
void delay(std::chrono::seconds d) {
#ifdef _WIN32
    std::this_thread::sleep_for(d);
#else
    ::timespec req{};
    req.tv_sec = d.count();

    // The C++ Standard does not specify whether `sleep_for()` should be signal-
    // aware; therefore, we use the `nanosleep()` syscall.
    if (::nanosleep(&req, nullptr) != 0) {
        std::error_code err = getSocketError();
        // We don't care about error conditions other than EINTR since a failure
        // here is not critical.
        if (err == std::errc::interrupted) {
            throw std::system_error{err};
        }
    }
#endif
}

class SocketListenOp;
class SocketConnectOp;
}  // namespace

class SocketImpl {
    friend class SocketListenOp;
    friend class SocketConnectOp;

public:
#ifdef _WIN32
    using Handle = SOCKET;
#else
    using Handle = int;
#endif

#ifdef _WIN32
    static constexpr Handle invalid_socket = INVALID_SOCKET;
#else
    static constexpr Handle invalid_socket = -1;
#endif

    explicit SocketImpl(Handle hnd) noexcept : hnd_{hnd} {}

    SocketImpl(const SocketImpl &other) = delete;

    SocketImpl &operator=(const SocketImpl &other) = delete;

    SocketImpl(SocketImpl &&other) noexcept = delete;

    SocketImpl &operator=(SocketImpl &&other) noexcept = delete;

    ~SocketImpl();

    std::unique_ptr<SocketImpl> accept() const;

    void closeOnExec() noexcept;

    void enableNonBlocking();

    void disableNonBlocking();

    bool enableNoDelay() noexcept;

    bool enableDualStack() noexcept;

#ifndef _WIN32
    bool enableAddressReuse() noexcept;
#endif

#ifdef _WIN32
    bool enableExclusiveAddressUse() noexcept;
#endif

    std::uint16_t getPort() const;

    Handle handle() const noexcept { return hnd_; }

private:
    bool setSocketFlag(int level, int optname, bool value) noexcept;

    Handle hnd_;
};
}  // namespace detail
}  // namespace xoscar

//
// libfmt formatters for `addrinfo` and `Socket`
//
namespace fmt {

template <>
struct formatter<::addrinfo> {
    constexpr decltype(auto) parse(format_parse_context &ctx) const {
        return ctx.begin();
    }

    template <typename FormatContext>
    decltype(auto) format(const ::addrinfo &addr, FormatContext &ctx) const {
        char host[NI_MAXHOST], port[NI_MAXSERV];  // NOLINT

        int r = ::getnameinfo(addr.ai_addr,
                              addr.ai_addrlen,
                              host,
                              NI_MAXHOST,
                              port,
                              NI_MAXSERV,
                              NI_NUMERICSERV);
        if (r != 0) {
            return format_to(ctx.out(), "?UNKNOWN?");
        }

        if (addr.ai_addr->sa_family == AF_INET) {
            return format_to(ctx.out(), "{}:{}", host, port);
        } else {
            return format_to(ctx.out(), "[{}]:{}", host, port);
        }
    }
};

template <>
struct formatter<xoscar::detail::SocketImpl> {
    constexpr decltype(auto) parse(format_parse_context &ctx) const {
        return ctx.begin();
    }

    template <typename FormatContext>
    decltype(auto) format(const xoscar::detail::SocketImpl &socket,
                          FormatContext &ctx) const {
        ::sockaddr_storage addr_s{};

        auto addr_ptr = reinterpret_cast<::sockaddr *>(&addr_s);

        ::socklen_t addr_len = sizeof(addr_s);

        if (::getsockname(socket.handle(), addr_ptr, &addr_len) != 0) {
            return format_to(ctx.out(), "?UNKNOWN?");
        }

        ::addrinfo addr{};
        addr.ai_addr = addr_ptr;
        addr.ai_addrlen = addr_len;

        return format_to(ctx.out(), "{}", addr);
    }
};

}  // namespace fmt

namespace xoscar {
namespace detail {

SocketImpl::~SocketImpl() {
#ifdef _WIN32
    ::closesocket(hnd_);
#else
    ::close(hnd_);
#endif
}

std::unique_ptr<SocketImpl> SocketImpl::accept() const {
    ::sockaddr_storage addr_s{};

    auto addr_ptr = reinterpret_cast<::sockaddr *>(&addr_s);

    ::socklen_t addr_len = sizeof(addr_s);

    Handle hnd = ::accept(hnd_, addr_ptr, &addr_len);
    if (hnd == invalid_socket) {
        std::error_code err = getSocketError();
        if (err == std::errc::interrupted) {
            throw std::system_error{err};
        }

        std::string msg{};
        if (err == std::errc::invalid_argument) {
            msg = fmt::format(
                "The server socket on {} is not listening for connections.",
                *this);
        } else {
            msg = fmt::format(
                "The server socket on {} has failed to accept a connection {}.",
                *this,
                err);
        }

        //    xoscar_ERROR(msg);

        throw SocketError{msg};
    }

    ::addrinfo addr{};
    addr.ai_addr = addr_ptr;
    addr.ai_addrlen = addr_len;

    //  xoscar_DEBUG(
    //      "The server socket on {} has accepted a connection from {}.",
    //      *this,
    //      addr);

    auto impl = std::make_unique<SocketImpl>(hnd);

    // Make sure that we do not "leak" our file descriptors to child processes.
    impl->closeOnExec();

    if (!impl->enableNoDelay()) {
        //    xoscar_WARNING(
        //        "The no-delay option cannot be enabled for the client socket
        //        on
        //        {}.", addr);
    }

    return impl;
}

void SocketImpl::closeOnExec() noexcept {
#ifndef _WIN32
    ::fcntl(hnd_, F_SETFD, FD_CLOEXEC);
#endif
}

void SocketImpl::enableNonBlocking() {
#ifdef _WIN32
    unsigned long value = 1;
    if (::ioctlsocket(hnd_, FIONBIO, &value) == 0) {
        return;
    }
#else
    int flg = ::fcntl(hnd_, F_GETFL);
    if (flg != -1) {
        if (::fcntl(hnd_, F_SETFL, flg | O_NONBLOCK) == 0) {
            return;
        }
    }
#endif
    throw SocketError{"The socket cannot be switched to non-blocking mode."};
}

// TODO: Remove once we migrate everything to non-blocking mode.
void SocketImpl::disableNonBlocking() {
#ifdef _WIN32
    unsigned long value = 0;
    if (::ioctlsocket(hnd_, FIONBIO, &value) == 0) {
        return;
    }
#else
    int flg = ::fcntl(hnd_, F_GETFL);
    if (flg != -1) {
        if (::fcntl(hnd_, F_SETFL, flg & ~O_NONBLOCK) == 0) {
            return;
        }
    }
#endif
    throw SocketError{"The socket cannot be switched to blocking mode."};
}

bool SocketImpl::enableNoDelay() noexcept {
    return setSocketFlag(IPPROTO_TCP, TCP_NODELAY, true);
}

bool SocketImpl::enableDualStack() noexcept {
    return setSocketFlag(IPPROTO_IPV6, IPV6_V6ONLY, false);
}

#ifndef _WIN32
bool SocketImpl::enableAddressReuse() noexcept {
    return setSocketFlag(SOL_SOCKET, SO_REUSEADDR, true);
}
#endif

#ifdef _WIN32
bool SocketImpl::enableExclusiveAddressUse() noexcept {
    return setSocketFlag(SOL_SOCKET, SO_EXCLUSIVEADDRUSE, true);
}
#endif

std::uint16_t SocketImpl::getPort() const {
    ::sockaddr_storage addr_s{};

    ::socklen_t addr_len = sizeof(addr_s);

    if (::getsockname(hnd_, reinterpret_cast<::sockaddr *>(&addr_s), &addr_len)
        != 0) {
        throw SocketError{"The port number of the socket cannot be retrieved."};
    }

    if (addr_s.ss_family == AF_INET) {
        return ntohs(reinterpret_cast<::sockaddr_in *>(&addr_s)->sin_port);
    } else {
        return ntohs(reinterpret_cast<::sockaddr_in6 *>(&addr_s)->sin6_port);
    }
}

bool SocketImpl::setSocketFlag(int level, int optname, bool value) noexcept {
#ifdef _WIN32
    auto buf = value ? TRUE : FALSE;
#else
    auto buf = value ? 1 : 0;
#endif
    return setSocketOption(hnd_, level, optname, &buf, sizeof(buf)) == 0;
}

namespace {

struct addrinfo_delete {
    void operator()(::addrinfo *addr) const noexcept { ::freeaddrinfo(addr); }
};

using addrinfo_ptr = std::unique_ptr<::addrinfo, addrinfo_delete>;

class SocketListenOp {
public:
    SocketListenOp(std::uint16_t port, const SocketOptions &opts);

    std::unique_ptr<SocketImpl> run();

private:
    bool tryListen(int family);

    bool tryListen(const ::addrinfo &addr);

    template <typename... Args>
    void recordError(fmt::string_view format, Args &&...args) {
        auto msg = fmt::vformat(format, fmt::make_format_args(args...));

        //    xoscar_WARNING(msg);

        errors_.emplace_back(std::move(msg));
    }

    std::string port_;
    const SocketOptions *opts_;
    std::vector<std::string> errors_{};
    std::unique_ptr<SocketImpl> socket_{};
};

SocketListenOp::SocketListenOp(std::uint16_t port, const SocketOptions &opts)
    : port_{fmt::to_string(port)}, opts_{&opts} {}

std::unique_ptr<SocketImpl> SocketListenOp::run() {
    if (opts_->prefer_ipv6()) {
        //    xoscar_DEBUG("The server socket will attempt to listen on an IPv6
        //    address.");
        if (tryListen(AF_INET6)) {
            return std::move(socket_);
        }

        //    xoscar_DEBUG("The server socket will attempt to listen on an IPv4
        //    address.");
        if (tryListen(AF_INET)) {
            return std::move(socket_);
        }
    } else {
        //    xoscar_DEBUG(
        //        "The server socket will attempt to listen on an IPv4 or IPv6
        //        address.");
        if (tryListen(AF_UNSPEC)) {
            return std::move(socket_);
        }
    }

    constexpr auto *msg = "The server socket has failed to listen on any local "
                          "network address.";

    //  xoscar_ERROR(msg);

    throw SocketError{fmt::format("{} {}", msg, fmt::join(errors_, " "))};
}

bool SocketListenOp::tryListen(int family) {
    ::addrinfo hints{}, *naked_result = nullptr;

    hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
    hints.ai_family = family;
    hints.ai_socktype = SOCK_STREAM;

    int r = ::getaddrinfo(nullptr, port_.c_str(), &hints, &naked_result);
    if (r != 0) {
        const char *gai_err = ::gai_strerror(r);

        recordError(
            "The local {}network addresses cannot be retrieved (gai error: "
            "{} - {}).",
            family == AF_INET    ? "IPv4 "
            : family == AF_INET6 ? "IPv6 "
                                 : "",
            r,
            gai_err);

        return false;
    }

    addrinfo_ptr result{naked_result};

    for (::addrinfo *addr = naked_result; addr != nullptr;
         addr = addr->ai_next) {
        //    xoscar_DEBUG("The server socket is attempting to listen on {}.",
        //    *addr);
        if (tryListen(*addr)) {
            return true;
        }
    }

    return false;
}

bool SocketListenOp::tryListen(const ::addrinfo &addr) {
    SocketImpl::Handle hnd
        = ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
    if (hnd == SocketImpl::invalid_socket) {
        recordError("The server socket cannot be initialized on {} {}.",
                    addr,
                    getSocketError());

        return false;
    }

    socket_ = std::make_unique<SocketImpl>(hnd);

#ifndef _WIN32
    if (!socket_->enableAddressReuse()) {
        //    xoscar_WARNING(
        //        "The address reuse option cannot be enabled for the server
        //        socket on {}.", addr);
    }
#endif

#ifdef _WIN32
    // The SO_REUSEADDR flag has a significantly different behavior on Windows
    // compared to Unix-like systems. It allows two or more processes to share
    // the same port simultaneously, which is totally unsafe.
    //
    // Here we follow the recommendation of Microsoft and use the non-standard
    // SO_EXCLUSIVEADDRUSE flag instead.
    if (!socket_->enableExclusiveAddressUse()) {
        // xoscar_WARNING("The exclusive address use option cannot be enabled
        // for the "
        //     "server socket on {}.",
        //     addr);
    }
#endif

    // Not all operating systems support dual-stack sockets by default. Since we
    // wish to use our IPv6 socket for IPv4 communication as well, we explicitly
    // ask the system to enable it.
    if (addr.ai_family == AF_INET6 && !socket_->enableDualStack()) {
        //    xoscar_WARNING(
        //        "The server socket does not support IPv4 communication on
        //        {}.", addr);
    }

    if (::bind(socket_->handle(), addr.ai_addr, addr.ai_addrlen) != 0) {
        recordError("The server socket has failed to bind to {} {}.",
                    addr,
                    getSocketError());

        return false;
    }

    // NOLINTNEXTLINE(bugprone-argument-comment)
    if (::listen(socket_->handle(), /*backlog=*/2048) != 0) {
        recordError("The server socket has failed to listen on {} {}.",
                    addr,
                    getSocketError());

        return false;
    }

    socket_->closeOnExec();

    //  xoscar_INFO("The server socket has started to listen on {}.", addr);

    return true;
}

class SocketConnectOp {
    using Clock = std::chrono::steady_clock;
    using Duration = std::chrono::steady_clock::duration;
    using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

    static const std::chrono::seconds delay_duration_;

    enum class ConnectResult { Success, Error, Retry };

public:
    SocketConnectOp(const std::string &host,
                    std::uint16_t port,
                    const SocketOptions &opts);

    std::unique_ptr<SocketImpl> run();

private:
    bool tryConnect(int family);

    ConnectResult tryConnect(const ::addrinfo &addr);

    ConnectResult tryConnectCore(const ::addrinfo &addr);

    [[noreturn]] void throwTimeoutError() const;

    template <typename... Args>
    void recordError(fmt::string_view format, Args &&...args) {
        auto msg = fmt::vformat(format, fmt::make_format_args(args...));

        //    xoscar_WARNING(msg);

        errors_.emplace_back(std::move(msg));
    }

    const char *host_;
    std::string port_;
    const SocketOptions *opts_;
    TimePoint deadline_{};
    std::vector<std::string> errors_{};
    std::unique_ptr<SocketImpl> socket_{};
};

const std::chrono::seconds SocketConnectOp::delay_duration_{1};

SocketConnectOp::SocketConnectOp(const std::string &host,
                                 std::uint16_t port,
                                 const SocketOptions &opts)
    : host_{host.c_str()}, port_{fmt::to_string(port)}, opts_{&opts} {}

std::unique_ptr<SocketImpl> SocketConnectOp::run() {
    if (opts_->prefer_ipv6()) {
        //    xoscar_DEBUG(
        //        "The client socket will attempt to connect to an IPv6 address
        //        of
        //        ({}, {}).", host_, port_);

        if (tryConnect(AF_INET6)) {
            return std::move(socket_);
        }

        //    xoscar_DEBUG(
        //        "The client socket will attempt to connect to an IPv4 address
        //        of
        //        ({}, {}).", host_, port_);

        if (tryConnect(AF_INET)) {
            return std::move(socket_);
        }
    } else {
        //    xoscar_DEBUG(
        //        "The client socket will attempt to connect to an IPv4 or IPv6
        //        address of ({}, {}).", host_, port_);

        if (tryConnect(AF_UNSPEC)) {
            return std::move(socket_);
        }
    }

    auto msg = fmt::format("The client socket has failed to connect to any "
                           "network address of ({}, {}).",
                           host_,
                           port_);

    //  xoscar_ERROR(msg);

    throw SocketError{fmt::format("{} {}", msg, fmt::join(errors_, " "))};
}

bool SocketConnectOp::tryConnect(int family) {
    ::addrinfo hints{};
    hints.ai_flags = AI_V4MAPPED | AI_ALL | AI_NUMERICSERV;
    hints.ai_family = family;
    hints.ai_socktype = SOCK_STREAM;

    deadline_ = Clock::now() + opts_->connect_timeout();

    std::size_t retry_attempt = 1;

    bool retry;  // NOLINT(cppcoreguidelines-init-variables)
    do {
        retry = false;

        errors_.clear();

        ::addrinfo *naked_result = nullptr;
        // patternlint-disable cpp-dns-deps
        int r = ::getaddrinfo(host_, port_.c_str(), &hints, &naked_result);
        if (r != 0) {
            const char *gai_err = ::gai_strerror(r);

            recordError(
                "The {}network addresses of ({}, {}) cannot be retrieved "
                "(gai error: {} - {}).",
                family == AF_INET    ? "IPv4 "
                : family == AF_INET6 ? "IPv6 "
                                     : "",
                host_,
                port_,
                r,
                gai_err);
            retry = true;
        } else {
            addrinfo_ptr result{naked_result};

            for (::addrinfo *addr = naked_result; addr != nullptr;
                 addr = addr->ai_next) {
                //        xoscar_TRACE("The client socket is attempting to
                //        connect to
                //        {}.", *addr);

                ConnectResult cr = tryConnect(*addr);
                if (cr == ConnectResult::Success) {
                    return true;
                }

                if (cr == ConnectResult::Retry) {
                    retry = true;
                }
            }
        }

        if (retry) {
            if (Clock::now() < deadline_ - delay_duration_) {
                // Prevent our log output to be too noisy, warn only every 30
                // seconds.
                if (retry_attempt == 30) {
                    //          xoscar_INFO(
                    //              "No socket on ({}, {}) is listening yet,
                    //              will retry.", host_, port_);

                    retry_attempt = 0;
                }

                // Wait one second to avoid choking the server.
                delay(delay_duration_);

                retry_attempt++;
            } else {
                throwTimeoutError();
            }
        }
    } while (retry);

    return false;
}

SocketConnectOp::ConnectResult
SocketConnectOp::tryConnect(const ::addrinfo &addr) {
    if (Clock::now() >= deadline_) {
        throwTimeoutError();
    }

    SocketImpl::Handle hnd
        = ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
    if (hnd == SocketImpl::invalid_socket) {
        recordError(
            "The client socket cannot be initialized to connect to {} {}.",
            addr,
            getSocketError());

        return ConnectResult::Error;
    }

    socket_ = std::make_unique<SocketImpl>(hnd);

    socket_->enableNonBlocking();

    ConnectResult cr = tryConnectCore(addr);
    if (cr == ConnectResult::Error) {
        std::error_code err = getSocketError();
        if (err == std::errc::interrupted) {
            throw std::system_error{err};
        }

        // Retry if the server is not yet listening or if its backlog is
        // exhausted.
        if (err == std::errc::connection_refused
            || err == std::errc::connection_reset) {
            //      xoscar_TRACE(
            //          "The server socket on {} is not yet listening {}, will
            //          retry.", addr, err);

            return ConnectResult::Retry;
        } else {
            recordError(
                "The client socket has failed to connect to {} {}.", addr, err);

            return ConnectResult::Error;
        }
    }

    socket_->closeOnExec();

    // TODO: Remove once we fully migrate to non-blocking mode.
    socket_->disableNonBlocking();

    //  xoscar_INFO("The client socket has connected to {} on {}.", addr,
    //  *socket_);

    if (!socket_->enableNoDelay()) {
        //    xoscar_WARNING(
        //        "The no-delay option cannot be enabled for the client socket
        //        on
        //        {}.", *socket_);
    }

    return ConnectResult::Success;
}

SocketConnectOp::ConnectResult
SocketConnectOp::tryConnectCore(const ::addrinfo &addr) {
    int r = ::connect(socket_->handle(), addr.ai_addr, addr.ai_addrlen);
    if (r == 0) {
        return ConnectResult::Success;
    }

    std::error_code err = getSocketError();
    if (err == std::errc::already_connected) {
        return ConnectResult::Success;
    }

    if (err != std::errc::operation_in_progress
        && err != std::errc::operation_would_block) {
        return ConnectResult::Error;
    }

    Duration remaining = deadline_ - Clock::now();
    if (remaining <= Duration::zero()) {
        throwTimeoutError();
    }

    ::pollfd pfd{};
    pfd.fd = socket_->handle();
    pfd.events = POLLOUT;

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);

    r = pollFd(&pfd, 1, static_cast<int>(ms.count()));
    if (r == 0) {
        throwTimeoutError();
    }
    if (r == -1) {
        return ConnectResult::Error;
    }

    int err_code = 0;

    ::socklen_t err_len = sizeof(int);

    r = getSocketOption(
        socket_->handle(), SOL_SOCKET, SO_ERROR, &err_code, &err_len);
    if (r != 0) {
        return ConnectResult::Error;
    }

    if (err_code != 0) {
        setSocketError(err_code);

        return ConnectResult::Error;
    } else {
        return ConnectResult::Success;
    }
}

void SocketConnectOp::throwTimeoutError() const {
    auto msg = fmt::format("The client socket has timed out after {} while "
                           "trying to connect to ({}, {}).",
                           opts_->connect_timeout(),
                           host_,
                           port_);

    //  xoscar_ERROR(msg);

    throw TimeoutError{msg};
}

}  // namespace

void Socket::initialize() {
#ifdef _WIN32
    static xoscar::once_flag init_flag{};

    // All processes that call socket functions on Windows must first initialize
    // the Winsock library.
    xoscar::call_once(init_flag, []() {
        WSADATA data{};
        if (::WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            throw SocketError{"The initialization of Winsock has failed."};
        }
    });
#endif
}

Socket Socket::listen(std::uint16_t port, const SocketOptions &opts) {
    SocketListenOp op{port, opts};

    return Socket{op.run()};
}

Socket Socket::connect(const std::string &host,
                       std::uint16_t port,
                       const SocketOptions &opts) {
    SocketConnectOp op{host, port, opts};

    return Socket{op.run()};
}

Socket::Socket(Socket &&other) noexcept = default;

Socket &Socket::operator=(Socket &&other) noexcept = default;

Socket::~Socket() = default;

Socket Socket::accept() const {
    if (impl_) {
        return Socket{impl_->accept()};
    }

    throw SocketError{"The socket is not initialized."};
}

int Socket::handle() const noexcept {
    if (impl_) {
        return impl_->handle();
    }
    return SocketImpl::invalid_socket;
}

std::uint16_t Socket::port() const {
    if (impl_) {
        return impl_->getPort();
    }
    return 0;
}

Socket::Socket(std::unique_ptr<SocketImpl> &&impl) noexcept
    : impl_{std::move(impl)} {}

}  // namespace detail

SocketError::~SocketError() = default;

}  // namespace xoscar
