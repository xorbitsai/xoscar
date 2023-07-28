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

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <functional>
#include <limits>
#ifdef _WIN32
#    include <winsock2.h>
#    include <ws2tcpip.h>
typedef SSIZE_T ssize_t;

#    pragma comment(lib, "Ws2_32.lib")
#else
#    include <netdb.h>
#    include <sys/poll.h>
#    include <sys/socket.h>
#    include <unistd.h>
#endif

#include <call_once.h>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace xoscar {

std::string StripBasename(const std::string &full_path) {
#ifdef _WIN32
    const std::string separators("/\\");
#else
    const std::string separators("/");
#endif
    size_t pos = full_path.find_last_of(separators);
    if (pos != std::string::npos) {
        return full_path.substr(pos + 1, std::string::npos);
    } else {
        return full_path;
    }
}

struct SourceLocation {
    const char *func;
    const char *file;
    uint32_t line;
};

class Error : public std::exception {
    // The actual error message.
    std::string msg_;

    // Context for the message (in order of decreasing specificity).  Context
    // will be automatically formatted appropriately, so it is not necessary to
    // add extra leading/trailing newlines to strings inside this vector
    std::vector<std::string> context_;

    // The C++ backtrace at the point when this exception was raised.  This
    // may be empty if there is no valid backtrace.  (We don't use optional
    // here to reduce the dependencies this file has.)
    std::string backtrace_;

    // These two are derived fields from msg_stack_ and backtrace_, but we need
    // fields for the strings so that we can return a const char* (as the
    // signature of std::exception requires).  Currently, the invariant
    // is that these fields are ALWAYS populated consistently with respect
    // to msg_stack_ and backtrace_.
    std::string what_;
    std::string what_without_backtrace_;

    // This is a little debugging trick: you can stash a relevant pointer
    // in caller, and then when you catch the exception, you can compare
    // against pointers you have on hand to get more information about
    // where the exception came from.  In Caffe2, this is used to figure
    // out which operator raised an exception.
    const void *caller_;

public:
    Error(SourceLocation source_location, std::string msg);

    // Caffe2-style error message
    Error(const char *file,
          const uint32_t line,
          const char *condition,
          const std::string &msg,
          const std::string &backtrace,
          const void *caller = nullptr);

    // Base constructor
    Error(std::string msg, std::string backtrace, const void *caller = nullptr);

    // Add some new context to the message stack.  The last added context
    // will be formatted at the end of the context list upon printing.
    // WARNING: This method is O(n) in the size of the stack, so don't go
    // wild adding a ridiculous amount of context to error messages.
    void add_context(std::string new_msg) {
        context_.push_back(std::move(new_msg));
        // TODO: Calling add_context O(n) times has O(n^2) cost.  We can fix
        // this perf problem by populating the fields lazily... if this ever
        // actually is a problem.
        // NB: If you do fix this, make sure you do it in a thread safe way!
        // what() is almost certainly expected to be thread safe even when
        // accessed across multiple threads
        refresh_what();
    }

    const std::string &msg() const { return msg_; }

    const std::vector<std::string> &context() const { return context_; }

    const std::string &backtrace() const { return backtrace_; }

    /// Returns the complete error message, including the source location.
    /// The returned pointer is invalidated if you call add_context() on
    /// this object.
    const char *what() const noexcept override { return what_.c_str(); }

    const void *caller() const noexcept { return caller_; }

    /// Returns only the error message string, without source location.
    /// The returned pointer is invalidated if you call add_context() on
    /// this object.
    const char *what_without_backtrace() const noexcept {
        return what_without_backtrace_.c_str();
    }

private:
    void refresh_what() {
        what_ = compute_what(/*include_backtrace*/ true);
        what_without_backtrace_ = compute_what(/*include_backtrace*/ false);
    }
    std::string compute_what(bool include_backtrace) const {
        std::ostringstream oss;

        oss << msg_;

        if (context_.size() == 1) {
            // Fold error and context in one line
            oss << " (" << context_[0] << ")";
        } else {
            for (const auto &c : context_) {
                oss << "\n  " << c;
            }
        }

        if (include_backtrace) {
            oss << "\n" << backtrace_;
        }

        return oss.str();
    }
};
Error::Error(std::string msg, std::string backtrace, const void *caller)
    : msg_(std::move(msg)), backtrace_(std::move(backtrace)), caller_(caller) {
    refresh_what();
}

std::string charToString(const char *chars) {
    std::string ret = chars;
    return ret;
}
// Caffe2-style error message
Error::Error(const char *file,
             const uint32_t line,
             const char *condition,
             const std::string &msg,
             const std::string &backtrace,
             const void *caller)
    : Error("[enforce fail at " + xoscar::StripBasename(file) + ":"
                + std::to_string(line) + "] " + charToString(condition) + ". "
                + msg,
            backtrace,
            caller) {}

Error::Error(SourceLocation source_location, std::string msg)
    : Error(std::move(msg),
            "Exception raised from " + charToString(source_location.file)
                + " (most recent call first):\n") {}

void xoscarCheckFail(const char *func,
                     const char *file,
                     uint32_t line,
                     const std::string &msg) {
    throw ::xoscar::Error({func, file, line}, msg);
}

void xoscarCheckFail(const char *func,
                     const char *file,
                     uint32_t line,
                     const char *msg) {
    throw ::xoscar::Error({func, file, line}, msg);
}

#if defined(__CUDACC__)
#    define XOSCAR_UNLIKELY_OR_CONST(e) e
#else
#    define XOSCAR_UNLIKELY_OR_CONST(e) XOSCAR_UNLIKELY(e)
#endif

template <typename T, typename... Args>
inline std::ostream &_str(std::ostream &ss, const T &t, const Args &...args) {
    return _str(_str(ss, t), args...);
}

template <typename... Args>
struct _str_wrapper final {
    static std::string call(const Args &...args) {
        std::ostringstream ss;
        _str(ss, args...);
        return ss.str();
    }
};

template <typename T>
struct CanonicalizeStrTypes {
    using type = const T &;
};

// Convert a list of string-like arguments into a single string.
template <typename... Args>
inline decltype(auto) str(const Args &...args) {
    return _str_wrapper<typename CanonicalizeStrTypes<Args>::type...>::call(
        args...);
}

template <typename... Args>
decltype(auto) xoscarCheckMsgImpl(const char * /*msg*/, const Args &...args) {
    return ::xoscar::str(args...);
}
inline const char *xoscarCheckMsgImpl(const char *msg) { return msg; }
// If there is just 1 user-provided C-string argument, use it.
inline const char *xoscarCheckMsgImpl(const char * /*msg*/, const char *args) {
    return args;
}

#define XOSCAR_CHECK_MSG(cond, type, ...)                                      \
    (::xoscar::xoscarCheckMsgImpl(                                             \
        "Expected " #cond " to be true, but got false.  "                      \
        "(Could this error message be improved?  If so, "                      \
        "please report an enhancement request to xoscar.)",                    \
        ##__VA_ARGS__))

#ifdef STRIP_ERROR_MESSAGES
#    define XOSCAR_CHECK(cond, ...)                                            \
        if (XOSCAR_UNLIKELY_OR_CONST(!(cond))) {                               \
            ::xoscar::xoscarCheckFail(                                         \
                __func__,                                                      \
                __FILE__,                                                      \
                static_cast<uint32_t>(__LINE__),                               \
                XOSCAR_CHECK_MSG(cond, "", __VA_ARGS__));                      \
        }
#else
#    define XOSCAR_CHECK(cond, ...)                                            \
        if (XOSCAR_UNLIKELY_OR_CONST(!(cond))) {                               \
            ::xoscar::xoscarCheckFail(                                         \
                __func__,                                                      \
                __FILE__,                                                      \
                static_cast<uint32_t>(__LINE__),                               \
                XOSCAR_CHECK_MSG(cond, "", ##__VA_ARGS__));                    \
        }
#endif

using RankType = uint32_t;
using SizeType = uint64_t;

// `errno` is only meaningful when it fails. E.g., a  successful `fork()` sets
// `errno` to `EINVAL` in child process on some macos
// (https://stackoverflow.com/a/20295079), and thus `errno` should really only
// be inspected if an error occurred.
//
// `success_cond` is an expression used to check if an error has happend. So for
// `fork()`, we can use `SYSCHECK(pid = fork(), pid != -1)`. The function output
// is stored in variable `__output` and may be used in `success_cond`.
#ifdef _WIN32
#    define SYSCHECK(expr, success_cond)                                       \
        while (true) {                                                         \
            auto __output = (expr);                                            \
            auto errno_local = WSAGetLastError();                              \
            (void) __output;                                                   \
            if (!(success_cond)) {                                             \
                if (errno == EINTR) {                                          \
                    continue;                                                  \
                } else if (errno_local == WSAETIMEDOUT) {                      \
                    XOSCAR_CHECK(false, "Socket Timeout");                     \
                } else if (errno_local == WSAEWOULDBLOCK) {                    \
                    XOSCAR_CHECK(false, "Buffer Full");                        \
                } else {                                                       \
                    throw std::system_error(errno_local,                       \
                                            std::system_category());           \
                }                                                              \
            } else {                                                           \
                break;                                                         \
            }                                                                  \
        }
#else
#    define SYSCHECK(expr, success_cond)                                       \
        while (true) {                                                         \
            auto __output = (expr);                                            \
            (void) __output;                                                   \
            if (!(success_cond)) {                                             \
                if (errno == EINTR) {                                          \
                    continue;                                                  \
                } else if (errno == EAGAIN || errno == EWOULDBLOCK) {          \
                    throw std::runtime_error("Socket Timeout");                \
                } else {                                                       \
                    throw std::system_error(errno, std::system_category());    \
                }                                                              \
            } else {                                                           \
                break;                                                         \
            }                                                                  \
        }
#endif

// Most functions indicate error by returning `-1`. This is a helper macro for
// this common case with `SYSCHECK`.
// Since SOCKET_ERROR = -1 in MSVC, so also leverage SYSCHECK_ERR_RETURN_NEG1
#define SYSCHECK_ERR_RETURN_NEG1(expr) SYSCHECK(expr, __output != -1)

namespace tcputil {
// Send and receive
template <typename T>
void sendBytes(int socket,
               const T *buffer,
               size_t length,
               bool moreData = false) {
    size_t bytesToSend = sizeof(T) * length;
    if (bytesToSend == 0) {
        return;
    }

    auto bytes = reinterpret_cast<const uint8_t *>(buffer);
    uint8_t *currentBytes = const_cast<uint8_t *>(bytes);

    int flags = 0;

#ifdef MSG_MORE
    if (moreData) {  // there is more data to send
        flags |= MSG_MORE;
    }
#endif

// Ignore SIGPIPE as the send() return value is always checked for error
#ifdef MSG_NOSIGNAL
    flags |= MSG_NOSIGNAL;
#endif

    while (bytesToSend > 0) {
        ssize_t bytesSent;
        SYSCHECK_ERR_RETURN_NEG1(
            bytesSent
            = ::send(socket, (const char *) currentBytes, bytesToSend, flags))
        if (bytesSent == 0) {
            throw std::system_error(ECONNRESET, std::system_category());
        }

        bytesToSend -= bytesSent;
        currentBytes += bytesSent;
    }
}

template <typename T>
void recvBytes(int socket, T *buffer, size_t length) {
    size_t bytesToReceive = sizeof(T) * length;
    if (bytesToReceive == 0) {
        return;
    }

    auto bytes = reinterpret_cast<uint8_t *>(buffer);
    uint8_t *currentBytes = bytes;

    while (bytesToReceive > 0) {
        ssize_t bytesReceived;
        SYSCHECK_ERR_RETURN_NEG1(bytesReceived
                                 = recv(socket,
                                        reinterpret_cast<char *>(currentBytes),
                                        bytesToReceive,
                                        0))
        if (bytesReceived == 0) {
            throw std::system_error(ECONNRESET, std::system_category());
        }

        bytesToReceive -= bytesReceived;
        currentBytes += bytesReceived;
    }
}

// send a vector's length and data
template <typename T>
void sendVector(int socket, const std::vector<T> &vec, bool moreData = false) {
    SizeType size = vec.size();
    sendBytes<SizeType>(socket, &size, 1, true);
    sendBytes<T>(socket, vec.data(), size, moreData);
}

// receive a vector as sent in sendVector
template <typename T>
std::vector<T> recvVector(int socket) {
    SizeType valueSize;
    recvBytes<SizeType>(socket, &valueSize, 1);
    std::vector<T> value(valueSize);
    recvBytes<T>(socket, value.data(), value.size());
    return value;
}

// this is only for convenience when sending rvalues
template <typename T>
void sendValue(int socket, const T &value, bool moreData = false) {
    sendBytes<T>(socket, &value, 1, moreData);
}

template <typename T>
T recvValue(int socket) {
    T value;
    recvBytes<T>(socket, &value, 1);
    return value;
}

// send a string's length and data
inline void
sendString(int socket, const std::string &str, bool moreData = false) {
    SizeType size = str.size();
    sendBytes<SizeType>(socket, &size, 1, true);
    sendBytes<char>(socket, str.data(), size, moreData);
}

// receive a string as sent in sendString
inline std::string recvString(int socket) {
    SizeType valueSize;
    recvBytes<SizeType>(socket, &valueSize, 1);
    std::vector<char> value(valueSize);
    recvBytes<char>(socket, value.data(), value.size());
    return std::string(value.data(), value.size());
}
}  // namespace tcputil

}  // namespace xoscar
