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
#include <netdb.h>
#include <string>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <system_error>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace xoscar {

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
                } else if (errno_local == WSAETIMEDOUT                         \
                           || errno_local == WSAEWOULDBLOCK) {                 \
                    TORCH_CHECK(false, "Socket Timeout");                      \
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
