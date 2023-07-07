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

#include "utils.hpp"

#include <sys/poll.h>
#include <vector>

namespace xoscar::tcputil {

#define CONNECT_SOCKET_OFFSET 2

inline int poll(struct pollfd *fds, unsigned long nfds, int timeout) {
    return ::poll(fds, nfds, timeout);
}

inline void
addPollfd(std::vector<struct pollfd> &fds, int socket, short events) {
    fds.push_back({.fd = socket, .events = events});
}

inline struct ::pollfd getPollfd(int socket, short events) {
    struct ::pollfd res = {.fd = socket, .events = events};
    return res;
}

}  // namespace xoscar::tcputil
