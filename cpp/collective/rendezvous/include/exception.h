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

#include <stdexcept>

namespace xoscar {

class XoscarError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

    XoscarError(const XoscarError &) = default;

    XoscarError &operator=(const XoscarError &) = default;

    XoscarError(XoscarError &&) = default;

    XoscarError &operator=(XoscarError &&) = default;

    ~XoscarError() override;
};

class TimeoutError : public XoscarError {
public:
    using XoscarError::XoscarError;

    TimeoutError(const TimeoutError &) = default;

    TimeoutError &operator=(const TimeoutError &) = default;

    TimeoutError(TimeoutError &&) = default;

    TimeoutError &operator=(TimeoutError &&) = default;

    ~TimeoutError() override;
};

}  // namespace xoscar
