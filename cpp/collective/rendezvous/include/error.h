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

#include <cstring>
#include <fmt/format.h>
#include <system_error>

namespace fmt {

template <>
struct formatter<std::error_category> {
    constexpr decltype(auto) parse(format_parse_context &ctx) const {
        return ctx.begin();
    }

    template <typename FormatContext>
    decltype(auto) format(const std::error_category &cat,
                          FormatContext &ctx) const {
        if (std::strcmp(cat.name(), "generic") == 0) {
            return format_to(ctx.out(), "errno");
        } else {
            return format_to(ctx.out(), "{} error", cat.name());
        }
    }
};

template <>
struct formatter<std::error_code> {
    constexpr decltype(auto) parse(format_parse_context &ctx) const {
        return ctx.begin();
    }

    template <typename FormatContext>
    decltype(auto) format(const std::error_code &err,
                          FormatContext &ctx) const {
        return format_to(ctx.out(),
                         "({}: {} - {})",
                         err.category(),
                         err.value(),
                         err.message());
    }
};

}  // namespace fmt

namespace xoscar {
namespace detail {

inline std::error_code lastError() noexcept {
    return std::error_code{errno, std::generic_category()};
}

}  // namespace detail
}  // namespace xoscar
