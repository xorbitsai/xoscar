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

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#    ifndef XOSCAR_LIKELY
#        define XOSCAR_LIKELY(expr)                                            \
            (__builtin_expect(static_cast<bool>(expr), 1))
#        define XOSCAR_UNLIKELY(expr)                                          \
            (__builtin_expect(static_cast<bool>(expr), 0))
#    endif
#else
#    ifndef XOSCAR_LIKELY
#        define XOSCAR_LIKELY(expr) (expr)
#        define XOSCAR_UNLIKELY(expr) (expr)
#    endif
#endif

namespace xoscar {

template <typename F, typename... args>
#if defined(__cpp_lib_is_invocable) && __cpp_lib_is_invocable >= 201703L
using invoke_result = typename std::invoke_result<F, args...>;
#else
using invoke_result = typename std::result_of<F && (args && ...)>;
#endif

template <typename F, typename... args>
using invoke_result_t = typename invoke_result<F, args...>::type;

template <typename Functor, typename... Args>
typename std::enable_if<
    std::is_member_pointer<typename std::decay<Functor>::type>::value,
    typename xoscar::invoke_result_t<Functor, Args...>>::type
invoke(Functor &&f, Args &&...args) {
    return std::mem_fn(std::forward<Functor>(f))(std::forward<Args>(args)...);
}

template <typename Functor, typename... Args>
typename std::enable_if<
    !std::is_member_pointer<typename std::decay<Functor>::type>::value,
    typename xoscar::invoke_result_t<Functor, Args...>>::type
invoke(Functor &&f, Args &&...args) {
    return std::forward<Functor>(f)(std::forward<Args>(args)...);
}

// custom xoscar call_once implementation to avoid the deadlock in
// std::call_once. The implementation here is a simplified version from folly
// and likely much much higher memory footprint.
template <typename Flag, typename F, typename... Args>
inline void call_once(Flag &flag, F &&f, Args &&...args) {
    if (XOSCAR_LIKELY(flag.test_once())) {
        return;
    }
    flag.call_once_slow(std::forward<F>(f), std::forward<Args>(args)...);
}

class once_flag {
public:
#ifndef _WIN32
    constexpr
#endif

        once_flag() noexcept = default;
    once_flag(const once_flag &) = delete;
    once_flag &operator=(const once_flag &) = delete;

private:
    template <typename Flag, typename F, typename... Args>
    friend void call_once(Flag &flag, F &&f, Args &&...args);

    template <typename F, typename... Args>
    void call_once_slow(F &&f, Args &&...args) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (init_.load(std::memory_order_relaxed)) {
            return;
        }
        invoke(f, std::forward<Args>(args)...);
        init_.store(true, std::memory_order_release);
    }

    bool test_once() { return init_.load(std::memory_order_acquire); }

    void reset_once() { init_.store(false, std::memory_order_release); }

private:
    std::mutex mutex_;
    std::atomic<bool> init_{false};
};

}  // namespace xoscar
