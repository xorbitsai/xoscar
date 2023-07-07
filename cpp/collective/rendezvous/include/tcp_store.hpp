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

#include "store.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace xoscar {
namespace detail {

class TCPServer;

class TCPClient;

class TCPCallbackClient;

struct SocketAddress {
    std::string host{};
    std::uint16_t port{};
};

}  // namespace detail

struct TCPStoreOptions {
    static constexpr std::uint16_t kDefaultPort = 29500;

    std::uint16_t port = kDefaultPort;
    bool isServer = false;
    std::optional<std::size_t> numWorkers = std::nullopt;
    bool waitWorkers = true;
    std::chrono::milliseconds timeout = Store::kDefaultTimeout;

    // A boolean value indicating whether multiple store instances can be
    // initialized with the same host:port pair.
    bool multiTenant = false;
};

class TCPStore : public Store {
public:
    explicit TCPStore(std::string host, const TCPStoreOptions &opts = {});

    [[deprecated("Use TCPStore(host, opts) instead.")]] explicit TCPStore(
        const std::string &masterAddr,
        std::uint16_t masterPort,
        std::optional<int> numWorkers = std::nullopt,
        bool isServer = false,
        const std::chrono::milliseconds &timeout = kDefaultTimeout,
        bool waitWorkers = true);

    ~TCPStore();

    void set(const std::string &key,
             const std::vector<uint8_t> &value) override;

    std::vector<uint8_t>
    compareSet(const std::string &key,
               const std::vector<uint8_t> &expectedValue,
               const std::vector<uint8_t> &desiredValue) override;

    std::vector<uint8_t> get(const std::string &key) override;

    int64_t add(const std::string &key, int64_t value) override;

    bool deleteKey(const std::string &key) override;

    // NOTE: calling other TCPStore APIs inside the callback is NOT threadsafe
    // watchKey() is a blocking operation. It will register the socket on
    // TCPStoreMasterDaemon and the callback on TCPStoreWorkerDaemon. It will
    // return once it has verified the callback is registered on both background
    // threads. Only one thread can call watchKey() at a time.
    void watchKey(const std::string &key, WatchKeyCallback callback) override;

    bool check(const std::vector<std::string> &keys) override;

    int64_t getNumKeys() override;

    void wait(const std::vector<std::string> &keys) override;

    void wait(const std::vector<std::string> &keys,
              const std::chrono::milliseconds &timeout) override;

    void append(const std::string &key,
                const std::vector<uint8_t> &value) override;

    std::vector<std::vector<uint8_t>>
    multiGet(const std::vector<std::string> &keys) override;

    void multiSet(const std::vector<std::string> &keys,
                  const std::vector<std::vector<uint8_t>> &values) override;

    bool hasExtendedApi() const override;

    // Waits for all workers to join.
    void waitForWorkers();

    // Returns the hostname used by the TCPStore.
    const std::string &getHost() const noexcept { return addr_.host; }

    // Returns the port used by the TCPStore.
    std::uint16_t getPort() const noexcept { return addr_.port; }

private:
    int64_t incrementValueBy(const std::string &key, int64_t delta);

    std::vector<uint8_t> doGet(const std::string &key);

    void doWait(std::vector<std::string> keys,
                std::chrono::milliseconds timeout);

    detail::SocketAddress addr_;
    std::shared_ptr<detail::TCPServer> server_;
    std::unique_ptr<detail::TCPClient> client_;
    std::unique_ptr<detail::TCPCallbackClient> callbackClient_;
    std::optional<std::size_t> numWorkers_;

    const std::string initKey_ = "init/";
    const std::string keyPrefix_ = "/";
    std::mutex activeOpLock_;
};

}  // namespace xoscar
