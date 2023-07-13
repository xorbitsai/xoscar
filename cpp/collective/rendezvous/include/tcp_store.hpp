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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <functional>
#include <gloo/rendezvous/store.h>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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

using WatchKeyCallback = std::function<void(std::optional<std::string>,
                                            std::optional<std::string>)>;

struct TCPStoreOptions {
    static constexpr std::uint16_t kDefaultPort = 29500;

    std::uint16_t port = kDefaultPort;
    bool isServer = false;
    std::optional<std::size_t> numWorkers = std::nullopt;
    bool waitWorkers = true;
    std::chrono::milliseconds timeout = std::chrono::seconds(300);

    // A boolean value indicating whether multiple store instances can be
    // initialized with the same host:port pair.
    bool multiTenant = false;
};

class TCPStore : public gloo::rendezvous::Store {
public:
    static constexpr std::chrono::milliseconds kDefaultTimeout
        = std::chrono::seconds(300);
    static constexpr std::chrono::milliseconds kNoTimeout
        = std::chrono::milliseconds::zero();
    explicit TCPStore(std::string host, const TCPStoreOptions &opts = {});

    [[deprecated("Use TCPStore(host, opts) instead.")]] explicit TCPStore(
        const std::string &masterAddr,
        std::uint16_t masterPort,
        std::optional<int> numWorkers = std::nullopt,
        bool isServer = false,
        const std::chrono::milliseconds &timeout = kDefaultTimeout,
        bool waitWorkers = true);

    ~TCPStore();

    void setTCP(const std::string &key, const std::vector<uint8_t> &value);

    std::vector<uint8_t> compareSet(const std::string &key,
                                    const std::vector<uint8_t> &expectedValue,
                                    const std::vector<uint8_t> &desiredValue);

    std::vector<uint8_t> getTCP(const std::string &key);

    int64_t add(const std::string &key, int64_t value) override;

    bool deleteKey(const std::string &key);

    // NOTE: calling other TCPStore APIs inside the callback is NOT threadsafe
    // watchKey() is a blocking operation. It will register the socket on
    // TCPStoreMasterDaemon and the callback on TCPStoreWorkerDaemon. It will
    // return once it has verified the callback is registered on both background
    // threads. Only one thread can call watchKey() at a time.
    void watchKey(const std::string &key, WatchKeyCallback callback);

    bool check(const std::vector<std::string> &keys);

    int64_t getNumKeys();

    void wait(const std::vector<std::string> &keys) override;

    void wait(const std::vector<std::string> &keys,
              const std::chrono::milliseconds &timeout) override;

    void append(const std::string &key, const std::vector<uint8_t> &value);

    std::vector<std::vector<uint8_t>>
    multiGet(const std::vector<std::string> &keys);

    void multiSet(const std::vector<std::string> &keys,
                  const std::vector<std::vector<uint8_t>> &values);

    bool hasExtendedApi() const;

    // Waits for all workers to join.
    void waitForWorkers();

    // Returns the hostname used by the TCPStore.
    const std::string &getHost() const noexcept { return addr_.host; }

    // Returns the port used by the TCPStore.
    std::uint16_t getPort() const noexcept { return addr_.port; }

    void set(const std::string &key, const std::vector<char> &data) override;

    std::vector<char> get(const std::string &key) override;

protected:
    std::chrono::milliseconds timeout_;

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
