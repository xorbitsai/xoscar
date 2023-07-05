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

#define GLOO_VERSION_MAJOR 0
#define GLOO_VERSION_MINOR 5
#define GLOO_VERSION_PATCH 0

static_assert(GLOO_VERSION_MINOR < 100,
              "Programming error: you set a minor version that is too big.");
static_assert(GLOO_VERSION_PATCH < 100,
              "Programming error: you set a patch version that is too big.");

#define GLOO_VERSION                                                           \
    (GLOO_VERSION_MAJOR * 10000 + GLOO_VERSION_MINOR * 100 + GLOO_VERSION_PATCH)

#define GLOO_USE_CUDA 0
#define GLOO_USE_NCCL 0
#define GLOO_USE_ROCM 0
#define GLOO_USE_RCCL 0
#define GLOO_USE_REDIS 0
#define GLOO_USE_IBVERBS 0
#define GLOO_USE_MPI 0
#define GLOO_USE_AVX 0
#define GLOO_USE_LIBUV 0

#define GLOO_HAVE_TRANSPORT_TCP 1
#define GLOO_HAVE_TRANSPORT_TCP_TLS 0
#define GLOO_HAVE_TRANSPORT_IBVERBS 0
#define GLOO_HAVE_TRANSPORT_UV 0
