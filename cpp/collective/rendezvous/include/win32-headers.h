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

#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#ifndef NOKERNEL
#    define NOKERNEL
#endif
#ifndef NOUSER
#    define NOUSER
#endif
#ifndef NOSERVICE
#    define NOSERVICE
#endif
#ifndef NOSOUND
#    define NOSOUND
#endif
#ifndef NOMCX
#    define NOMCX
#endif
#ifndef NOGDI
#    define NOGDI

#endif
#ifndef NOMSG
#    define NOMSG
#endif
#ifndef NOMB
#    define NOMB
#endif
#ifndef NOCLIPBOARD
#    define NOCLIPBOARD
#endif

// dbghelp seems to require windows.h.
// clang-format off
#include <windows.h>
#include <dbghelp.h>
// clang-format on

#undef VOID
#undef DELETE
#undef IN
#undef THIS
#undef CONST
#undef NAN
#undef UNKNOWN
#undef NONE
#undef ANY
#undef IGNORE
#undef STRICT
#undef GetObject
#undef CreateSemaphore
#undef Yield
#undef RotateRight32
#undef RotateLeft32
#undef RotateRight64
#undef RotateLeft64
