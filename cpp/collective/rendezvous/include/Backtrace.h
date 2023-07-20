#ifndef C10_UTIL_BACKTRACE_H_
#define C10_UTIL_BACKTRACE_H_

#include <cstddef>
#include <string>
#include <typeinfo>


namespace xoscar {
std::string get_backtrace(
    size_t frames_to_skip = 0,
    size_t maximum_number_of_frames = 64,
    bool skip_python_frames = true);
} // namespace xoscar

#endif // C10_UTIL_BACKTRACE_H_
