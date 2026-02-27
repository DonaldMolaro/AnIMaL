#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace animal::test {

using TestFn = std::function<void()>;
using TestCase = std::pair<std::string, TestFn>;

inline bool approx(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps;
}

inline void require(bool cond, const std::string& message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

}  // namespace animal::test
