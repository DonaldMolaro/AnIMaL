#pragma once

#include <cstddef>
#include <string>

#include "animal/matrix.hpp"

namespace animal {

struct Dataset {
    Matrix x;
    Matrix y;
};

Dataset load_xy_csv(const std::string& path, std::size_t feature_count, bool has_header = false);

}  // namespace animal
