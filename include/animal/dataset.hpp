#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include "animal/matrix.hpp"

namespace animal {

struct Dataset {
    Matrix x;
    Matrix y;
};

Dataset load_xy_csv(const std::string& path, std::size_t feature_count, bool has_header = false);

std::pair<Dataset, Dataset> train_test_split(const Dataset& data, double test_ratio = 0.2,
                                              unsigned int seed = 42);

Dataset shuffle_dataset(const Dataset& data, unsigned int seed = 42);

Matrix normalize(const Matrix& m);
Matrix standardize(const Matrix& m);

}  // namespace animal
