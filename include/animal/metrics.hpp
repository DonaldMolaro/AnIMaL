#pragma once

#include <cstddef>
#include <vector>

#include "animal/matrix.hpp"

namespace animal {

double accuracy(const Matrix& predictions, const Matrix& labels);

Matrix confusion_matrix(const Matrix& predictions, const Matrix& labels,
                        std::size_t num_classes);

}  // namespace animal
