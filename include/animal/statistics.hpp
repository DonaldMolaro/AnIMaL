#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "animal/matrix.hpp"

namespace animal {

struct ModeResult {
    double value = 0.0;
    std::size_t frequency = 0;
    bool unique = false;
};

double mean(const std::vector<double>& values);
double median(std::vector<double> values);
std::optional<ModeResult> mode(const std::vector<double>& values);
double variance(const std::vector<double>& values);
double stddev(const std::vector<double>& values);
double covariance(const std::vector<double>& x, const std::vector<double>& y);
double pearson_correlation(const std::vector<double>& x, const std::vector<double>& y);
double r_squared(const std::vector<double>& actual, const std::vector<double>& predicted);

std::vector<double> column_values(const Matrix& matrix, std::size_t column_index);

}  // namespace animal
