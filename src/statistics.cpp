#include "animal/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace animal {

namespace {

long long quantize(double value) {
    constexpr double scale = 1e9;
    return static_cast<long long>(std::llround(value * scale));
}

double dequantize(long long value) {
    constexpr double scale = 1e9;
    return static_cast<double>(value) / scale;
}

}  // namespace

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::runtime_error("mean: input must not be empty");
    }

    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }

    return sum / static_cast<double>(values.size());
}

double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("median: input must not be empty");
    }

    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();

    if (n % 2 == 1) {
        return values[n / 2];
    }

    return (values[(n / 2) - 1] + values[n / 2]) / 2.0;
}

std::optional<ModeResult> mode(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::runtime_error("mode: input must not be empty");
    }

    std::map<long long, std::size_t> counts;
    for (double value : values) {
        ++counts[quantize(value)];
    }

    std::size_t best_frequency = 0;
    long long best_key = 0;
    std::size_t modes_at_best = 0;

    for (const auto& entry : counts) {
        const long long key = entry.first;
        const std::size_t frequency = entry.second;
        if (frequency > best_frequency) {
            best_frequency = frequency;
            best_key = key;
            modes_at_best = 1;
        } else if (frequency == best_frequency) {
            ++modes_at_best;
        }
    }

    if (best_frequency <= 1) {
        return std::nullopt;
    }

    if (modes_at_best > 1) {
        return std::nullopt;
    }

    return ModeResult{dequantize(best_key), best_frequency, true};
}

double variance(const std::vector<double>& values) {
    if (values.size() < 2) {
        throw std::runtime_error("variance: need at least 2 values");
    }
    return covariance(values, values);
}

double stddev(const std::vector<double>& values) {
    return std::sqrt(variance(values));
}

double covariance(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.empty() || y.empty()) {
        throw std::runtime_error("covariance: inputs must not be empty");
    }
    if (x.size() != y.size()) {
        throw std::runtime_error("covariance: x and y must have the same length");
    }
    if (x.size() < 2) {
        throw std::runtime_error("covariance: need at least 2 points");
    }

    const double x_mean = mean(x);
    const double y_mean = mean(y);

    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        sum += (x[i] - x_mean) * (y[i] - y_mean);
    }

    return sum / static_cast<double>(x.size() - 1);
}

double pearson_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    const double cov = covariance(x, y);
    const double var_x = covariance(x, x);
    const double var_y = covariance(y, y);
    const double denom = std::sqrt(var_x * var_y);

    if (denom == 0.0) {
        throw std::runtime_error("pearson_correlation: variance of x or y is zero");
    }

    return cov / denom;
}

double r_squared(const std::vector<double>& actual, const std::vector<double>& predicted) {
    if (actual.empty() || predicted.empty()) {
        throw std::runtime_error("r_squared: inputs must not be empty");
    }
    if (actual.size() != predicted.size()) {
        throw std::runtime_error("r_squared: actual and predicted must have the same length");
    }

    const double y_mean = mean(actual);
    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double residual = actual[i] - predicted[i];
        ss_res += residual * residual;
        const double deviation = actual[i] - y_mean;
        ss_tot += deviation * deviation;
    }

    if (ss_tot == 0.0) {
        return 1.0;
    }
    return 1.0 - (ss_res / ss_tot);
}

std::vector<double> column_values(const Matrix& matrix, std::size_t column_index) {
    if (column_index >= matrix.cols()) {
        throw std::runtime_error("column_values: column index out of range");
    }

    std::vector<double> out;
    out.reserve(matrix.rows());
    for (std::size_t r = 0; r < matrix.rows(); ++r) {
        out.push_back(matrix(r, column_index));
    }
    return out;
}

}  // namespace animal
