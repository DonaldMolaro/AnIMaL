#include "animal/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace animal {

Dataset load_xy_csv(const std::string& path, std::size_t feature_count, bool has_header) {
    if (feature_count == 0) {
        throw std::runtime_error("load_xy_csv: feature_count must be > 0");
    }

    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("load_xy_csv: could not open file: " + path);
    }

    std::vector<std::vector<double>> rows;
    std::string line;

    if (has_header) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream line_stream(line);
        std::string cell;
        std::vector<double> values;

        while (std::getline(line_stream, cell, ',')) {
            if (!cell.empty() && cell.back() == '\r') {
                cell.pop_back();
            }
            values.push_back(std::stod(cell));
        }

        const std::size_t expected_cols = feature_count + 1;
        if (values.size() != expected_cols) {
            throw std::runtime_error("load_xy_csv: expected " + std::to_string(expected_cols) +
                                     " columns, got " + std::to_string(values.size()));
        }

        rows.push_back(std::move(values));
    }

    if (rows.empty()) {
        throw std::runtime_error("load_xy_csv: no data rows found in: " + path);
    }

    Matrix x(rows.size(), feature_count);
    Matrix y(rows.size(), 1);

    for (std::size_t r = 0; r < rows.size(); ++r) {
        for (std::size_t c = 0; c < feature_count; ++c) {
            x(r, c) = rows[r][c];
        }
        y(r, 0) = rows[r][feature_count];
    }

    return Dataset{x, y};
}

namespace {

Dataset reorder_rows(const Dataset& data, const std::vector<std::size_t>& indices) {
    const std::size_t n = indices.size();
    Matrix x(n, data.x.cols());
    Matrix y(n, data.y.cols());
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src = indices[i];
        for (std::size_t c = 0; c < data.x.cols(); ++c) {
            x(i, c) = data.x(src, c);
        }
        for (std::size_t c = 0; c < data.y.cols(); ++c) {
            y(i, c) = data.y(src, c);
        }
    }
    return Dataset{x, y};
}

}  // namespace

std::pair<Dataset, Dataset> train_test_split(const Dataset& data, double test_ratio,
                                              unsigned int seed) {
    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        throw std::runtime_error("train_test_split: test_ratio must be in (0, 1)");
    }
    if (data.x.rows() != data.y.rows()) {
        throw std::runtime_error("train_test_split: x and y row count mismatch");
    }

    const std::size_t n = data.x.rows();
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    const std::size_t test_count = std::max(static_cast<std::size_t>(1),
        static_cast<std::size_t>(std::round(static_cast<double>(n) * test_ratio)));
    const std::size_t train_count = n - test_count;

    std::vector<std::size_t> train_idx(indices.begin(), indices.begin() + static_cast<long>(train_count));
    std::vector<std::size_t> test_idx(indices.begin() + static_cast<long>(train_count), indices.end());

    return {reorder_rows(data, train_idx), reorder_rows(data, test_idx)};
}

Dataset shuffle_dataset(const Dataset& data, unsigned int seed) {
    if (data.x.rows() != data.y.rows()) {
        throw std::runtime_error("shuffle_dataset: x and y row count mismatch");
    }

    const std::size_t n = data.x.rows();
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    return reorder_rows(data, indices);
}

Matrix normalize(const Matrix& m) {
    if (m.rows() == 0) {
        throw std::runtime_error("normalize: matrix must not be empty");
    }

    Matrix out(m.rows(), m.cols());
    for (std::size_t c = 0; c < m.cols(); ++c) {
        double col_min = m(0, c);
        double col_max = m(0, c);
        for (std::size_t r = 1; r < m.rows(); ++r) {
            if (m(r, c) < col_min) col_min = m(r, c);
            if (m(r, c) > col_max) col_max = m(r, c);
        }
        const double range = col_max - col_min;
        for (std::size_t r = 0; r < m.rows(); ++r) {
            out(r, c) = (range == 0.0) ? 0.0 : (m(r, c) - col_min) / range;
        }
    }
    return out;
}

Matrix standardize(const Matrix& m) {
    if (m.rows() < 2) {
        throw std::runtime_error("standardize: need at least 2 rows");
    }

    Matrix out(m.rows(), m.cols());
    for (std::size_t c = 0; c < m.cols(); ++c) {
        double col_sum = 0.0;
        for (std::size_t r = 0; r < m.rows(); ++r) {
            col_sum += m(r, c);
        }
        const double col_mean = col_sum / static_cast<double>(m.rows());

        double sq_sum = 0.0;
        for (std::size_t r = 0; r < m.rows(); ++r) {
            const double d = m(r, c) - col_mean;
            sq_sum += d * d;
        }
        const double col_std = std::sqrt(sq_sum / static_cast<double>(m.rows() - 1));

        for (std::size_t r = 0; r < m.rows(); ++r) {
            out(r, c) = (col_std == 0.0) ? 0.0 : (m(r, c) - col_mean) / col_std;
        }
    }
    return out;
}

}  // namespace animal
