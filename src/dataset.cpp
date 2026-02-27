#include "animal/dataset.hpp"

#include <fstream>
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

}  // namespace animal
