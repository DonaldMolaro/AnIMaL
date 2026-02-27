#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace animal {

class Matrix {
public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols, double value = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    Matrix(std::initializer_list<std::initializer_list<double>> values) {
        rows_ = values.size();
        cols_ = values.begin()->size();
        data_.reserve(rows_ * cols_);

        for (const auto& row : values) {
            if (row.size() != cols_) {
                throw std::runtime_error("Matrix rows must have equal lengths");
            }
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    static Matrix random(std::size_t rows, std::size_t cols, double scale = 1.0) {
        Matrix m(rows, cols);
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<double> dist(0.0, scale);
        for (double& v : m.data_) {
            v = dist(rng);
        }
        return m;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return data_.size(); }

    double& operator()(std::size_t r, std::size_t c) {
        return data_[r * cols_ + c];
    }

    double operator()(std::size_t r, std::size_t c) const {
        return data_[r * cols_ + c];
    }

    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }

    Matrix transpose() const {
        Matrix out(cols_, rows_);
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < cols_; ++c) {
                out(c, r) = (*this)(r, c);
            }
        }
        return out;
    }

    Matrix matmul(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::runtime_error("matmul dimension mismatch");
        }

        Matrix out(rows_, other.cols_);
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t k = 0; k < cols_; ++k) {
                const double left = (*this)(r, k);
                for (std::size_t c = 0; c < other.cols_; ++c) {
                    out(r, c) += left * other(k, c);
                }
            }
        }
        return out;
    }

    Matrix add_row_vector(const Matrix& row_vec) const {
        if (row_vec.rows_ != 1 || row_vec.cols_ != cols_) {
            throw std::runtime_error("add_row_vector expects shape (1, cols)");
        }

        Matrix out(rows_, cols_);
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < cols_; ++c) {
                out(r, c) = (*this)(r, c) + row_vec(0, c);
            }
        }
        return out;
    }

    Matrix map(const std::function<double(double)>& fn) const {
        Matrix out(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = fn(data_[i]);
        }
        return out;
    }

    Matrix operator+(const Matrix& other) const {
        assert_same_shape(other, "operator+");
        Matrix out(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] + other.data_[i];
        }
        return out;
    }

    Matrix operator-(const Matrix& other) const {
        assert_same_shape(other, "operator-");
        Matrix out(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] - other.data_[i];
        }
        return out;
    }

    Matrix hadamard(const Matrix& other) const {
        assert_same_shape(other, "hadamard");
        Matrix out(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] * other.data_[i];
        }
        return out;
    }

    Matrix operator*(double scalar) const {
        Matrix out(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] * scalar;
        }
        return out;
    }

    Matrix operator/(double scalar) const {
        Matrix out(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] / scalar;
        }
        return out;
    }

    Matrix& operator-=(const Matrix& other) {
        assert_same_shape(other, "operator-=");
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    Matrix column_mean() const {
        Matrix out(1, cols_);
        const double inv_rows = rows_ > 0 ? 1.0 / static_cast<double>(rows_) : 0.0;
        for (std::size_t c = 0; c < cols_; ++c) {
            double sum = 0.0;
            for (std::size_t r = 0; r < rows_; ++r) {
                sum += (*this)(r, c);
            }
            out(0, c) = sum * inv_rows;
        }
        return out;
    }

    double mean() const {
        if (data_.empty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (double v : data_) {
            sum += v;
        }
        return sum / static_cast<double>(data_.size());
    }

private:
    void assert_same_shape(const Matrix& other, const char* op_name) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::runtime_error(std::string(op_name) + " shape mismatch");
        }
    }

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

inline std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    for (std::size_t r = 0; r < m.rows(); ++r) {
        os << "[";
        for (std::size_t c = 0; c < m.cols(); ++c) {
            os << m(r, c);
            if (c + 1 < m.cols()) {
                os << ", ";
            }
        }
        os << "]";
        if (r + 1 < m.rows()) {
            os << '\n';
        }
    }
    return os;
}

}  // namespace animal
