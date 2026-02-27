#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>

#include "animal/matrix.hpp"

namespace animal {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& grad_output) = 0;
    virtual void update(double learning_rate) = 0;
};

class Dense final : public Layer {
public:
    Dense(std::size_t input_dim, std::size_t output_dim)
        : weights_(Matrix::random(input_dim, output_dim, std::sqrt(2.0 / static_cast<double>(input_dim)))),
          bias_(1, output_dim, 0.0),
          grad_weights_(input_dim, output_dim, 0.0),
          grad_bias_(1, output_dim, 0.0) {}

    Matrix forward(const Matrix& input) override {
        input_cache_ = input;
        return input.matmul(weights_).add_row_vector(bias_);
    }

    Matrix backward(const Matrix& grad_output) override {
        if (grad_output.cols() != weights_.cols()) {
            throw std::runtime_error("Dense backward: grad_output shape mismatch");
        }

        const double batch_size = static_cast<double>(grad_output.rows());
        grad_weights_ = input_cache_.transpose().matmul(grad_output) / batch_size;
        grad_bias_ = grad_output.column_mean();

        return grad_output.matmul(weights_.transpose());
    }

    void update(double learning_rate) override {
        weights_ -= grad_weights_ * learning_rate;
        bias_ -= grad_bias_ * learning_rate;
    }

private:
    Matrix weights_;
    Matrix bias_;

    Matrix input_cache_;
    Matrix grad_weights_;
    Matrix grad_bias_;
};

class ReLU final : public Layer {
public:
    Matrix forward(const Matrix& input) override {
        input_cache_ = input;
        return input.map([](double x) { return x > 0.0 ? x : 0.0; });
    }

    Matrix backward(const Matrix& grad_output) override {
        Matrix grad_input(grad_output.rows(), grad_output.cols());
        for (std::size_t r = 0; r < grad_output.rows(); ++r) {
            for (std::size_t c = 0; c < grad_output.cols(); ++c) {
                grad_input(r, c) = input_cache_(r, c) > 0.0 ? grad_output(r, c) : 0.0;
            }
        }
        return grad_input;
    }

    void update(double) override {}

private:
    Matrix input_cache_;
};

class Sigmoid final : public Layer {
public:
    Matrix forward(const Matrix& input) override {
        output_cache_ = input.map([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
        return output_cache_;
    }

    Matrix backward(const Matrix& grad_output) override {
        Matrix grad_input(grad_output.rows(), grad_output.cols());
        for (std::size_t r = 0; r < grad_output.rows(); ++r) {
            for (std::size_t c = 0; c < grad_output.cols(); ++c) {
                const double y = output_cache_(r, c);
                grad_input(r, c) = grad_output(r, c) * y * (1.0 - y);
            }
        }
        return grad_input;
    }

    void update(double) override {}

private:
    Matrix output_cache_;
};

class Softmax final : public Layer {
public:
    Matrix forward(const Matrix& input) override {
        output_cache_ = Matrix(input.rows(), input.cols());
        for (std::size_t r = 0; r < input.rows(); ++r) {
            double row_max = input(r, 0);
            for (std::size_t c = 1; c < input.cols(); ++c) {
                if (input(r, c) > row_max) {
                    row_max = input(r, c);
                }
            }

            double exp_sum = 0.0;
            for (std::size_t c = 0; c < input.cols(); ++c) {
                const double e = std::exp(input(r, c) - row_max);
                output_cache_(r, c) = e;
                exp_sum += e;
            }

            for (std::size_t c = 0; c < input.cols(); ++c) {
                output_cache_(r, c) /= exp_sum;
            }
        }

        return output_cache_;
    }

    Matrix backward(const Matrix& grad_output) override {
        Matrix grad_input(grad_output.rows(), grad_output.cols(), 0.0);
        for (std::size_t r = 0; r < grad_output.rows(); ++r) {
            for (std::size_t i = 0; i < grad_output.cols(); ++i) {
                double sum = 0.0;
                for (std::size_t j = 0; j < grad_output.cols(); ++j) {
                    const double s_i = output_cache_(r, i);
                    const double s_j = output_cache_(r, j);
                    const double jacobian_ij = (i == j) ? s_i * (1.0 - s_i) : -s_i * s_j;
                    sum += jacobian_ij * grad_output(r, j);
                }
                grad_input(r, i) = sum;
            }
        }
        return grad_input;
    }

    void update(double) override {}

private:
    Matrix output_cache_;
};

}  // namespace animal
