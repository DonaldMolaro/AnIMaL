#pragma once

#include <cmath>
#include <memory>
#include <random>
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

class Tanh final : public Layer {
public:
    Matrix forward(const Matrix& input) override {
        output_cache_ = input.map([](double x) { return std::tanh(x); });
        return output_cache_;
    }

    Matrix backward(const Matrix& grad_output) override {
        Matrix grad_input(grad_output.rows(), grad_output.cols());
        for (std::size_t r = 0; r < grad_output.rows(); ++r) {
            for (std::size_t c = 0; c < grad_output.cols(); ++c) {
                const double y = output_cache_(r, c);
                grad_input(r, c) = grad_output(r, c) * (1.0 - y * y);
            }
        }
        return grad_input;
    }

    void update(double) override {}

private:
    Matrix output_cache_;
};

class Dropout final : public Layer {
public:
    explicit Dropout(double rate = 0.5) : rate_(rate) {
        if (rate_ < 0.0 || rate_ >= 1.0) {
            throw std::runtime_error("Dropout rate must be in [0, 1)");
        }
    }

    void set_training(bool training) { training_ = training; }

    Matrix forward(const Matrix& input) override {
        if (!training_) {
            return input;
        }

        mask_ = Matrix(input.rows(), input.cols());
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const double scale = 1.0 / (1.0 - rate_);

        Matrix out(input.rows(), input.cols());
        for (std::size_t r = 0; r < input.rows(); ++r) {
            for (std::size_t c = 0; c < input.cols(); ++c) {
                const double keep = dist(rng) >= rate_ ? 1.0 : 0.0;
                mask_(r, c) = keep;
                out(r, c) = input(r, c) * keep * scale;
            }
        }
        return out;
    }

    Matrix backward(const Matrix& grad_output) override {
        if (!training_) {
            return grad_output;
        }
        const double scale = 1.0 / (1.0 - rate_);
        return grad_output.hadamard(mask_) * scale;
    }

    void update(double) override {}

private:
    double rate_;
    bool training_ = true;
    Matrix mask_;
};

class BatchNorm final : public Layer {
public:
    explicit BatchNorm(std::size_t features, double momentum = 0.1, double epsilon = 1e-5)
        : features_(features),
          momentum_(momentum),
          epsilon_(epsilon),
          gamma_(1, features, 1.0),
          beta_(1, features, 0.0),
          running_mean_(1, features, 0.0),
          running_var_(1, features, 1.0),
          grad_gamma_(1, features, 0.0),
          grad_beta_(1, features, 0.0) {}

    void set_training(bool training) { training_ = training; }

    Matrix forward(const Matrix& input) override {
        if (input.cols() != features_) {
            throw std::runtime_error("BatchNorm: input feature count mismatch");
        }

        const std::size_t n = input.rows();
        const double dn = static_cast<double>(n);

        if (training_) {
            batch_mean_ = input.column_mean();

            batch_var_ = Matrix(1, features_, 0.0);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t c = 0; c < features_; ++c) {
                    const double d = input(r, c) - batch_mean_(0, c);
                    batch_var_(0, c) += d * d;
                }
            }
            for (std::size_t c = 0; c < features_; ++c) {
                batch_var_(0, c) /= dn;
            }

            for (std::size_t c = 0; c < features_; ++c) {
                running_mean_(0, c) = (1.0 - momentum_) * running_mean_(0, c) +
                                       momentum_ * batch_mean_(0, c);
                running_var_(0, c) = (1.0 - momentum_) * running_var_(0, c) +
                                      momentum_ * batch_var_(0, c);
            }

            normalized_ = Matrix(n, features_);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t c = 0; c < features_; ++c) {
                    normalized_(r, c) = (input(r, c) - batch_mean_(0, c)) /
                                         std::sqrt(batch_var_(0, c) + epsilon_);
                }
            }
        } else {
            normalized_ = Matrix(n, features_);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t c = 0; c < features_; ++c) {
                    normalized_(r, c) = (input(r, c) - running_mean_(0, c)) /
                                         std::sqrt(running_var_(0, c) + epsilon_);
                }
            }
        }

        Matrix out(n, features_);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < features_; ++c) {
                out(r, c) = gamma_(0, c) * normalized_(r, c) + beta_(0, c);
            }
        }

        input_cache_ = input;
        return out;
    }

    Matrix backward(const Matrix& grad_output) override {
        const std::size_t n = grad_output.rows();
        const double dn = static_cast<double>(n);

        grad_gamma_ = Matrix(1, features_, 0.0);
        grad_beta_ = Matrix(1, features_, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < features_; ++c) {
                grad_gamma_(0, c) += grad_output(r, c) * normalized_(r, c);
                grad_beta_(0, c) += grad_output(r, c);
            }
        }

        Matrix grad_input(n, features_);
        for (std::size_t c = 0; c < features_; ++c) {
            const double inv_std = 1.0 / std::sqrt(batch_var_(0, c) + epsilon_);
            for (std::size_t r = 0; r < n; ++r) {
                grad_input(r, c) = gamma_(0, c) * inv_std / dn *
                    (dn * grad_output(r, c) -
                     grad_beta_(0, c) -
                     normalized_(r, c) * grad_gamma_(0, c));
            }
        }

        return grad_input;
    }

    void update(double learning_rate) override {
        gamma_ -= grad_gamma_ * learning_rate;
        beta_ -= grad_beta_ * learning_rate;
    }

private:
    std::size_t features_;
    double momentum_;
    double epsilon_;
    Matrix gamma_;
    Matrix beta_;
    Matrix running_mean_;
    Matrix running_var_;
    Matrix batch_mean_;
    Matrix batch_var_;
    Matrix normalized_;
    Matrix input_cache_;
    Matrix grad_gamma_;
    Matrix grad_beta_;
    bool training_ = true;
};

}  // namespace animal
