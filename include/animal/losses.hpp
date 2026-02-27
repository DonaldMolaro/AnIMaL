#pragma once

#include <cmath>

#include "animal/matrix.hpp"

namespace animal {

class Loss {
public:
    virtual ~Loss() = default;
    virtual double forward(const Matrix& prediction, const Matrix& target) = 0;
    virtual Matrix backward(const Matrix& prediction, const Matrix& target) = 0;
};

class MSELoss final : public Loss {
public:
    double forward(const Matrix& prediction, const Matrix& target) override {
        const Matrix diff = prediction - target;
        return diff.hadamard(diff).mean();
    }

    Matrix backward(const Matrix& prediction, const Matrix& target) override {
        const Matrix diff = prediction - target;
        return diff * (2.0 / static_cast<double>(diff.size()));
    }
};

class CrossEntropyLoss final : public Loss {
public:
    double forward(const Matrix& prediction, const Matrix& target) override {
        check_shapes(prediction, target);

        double total = 0.0;
        for (std::size_t r = 0; r < prediction.rows(); ++r) {
            for (std::size_t c = 0; c < prediction.cols(); ++c) {
                if (target(r, c) > 0.0) {
                    const double p = clamp_probability(prediction(r, c));
                    total -= target(r, c) * std::log(p);
                }
            }
        }
        return total / static_cast<double>(prediction.rows());
    }

    Matrix backward(const Matrix& prediction, const Matrix& target) override {
        check_shapes(prediction, target);

        Matrix grad(prediction.rows(), prediction.cols(), 0.0);
        const double inv_batch = 1.0 / static_cast<double>(prediction.rows());
        for (std::size_t r = 0; r < prediction.rows(); ++r) {
            for (std::size_t c = 0; c < prediction.cols(); ++c) {
                const double p = clamp_probability(prediction(r, c));
                grad(r, c) = -(target(r, c) / p) * inv_batch;
            }
        }
        return grad;
    }

private:
    static double clamp_probability(double p) {
        constexpr double eps = 1e-12;
        if (p < eps) {
            return eps;
        }
        if (p > 1.0 - eps) {
            return 1.0 - eps;
        }
        return p;
    }

    static void check_shapes(const Matrix& prediction, const Matrix& target) {
        if (prediction.rows() != target.rows() || prediction.cols() != target.cols()) {
            throw std::runtime_error("CrossEntropyLoss: prediction/target shape mismatch");
        }
    }
};

}  // namespace animal
