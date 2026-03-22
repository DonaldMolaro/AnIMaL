#include "animal/regression.hpp"

#include <cmath>
#include <stdexcept>

namespace animal {

void LinearRegression::fit(const Matrix& x, const Matrix& y, int epochs, double learning_rate) {
    if (x.rows() == 0 || x.cols() == 0) {
        throw std::runtime_error("LinearRegression: x must be non-empty");
    }
    if (y.rows() != x.rows() || y.cols() != 1) {
        throw std::runtime_error("LinearRegression: y must have shape (x.rows, 1)");
    }
    if (epochs <= 0 || learning_rate <= 0.0) {
        throw std::runtime_error("LinearRegression: epochs and learning_rate must be positive");
    }

    weights_ = Matrix(x.cols(), 1, 0.0);
    bias_ = 0.0;
    history_.clear();
    history_.reserve(static_cast<std::size_t>(epochs));

    const double n = static_cast<double>(x.rows());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        Matrix prediction = x.matmul(weights_);
        for (std::size_t r = 0; r < prediction.rows(); ++r) {
            prediction(r, 0) += bias_;
        }

        const Matrix error = prediction - y;
        history_.push_back(error.hadamard(error).mean());

        const Matrix grad_w = x.transpose().matmul(error) / n;

        double grad_b = 0.0;
        for (std::size_t r = 0; r < error.rows(); ++r) {
            grad_b += error(r, 0);
        }
        grad_b /= n;

        weights_ -= grad_w * learning_rate;
        bias_ -= learning_rate * grad_b;
    }

    fitted_ = true;
}

Matrix LinearRegression::predict(const Matrix& x) const {
    ensure_fitted();
    if (x.cols() != weights_.rows()) {
        throw std::runtime_error("LinearRegression: x feature count mismatch");
    }

    Matrix prediction = x.matmul(weights_);
    for (std::size_t r = 0; r < prediction.rows(); ++r) {
        prediction(r, 0) += bias_;
    }
    return prediction;
}

double LinearRegression::mse(const Matrix& x, const Matrix& y) const {
    const Matrix prediction = predict(x);
    const Matrix diff = prediction - y;
    return diff.hadamard(diff).mean();
}

double LinearRegression::r_squared(const Matrix& x, const Matrix& y) const {
    const Matrix prediction = predict(x);
    double y_sum = 0.0;
    for (std::size_t r = 0; r < y.rows(); ++r) {
        y_sum += y(r, 0);
    }
    const double y_mean = y_sum / static_cast<double>(y.rows());

    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (std::size_t r = 0; r < y.rows(); ++r) {
        const double residual = y(r, 0) - prediction(r, 0);
        ss_res += residual * residual;
        const double deviation = y(r, 0) - y_mean;
        ss_tot += deviation * deviation;
    }

    if (ss_tot == 0.0) {
        return 1.0;
    }
    return 1.0 - (ss_res / ss_tot);
}

const Matrix& LinearRegression::coefficients() const {
    ensure_fitted();
    return weights_;
}

double LinearRegression::intercept() const {
    ensure_fitted();
    return bias_;
}

void LinearRegression::ensure_fitted() const {
    if (!fitted_) {
        throw std::runtime_error("LinearRegression: call fit() before predict()/coefficients()");
    }
}

PolynomialRegression::PolynomialRegression(int degree) : degree_(degree) {
    if (degree_ <= 0) {
        throw std::runtime_error("PolynomialRegression: degree must be > 0");
    }
}

void PolynomialRegression::fit(const Matrix& x, const Matrix& y, int epochs, double learning_rate) {
    linear_.fit(expand_features(x), y, epochs, learning_rate);
}

Matrix PolynomialRegression::predict(const Matrix& x) const {
    return linear_.predict(expand_features(x));
}

double PolynomialRegression::mse(const Matrix& x, const Matrix& y) const {
    return linear_.mse(expand_features(x), y);
}

double PolynomialRegression::r_squared(const Matrix& x, const Matrix& y) const {
    return linear_.r_squared(expand_features(x), y);
}

const Matrix& PolynomialRegression::coefficients() const {
    return linear_.coefficients();
}

double PolynomialRegression::intercept() const {
    return linear_.intercept();
}

Matrix PolynomialRegression::expand_features(const Matrix& x) const {
    if (x.cols() != 1) {
        throw std::runtime_error("PolynomialRegression: x must have shape (n, 1)");
    }

    Matrix out(x.rows(), static_cast<std::size_t>(degree_));
    for (std::size_t r = 0; r < x.rows(); ++r) {
        const double base = x(r, 0);
        for (int p = 1; p <= degree_; ++p) {
            out(r, static_cast<std::size_t>(p - 1)) = std::pow(base, static_cast<double>(p));
        }
    }
    return out;
}

}  // namespace animal
