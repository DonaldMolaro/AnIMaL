#pragma once

#include <cstddef>

#include "animal/matrix.hpp"

namespace animal {

class LinearRegression {
public:
    void fit(const Matrix& x, const Matrix& y, int epochs = 5000, double learning_rate = 0.01);
    Matrix predict(const Matrix& x) const;
    double mse(const Matrix& x, const Matrix& y) const;
    const Matrix& coefficients() const;
    double intercept() const;
    bool is_fitted() const { return fitted_; }

private:
    void ensure_fitted() const;

    Matrix weights_;
    double bias_ = 0.0;
    bool fitted_ = false;
};

class PolynomialRegression {
public:
    explicit PolynomialRegression(int degree);
    void fit(const Matrix& x, const Matrix& y, int epochs = 8000, double learning_rate = 0.01);
    Matrix predict(const Matrix& x) const;
    double mse(const Matrix& x, const Matrix& y) const;
    const Matrix& coefficients() const;
    double intercept() const;
    int degree() const { return degree_; }
    bool is_fitted() const { return linear_.is_fitted(); }

private:
    Matrix expand_features(const Matrix& x) const;

    int degree_;
    LinearRegression linear_;
};

}  // namespace animal
