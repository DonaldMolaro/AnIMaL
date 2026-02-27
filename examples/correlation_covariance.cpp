#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "animal/statistics.hpp"

namespace {

void print_case(const std::string& name, const std::vector<double>& x, const std::vector<double>& y) {
    const double cov = animal::covariance(x, y);
    const double corr = animal::pearson_correlation(x, y);

    std::cout << name << "\n";
    std::cout << "covariance: " << cov << "\n";
    std::cout << "correlation: " << corr << "\n";
    if (corr > 0.0) {
        std::cout << "interpretation: positive linear relationship\n\n";
    } else if (corr < 0.0) {
        std::cout << "interpretation: negative linear relationship\n\n";
    } else {
        std::cout << "interpretation: no linear relationship\n\n";
    }
}

}  // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const std::vector<double> x_pos = {1, 2, 3, 4, 5, 6};
    const std::vector<double> y_pos = {2, 4, 6, 8, 10, 12};

    const std::vector<double> x_neg = {1, 2, 3, 4, 5, 6};
    const std::vector<double> y_neg = {12, 10, 8, 6, 4, 2};

    const std::vector<double> x_weak = {1, 2, 3, 4, 5, 6};
    const std::vector<double> y_weak = {3, 5, 4, 6, 5, 4};

    print_case("Case 1: strong positive", x_pos, y_pos);
    print_case("Case 2: strong negative", x_neg, y_neg);
    print_case("Case 3: weak/no clear linear pattern", x_weak, y_weak);

    return 0;
}
