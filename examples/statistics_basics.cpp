#include <iomanip>
#include <iostream>
#include <vector>

#include "animal/statistics.hpp"

int main() {
    using namespace animal;

    const std::vector<double> distribution = {2, 4, 4, 4, 5, 5, 7, 9};

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Distribution: ";
    for (double v : distribution) {
        std::cout << v << ' ';
    }
    std::cout << "\n";

    std::cout << "Mean: " << mean(distribution) << "\n";
    std::cout << "Median: " << median(distribution) << "\n";

    const auto mode_result = mode(distribution);
    if (mode_result.has_value()) {
        std::cout << "Mode: " << mode_result->value << " (frequency=" << mode_result->frequency << ")\n";
    } else {
        std::cout << "Mode: none (no unique repeated value)\n";
    }

    return 0;
}
