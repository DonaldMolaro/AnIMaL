#include <iostream>
#include <vector>

#include "test_common.hpp"
#include "test_registry.hpp"

int main() {
    std::vector<animal::test::TestCase> tests;
    tests.reserve(16);

    register_matrix_tests(tests);
    register_dataset_tests(tests);
    register_statistics_tests(tests);
    register_bayes_tests(tests);
    register_losses_layers_tests(tests);
    register_regression_tests(tests);
    register_metrics_tests(tests);

    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "[PASS] " << test.first << "\n";
        } catch (const std::exception& ex) {
            ++failures;
            std::cout << "[FAIL] " << test.first << ": " << ex.what() << "\n";
        } catch (...) {
            ++failures;
            std::cout << "[FAIL] " << test.first << ": unknown exception\n";
        }
    }

    if (failures > 0) {
        std::cout << failures << " test(s) failed\n";
        return 1;
    }

    std::cout << "All tests passed\n";
    return 0;
}
