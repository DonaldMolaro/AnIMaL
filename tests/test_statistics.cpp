#include <vector>

#include "animal/statistics.hpp"
#include "test_common.hpp"
#include "test_registry.hpp"

namespace {

void test_statistics() {
    const std::vector<double> v = {2, 4, 4, 4, 5, 5, 7, 9};
    animal::test::require(animal::test::approx(animal::mean(v), 5.0), "mean failed");
    animal::test::require(animal::test::approx(animal::median(v), 4.5), "median failed");

    const auto m = animal::mode(v);
    animal::test::require(m.has_value(), "mode should exist");
    animal::test::require(animal::test::approx(m->value, 4.0), "mode value wrong");
    animal::test::require(m->frequency == 3, "mode frequency wrong");

    const std::vector<double> none = {1, 2, 3, 4};
    animal::test::require(!animal::mode(none).has_value(), "mode should be nullopt for all-unique data");

    const std::vector<double> x = {1, 2, 3, 4, 5, 6};
    const std::vector<double> y_pos = {2, 4, 6, 8, 10, 12};
    const std::vector<double> y_neg = {12, 10, 8, 6, 4, 2};

    animal::test::require(
        animal::test::approx(animal::pearson_correlation(x, y_pos), 1.0), "correlation positive failed");
    animal::test::require(
        animal::test::approx(animal::pearson_correlation(x, y_neg), -1.0), "correlation negative failed");

    bool threw = false;
    try {
        (void)animal::pearson_correlation({1, 1, 1}, {2, 3, 4});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    animal::test::require(threw, "zero variance correlation should throw");
}

}  // namespace

void register_statistics_tests(std::vector<animal::test::TestCase>& tests) {
    tests.push_back({"statistics", test_statistics});
}
