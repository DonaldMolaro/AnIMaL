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

void test_variance_stddev() {
    const std::vector<double> v = {2, 4, 4, 4, 5, 5, 7, 9};
    const double var = animal::variance(v);
    animal::test::require(animal::test::approx(var, 4.571428, 1e-4), "variance failed");

    const double sd = animal::stddev(v);
    animal::test::require(animal::test::approx(sd, 2.13809, 1e-4), "stddev failed");

    bool threw = false;
    try { (void)animal::variance({1.0}); }
    catch (const std::runtime_error&) { threw = true; }
    animal::test::require(threw, "variance with 1 element should throw");
}

void test_r_squared() {
    const std::vector<double> actual = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> perfect = {1.0, 2.0, 3.0, 4.0, 5.0};
    animal::test::require(animal::test::approx(animal::r_squared(actual, perfect), 1.0),
                          "r_squared perfect prediction should be 1.0");

    const std::vector<double> mean_pred = {3.0, 3.0, 3.0, 3.0, 3.0};
    animal::test::require(animal::test::approx(animal::r_squared(actual, mean_pred), 0.0),
                          "r_squared mean prediction should be 0.0");

    const std::vector<double> bad_pred = {5.0, 4.0, 3.0, 2.0, 1.0};
    animal::test::require(animal::r_squared(actual, bad_pred) < 0.0,
                          "r_squared worse than mean should be negative");
}

}  // namespace

void register_statistics_tests(std::vector<animal::test::TestCase>& tests) {
    tests.push_back({"statistics", test_statistics});
    tests.push_back({"variance_stddev", test_variance_stddev});
    tests.push_back({"r_squared", test_r_squared});
}
