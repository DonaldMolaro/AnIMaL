#include <vector>

#include "animal/bayes.hpp"
#include "test_common.hpp"
#include "test_registry.hpp"

namespace {

void test_bayes() {
    const double p_b = animal::total_probability(0.95, 0.01, 0.05);
    animal::test::require(animal::test::approx(p_b, 0.059), "total probability failed");

    const double posterior = animal::bayes_posterior(0.95, 0.01, p_b);
    animal::test::require(animal::test::approx(posterior, 0.161016949, 1e-6), "bayes posterior failed");

    bool threw = false;
    try {
        (void)animal::bayes_posterior(1.2, 0.5, 0.5);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    animal::test::require(threw, "invalid probability should throw");
}

}  // namespace

void register_bayes_tests(std::vector<animal::test::TestCase>& tests) {
    tests.push_back({"bayes", test_bayes});
}
