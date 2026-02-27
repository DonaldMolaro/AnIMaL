#include "animal/bayes.hpp"

#include <string>
#include <stdexcept>

namespace animal {

namespace {

void validate_probability(double p, const char* name) {
    if (p < 0.0 || p > 1.0) {
        throw std::runtime_error(std::string(name) + " must be in [0, 1]");
    }
}

}  // namespace

double bayes_posterior(double p_b_given_a, double p_a, double p_b) {
    validate_probability(p_b_given_a, "P(B|A)");
    validate_probability(p_a, "P(A)");
    validate_probability(p_b, "P(B)");

    if (p_b == 0.0) {
        throw std::runtime_error("P(B) must be > 0 for Bayes posterior");
    }

    return (p_b_given_a * p_a) / p_b;
}

double total_probability(double p_b_given_a, double p_a, double p_b_given_not_a) {
    validate_probability(p_b_given_a, "P(B|A)");
    validate_probability(p_a, "P(A)");
    validate_probability(p_b_given_not_a, "P(B|~A)");

    const double p_not_a = 1.0 - p_a;
    return (p_b_given_a * p_a) + (p_b_given_not_a * p_not_a);
}

}  // namespace animal
