#pragma once

namespace animal {

// Computes posterior P(A|B) using Bayes' rule:
// P(A|B) = P(B|A) * P(A) / P(B)
double bayes_posterior(double p_b_given_a, double p_a, double p_b);

// Computes total evidence P(B) for two-class partition {A, not A}:
// P(B) = P(B|A)P(A) + P(B|~A)P(~A)
double total_probability(double p_b_given_a, double p_a, double p_b_given_not_a);

}  // namespace animal
