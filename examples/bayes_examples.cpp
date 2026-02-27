#include <iomanip>
#include <iostream>

#include "animal/bayes.hpp"

int main() {
    using namespace animal;

    std::cout << std::fixed << std::setprecision(6);

    // Example 1: Medical test
    // A = has disease, B = test is positive
    const double p_a1 = 0.01;
    const double p_b_given_a1 = 0.95;
    const double p_b_given_not_a1 = 0.05;
    const double p_b1 = total_probability(p_b_given_a1, p_a1, p_b_given_not_a1);
    const double p_a_given_b1 = bayes_posterior(p_b_given_a1, p_a1, p_b1);

    std::cout << "Example 1 (medical test):\n";
    std::cout << "P(A)=" << p_a1 << ", P(B|A)=" << p_b_given_a1 << ", P(B|~A)=" << p_b_given_not_a1
              << "\n";
    std::cout << "P(B)=" << p_b1 << "\n";
    std::cout << "Posterior P(A|B)=" << p_a_given_b1 << "\n\n";

    // Example 2: Spam filtering
    // A = email is spam, B = contains word "free"
    const double p_a2 = 0.20;
    const double p_b_given_a2 = 0.70;
    const double p_b_given_not_a2 = 0.10;
    const double p_b2 = total_probability(p_b_given_a2, p_a2, p_b_given_not_a2);
    const double p_a_given_b2 = bayes_posterior(p_b_given_a2, p_a2, p_b2);

    std::cout << "Example 2 (spam filter):\n";
    std::cout << "P(A)=" << p_a2 << ", P(B|A)=" << p_b_given_a2 << ", P(B|~A)=" << p_b_given_not_a2
              << "\n";
    std::cout << "P(B)=" << p_b2 << "\n";
    std::cout << "Posterior P(A|B)=" << p_a_given_b2 << "\n\n";

    // Example 3: Weather and umbrella
    // A = raining, B = carries umbrella
    const double p_a3 = 0.30;
    const double p_b_given_a3 = 0.90;
    const double p_b_given_not_a3 = 0.20;
    const double p_b3 = total_probability(p_b_given_a3, p_a3, p_b_given_not_a3);
    const double p_a_given_b3 = bayes_posterior(p_b_given_a3, p_a3, p_b3);

    std::cout << "Example 3 (umbrella):\n";
    std::cout << "P(A)=" << p_a3 << ", P(B|A)=" << p_b_given_a3 << ", P(B|~A)=" << p_b_given_not_a3
              << "\n";
    std::cout << "P(B)=" << p_b3 << "\n";
    std::cout << "Posterior P(A|B)=" << p_a_given_b3 << "\n";

    return 0;
}
