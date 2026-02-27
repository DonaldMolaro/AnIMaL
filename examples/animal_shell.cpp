#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "animal/bayes.hpp"
#include "animal/dataset.hpp"
#include "animal/regression.hpp"
#include "animal/statistics.hpp"

namespace {

void print_help() {
    std::cout << "Commands:\n"
              << "  help\n"
              << "  quit | exit\n"
              << "  load linear <csv_path>\n"
              << "  load poly <csv_path> [degree]\n"
              << "  fit linear [epochs] [learning_rate]\n"
              << "  fit poly [epochs] [learning_rate]\n"
              << "  show linear\n"
              << "  show poly\n"
              << "  predict linear <x>\n"
              << "  predict poly <x>\n"
              << "  stats <v1> <v2> ...\n"
              << "  stats y linear|poly\n"
              << "  stats relation linear|poly\n"
              << "  bayes posterior <P(B|A)> <P(A)> <P(B)>\n"
              << "  bayes from_likelihood <P(B|A)> <P(A)> <P(B|~A)>\n"
              << "  lesson linear [csv_path]\n"
              << "  lesson poly [csv_path] [degree]\n"
              << "  lesson bayes\n"
              << "  lesson correlation\n"
              << "\n"
              << "Defaults:\n"
              << "  linear epochs=6000 lr=0.01\n"
              << "  poly   epochs=12000 lr=0.03\n"
              << "  lesson linear data=data/linear_regression.csv\n"
              << "  lesson poly   data=data/polynomial_regression.csv degree=2\n";
}

animal::Matrix single_input(double x) {
    animal::Matrix m(1, 1);
    m(0, 0) = x;
    return m;
}

void print_linear_summary(const animal::LinearRegression& model) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Linear coefficients:\n" << model.coefficients() << "\nintercept: " << model.intercept()
              << "\n";
}

void print_poly_summary(const animal::PolynomialRegression& model) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Polynomial coefficients (x^1..x^d):\n"
              << model.coefficients() << "\nintercept: " << model.intercept()
              << "\ndegree: " << model.degree() << "\n";
}

void print_distribution_stats(const std::vector<double>& values, const std::string& label) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Distribution stats (" << label << ")\n";
    std::cout << "count: " << values.size() << "\n";
    std::cout << "mean: " << animal::mean(values) << "\n";
    std::cout << "median: " << animal::median(values) << "\n";
    const auto mode_result = animal::mode(values);
    if (mode_result.has_value()) {
        std::cout << "mode: " << mode_result->value << " (frequency=" << mode_result->frequency << ")\n";
    } else {
        std::cout << "mode: none (no unique repeated value)\n";
    }
}

void print_bayes_example(
    const std::string& title, double p_a, double p_b_given_a, double p_b_given_not_a) {
    const double p_b = animal::total_probability(p_b_given_a, p_a, p_b_given_not_a);
    const double posterior = animal::bayes_posterior(p_b_given_a, p_a, p_b);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << title << "\n";
    std::cout << "P(A)=" << p_a << ", P(B|A)=" << p_b_given_a << ", P(B|~A)=" << p_b_given_not_a << "\n";
    std::cout << "P(B)=" << p_b << "\n";
    std::cout << "P(A|B)=" << posterior << "\n";
}

void run_bayes_lesson() {
    std::cout << "Lesson: Bayes Rule\n";
    std::cout << "Step 1: Formula P(A|B) = P(B|A) * P(A) / P(B)\n";
    std::cout << "Step 2: Use total probability when P(B) is unknown:\n";
    std::cout << "        P(B) = P(B|A)P(A) + P(B|~A)P(~A)\n";
    std::cout << "Step 3: Work through examples.\n";

    print_bayes_example("  Example 1 (medical test):", 0.01, 0.95, 0.05);
    print_bayes_example("  Example 2 (spam filter):", 0.20, 0.70, 0.10);
    print_bayes_example("  Example 3 (umbrella):", 0.30, 0.90, 0.20);
    std::cout << "Lesson complete.\n";
}

void run_correlation_lesson() {
    const std::vector<double> x_pos = {1, 2, 3, 4, 5, 6};
    const std::vector<double> y_pos = {2, 4, 6, 8, 10, 12};
    const std::vector<double> x_neg = {1, 2, 3, 4, 5, 6};
    const std::vector<double> y_neg = {12, 10, 8, 6, 4, 2};
    const std::vector<double> x_weak = {1, 2, 3, 4, 5, 6};
    const std::vector<double> y_weak = {3, 5, 4, 6, 5, 4};

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Lesson: Correlation and Covariance\n";
    std::cout << "Step 1: Covariance sign indicates direction (positive/negative).\n";
    std::cout << "Step 2: Pearson correlation rescales to [-1, 1].\n";

    std::cout << "Example A (positive): cov=" << animal::covariance(x_pos, y_pos)
              << ", corr=" << animal::pearson_correlation(x_pos, y_pos) << "\n";
    std::cout << "Example B (negative): cov=" << animal::covariance(x_neg, y_neg)
              << ", corr=" << animal::pearson_correlation(x_neg, y_neg) << "\n";
    std::cout << "Example C (weak): cov=" << animal::covariance(x_weak, y_weak)
              << ", corr=" << animal::pearson_correlation(x_weak, y_weak) << "\n";
    std::cout << "Lesson complete.\n";
}

void run_linear_lesson(animal::LinearRegression& model, animal::Dataset& data, const std::string& path) {
    std::cout << "Lesson: Linear Regression\n";
    std::cout << "Step 1: Load data from " << path << "\n";
    data = animal::load_xy_csv(path, 1);
    std::cout << "Loaded " << data.x.rows() << " rows.\n";

    std::cout << "Step 2: Fit y = w*x + b with gradient descent (epochs=6000, lr=0.01).\n";
    model.fit(data.x, data.y, 6000, 0.01);

    std::cout << "Step 3: Inspect learned parameters and training error.\n";
    print_linear_summary(model);
    std::cout << "Training MSE: " << model.mse(data.x, data.y) << "\n";

    std::cout << "Step 4: Predict for x=13.\n";
    const animal::Matrix y = model.predict(single_input(13.0));
    std::cout << "Prediction: y=" << y(0, 0) << "\n";
    std::cout << "Lesson complete.\n";
}

void run_poly_lesson(
    animal::PolynomialRegression& model, animal::Dataset& data, const std::string& path, int degree) {
    std::cout << "Lesson: Polynomial Regression\n";
    std::cout << "Step 1: Load data from " << path << "\n";
    data = animal::load_xy_csv(path, 1);
    std::cout << "Loaded " << data.x.rows() << " rows.\n";

    std::cout << "Step 2: Fit polynomial model with degree=" << degree
              << " (epochs=12000, lr=0.03).\n";
    model = animal::PolynomialRegression(degree);
    model.fit(data.x, data.y, 12000, 0.03);

    std::cout << "Step 3: Inspect learned coefficients and training error.\n";
    print_poly_summary(model);
    std::cout << "Training MSE: " << model.mse(data.x, data.y) << "\n";

    std::cout << "Step 4: Predict for x=2.5.\n";
    const animal::Matrix y = model.predict(single_input(2.5));
    std::cout << "Prediction: y=" << y(0, 0) << "\n";
    std::cout << "Lesson complete.\n";
}

}  // namespace

int main() {
    using namespace animal;

    Dataset linear_data;
    Dataset poly_data;
    bool has_linear_data = false;
    bool has_poly_data = false;

    LinearRegression linear_model;
    PolynomialRegression poly_model(2);

    bool linear_trained = false;
    bool poly_trained = false;

    std::cout << "AnIMaL Teaching Shell\n";
    std::cout << "Type 'help' for commands.\n\n";

    std::string line;
    while (true) {
        std::cout << "animal> ";
        if (!std::getline(std::cin, line)) {
            std::cout << "\n";
            break;
        }

        std::istringstream input(line);
        std::string cmd;
        input >> cmd;

        if (cmd.empty()) {
            continue;
        }

        try {
            if (cmd == "help") {
                print_help();
                continue;
            }

            if (cmd == "quit" || cmd == "exit") {
                break;
            }

            if (cmd == "stats") {
                std::string target;
                input >> target;

                if (target.empty()) {
                    std::cout << "Usage: stats <v1> <v2> ... | stats y linear|poly\n";
                    continue;
                }

                if (target == "y") {
                    std::string which;
                    input >> which;
                    if (which == "linear") {
                        if (!has_linear_data) {
                            std::cout << "Load linear data first: load linear <csv_path>\n";
                            continue;
                        }
                        print_distribution_stats(column_values(linear_data.y, 0), "linear y");
                        continue;
                    }
                    if (which == "poly") {
                        if (!has_poly_data) {
                            std::cout << "Load polynomial data first: load poly <csv_path> [degree]\n";
                            continue;
                        }
                        print_distribution_stats(column_values(poly_data.y, 0), "poly y");
                        continue;
                    }
                    std::cout << "Usage: stats y linear|poly\n";
                    continue;
                }

                if (target == "relation") {
                    std::string which;
                    input >> which;
                    std::vector<double> x_values;
                    std::vector<double> y_values;

                    if (which == "linear") {
                        if (!has_linear_data) {
                            std::cout << "Load linear data first: load linear <csv_path>\n";
                            continue;
                        }
                        x_values = column_values(linear_data.x, 0);
                        y_values = column_values(linear_data.y, 0);
                    } else if (which == "poly") {
                        if (!has_poly_data) {
                            std::cout << "Load polynomial data first: load poly <csv_path> [degree]\n";
                            continue;
                        }
                        x_values = column_values(poly_data.x, 0);
                        y_values = column_values(poly_data.y, 0);
                    } else {
                        std::cout << "Usage: stats relation linear|poly\n";
                        continue;
                    }

                    std::cout << std::fixed << std::setprecision(6);
                    std::cout << "covariance: " << covariance(x_values, y_values) << "\n";
                    std::cout << "correlation: " << pearson_correlation(x_values, y_values) << "\n";
                    continue;
                }

                std::vector<double> values;
                values.push_back(std::stod(target));
                std::string token;
                while (input >> token) {
                    values.push_back(std::stod(token));
                }
                print_distribution_stats(values, "inline values");
                continue;
            }

            if (cmd == "lesson") {
                std::string target;
                input >> target;

                if (target == "linear") {
                    std::string path = "data/linear_regression.csv";
                    if (input >> path) {
                    }
                    run_linear_lesson(linear_model, linear_data, path);
                    has_linear_data = true;
                    linear_trained = true;
                    continue;
                }

                if (target == "poly") {
                    std::string path = "data/polynomial_regression.csv";
                    int degree = 2;
                    if (input >> path) {
                        if (input >> degree) {
                        }
                    }
                    run_poly_lesson(poly_model, poly_data, path, degree);
                    has_poly_data = true;
                    poly_trained = true;
                    continue;
                }

                if (target == "bayes") {
                    run_bayes_lesson();
                    continue;
                }

                if (target == "correlation") {
                    run_correlation_lesson();
                    continue;
                }

                std::cout << "Usage: lesson linear [csv_path] | lesson poly [csv_path] [degree] | lesson bayes | "
                             "lesson correlation\n";
                continue;
            }

            if (cmd == "bayes") {
                std::string mode;
                input >> mode;
                if (mode == "posterior") {
                    double p_b_given_a = 0.0;
                    double p_a = 0.0;
                    double p_b = 0.0;
                    input >> p_b_given_a >> p_a >> p_b;
                    if (input.fail()) {
                        std::cout << "Usage: bayes posterior <P(B|A)> <P(A)> <P(B)>\n";
                        continue;
                    }
                    std::cout << std::fixed << std::setprecision(6);
                    std::cout << "P(A|B)=" << bayes_posterior(p_b_given_a, p_a, p_b) << "\n";
                    continue;
                }

                if (mode == "from_likelihood") {
                    double p_b_given_a = 0.0;
                    double p_a = 0.0;
                    double p_b_given_not_a = 0.0;
                    input >> p_b_given_a >> p_a >> p_b_given_not_a;
                    if (input.fail()) {
                        std::cout << "Usage: bayes from_likelihood <P(B|A)> <P(A)> <P(B|~A)>\n";
                        continue;
                    }
                    const double p_b = total_probability(p_b_given_a, p_a, p_b_given_not_a);
                    const double posterior = bayes_posterior(p_b_given_a, p_a, p_b);
                    std::cout << std::fixed << std::setprecision(6);
                    std::cout << "P(B)=" << p_b << ", P(A|B)=" << posterior << "\n";
                    continue;
                }

                std::cout
                    << "Usage: bayes posterior <P(B|A)> <P(A)> <P(B)> | bayes from_likelihood <P(B|A)> <P(A)> "
                       "<P(B|~A)>\n";
                continue;
            }

            if (cmd == "load") {
                std::string target;
                std::string path;
                input >> target >> path;
                if (target.empty() || path.empty()) {
                    std::cout << "Usage: load linear <csv_path> | load poly <csv_path> [degree]\n";
                    continue;
                }

                if (target == "linear") {
                    linear_data = load_xy_csv(path, 1);
                    has_linear_data = true;
                    linear_trained = false;
                    std::cout << "Loaded linear dataset with " << linear_data.x.rows() << " rows.\n";
                    continue;
                }

                if (target == "poly") {
                    int degree = 2;
                    if (input >> degree) {
                        poly_model = PolynomialRegression(degree);
                    } else {
                        poly_model = PolynomialRegression(2);
                    }
                    poly_data = load_xy_csv(path, 1);
                    has_poly_data = true;
                    poly_trained = false;
                    std::cout << "Loaded polynomial dataset with " << poly_data.x.rows()
                              << " rows (degree=" << poly_model.degree() << ").\n";
                    continue;
                }

                std::cout << "Unknown target: " << target << "\n";
                continue;
            }

            if (cmd == "fit") {
                std::string target;
                input >> target;
                if (target == "linear") {
                    if (!has_linear_data) {
                        std::cout << "Load linear data first: load linear <csv_path>\n";
                        continue;
                    }
                    int epochs = 6000;
                    double lr = 0.01;
                    input >> epochs >> lr;
                    linear_model.fit(linear_data.x, linear_data.y, epochs, lr);
                    linear_trained = true;
                    std::cout << "Linear model trained. MSE=" << linear_model.mse(linear_data.x, linear_data.y)
                              << "\n";
                    continue;
                }

                if (target == "poly") {
                    if (!has_poly_data) {
                        std::cout << "Load polynomial data first: load poly <csv_path> [degree]\n";
                        continue;
                    }
                    int epochs = 12000;
                    double lr = 0.03;
                    input >> epochs >> lr;
                    poly_model.fit(poly_data.x, poly_data.y, epochs, lr);
                    poly_trained = true;
                    std::cout << "Polynomial model trained. MSE=" << poly_model.mse(poly_data.x, poly_data.y)
                              << "\n";
                    continue;
                }

                std::cout << "Usage: fit linear [epochs] [learning_rate] | fit poly [epochs] [learning_rate]\n";
                continue;
            }

            if (cmd == "show") {
                std::string target;
                input >> target;
                if (target == "linear") {
                    if (!linear_trained) {
                        std::cout << "Linear model is not trained.\n";
                        continue;
                    }
                    print_linear_summary(linear_model);
                    continue;
                }

                if (target == "poly") {
                    if (!poly_trained) {
                        std::cout << "Polynomial model is not trained.\n";
                        continue;
                    }
                    print_poly_summary(poly_model);
                    continue;
                }

                std::cout << "Usage: show linear | show poly\n";
                continue;
            }

            if (cmd == "predict") {
                std::string target;
                double x = 0.0;
                input >> target >> x;
                if (target.empty() || input.fail()) {
                    std::cout << "Usage: predict linear <x> | predict poly <x>\n";
                    continue;
                }

                if (target == "linear") {
                    if (!linear_trained) {
                        std::cout << "Linear model is not trained.\n";
                        continue;
                    }
                    const Matrix y = linear_model.predict(single_input(x));
                    std::cout << "y=" << y(0, 0) << "\n";
                    continue;
                }

                if (target == "poly") {
                    if (!poly_trained) {
                        std::cout << "Polynomial model is not trained.\n";
                        continue;
                    }
                    const Matrix y = poly_model.predict(single_input(x));
                    std::cout << "y=" << y(0, 0) << "\n";
                    continue;
                }

                std::cout << "Usage: predict linear <x> | predict poly <x>\n";
                continue;
            }

            std::cout << "Unknown command. Type 'help' for a list of commands.\n";
        } catch (const std::exception& ex) {
            std::cout << "Error: " << ex.what() << "\n";
        }
    }

    return 0;
}
