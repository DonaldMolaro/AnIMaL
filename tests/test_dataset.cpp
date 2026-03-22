#include <fstream>
#include <vector>

#include "animal/dataset.hpp"
#include "test_common.hpp"
#include "test_registry.hpp"

namespace {

void test_dataset_loading() {
    const animal::Dataset d = animal::load_xy_csv("data/linear_regression.csv", 1);
    animal::test::require(d.x.rows() == 15 && d.x.cols() == 1, "dataset x shape invalid");
    animal::test::require(d.y.rows() == 15 && d.y.cols() == 1, "dataset y shape invalid");

    const std::string path = "/tmp/animal_csv_header_test.csv";
    {
        std::ofstream out(path);
        out << "x,y\n";
        out << "1,2\n";
        out << "3,4\n";
    }
    const animal::Dataset with_header = animal::load_xy_csv(path, 1, true);
    animal::test::require(with_header.x.rows() == 2, "dataset header parse failed");

    bool threw = false;
    try {
        (void)animal::load_xy_csv("/tmp/does_not_exist.csv", 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    animal::test::require(threw, "load missing csv should throw");
}

void test_train_test_split() {
    animal::Matrix x(10, 2);
    animal::Matrix y(10, 1);
    for (std::size_t i = 0; i < 10; ++i) {
        x(i, 0) = static_cast<double>(i);
        x(i, 1) = static_cast<double>(i) * 2.0;
        y(i, 0) = static_cast<double>(i) * 10.0;
    }
    animal::Dataset data{x, y};

    auto [train, test] = animal::train_test_split(data, 0.3, 42);
    animal::test::require(train.x.rows() == 7, "train split should have 7 rows");
    animal::test::require(test.x.rows() == 3, "test split should have 3 rows");
    animal::test::require(train.x.cols() == 2, "train x cols wrong");
    animal::test::require(test.y.cols() == 1, "test y cols wrong");

    bool threw = false;
    try { (void)animal::train_test_split(data, 0.0); }
    catch (const std::runtime_error&) { threw = true; }
    animal::test::require(threw, "test_ratio 0 should throw");
}

void test_shuffle_dataset() {
    animal::Matrix x(5, 1);
    animal::Matrix y(5, 1);
    for (std::size_t i = 0; i < 5; ++i) {
        x(i, 0) = static_cast<double>(i);
        y(i, 0) = static_cast<double>(i);
    }
    animal::Dataset data{x, y};

    const animal::Dataset shuffled = animal::shuffle_dataset(data, 123);
    animal::test::require(shuffled.x.rows() == 5, "shuffled row count wrong");

    double x_sum = 0.0;
    for (std::size_t i = 0; i < 5; ++i) {
        x_sum += shuffled.x(i, 0);
    }
    animal::test::require(animal::test::approx(x_sum, 10.0), "shuffle changed element values");
}

void test_normalize_standardize() {
    animal::Matrix m = {{0.0, 10.0}, {5.0, 20.0}, {10.0, 30.0}};

    const animal::Matrix normed = animal::normalize(m);
    animal::test::require(animal::test::approx(normed(0, 0), 0.0), "normalize min should be 0");
    animal::test::require(animal::test::approx(normed(2, 0), 1.0), "normalize max should be 1");
    animal::test::require(animal::test::approx(normed(1, 1), 0.5), "normalize mid should be 0.5");

    const animal::Matrix std = animal::standardize(m);
    double col0_sum = 0.0;
    for (std::size_t r = 0; r < std.rows(); ++r) {
        col0_sum += std(r, 0);
    }
    animal::test::require(animal::test::approx(col0_sum / 3.0, 0.0, 1e-10),
                          "standardized column should have mean ~0");
}

}  // namespace

void register_dataset_tests(std::vector<animal::test::TestCase>& tests) {
    tests.push_back({"dataset_loading", test_dataset_loading});
    tests.push_back({"train_test_split", test_train_test_split});
    tests.push_back({"shuffle_dataset", test_shuffle_dataset});
    tests.push_back({"normalize_standardize", test_normalize_standardize});
}
