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

}  // namespace

void register_dataset_tests(std::vector<animal::test::TestCase>& tests) {
    tests.push_back({"dataset_loading", test_dataset_loading});
}
