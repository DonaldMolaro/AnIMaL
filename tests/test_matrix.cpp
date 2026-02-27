#include <vector>

#include "animal/matrix.hpp"
#include "test_common.hpp"
#include "test_registry.hpp"

namespace {

void test_matrix_ops() {
    animal::Matrix a = {{1.0, 2.0}, {3.0, 4.0}};
    animal::Matrix b = {{5.0, 6.0}, {7.0, 8.0}};

    const animal::Matrix sum = a + b;
    animal::test::require(animal::test::approx(sum(1, 1), 12.0), "matrix add failed");

    const animal::Matrix diff = b - a;
    animal::test::require(animal::test::approx(diff(0, 0), 4.0), "matrix subtract failed");

    const animal::Matrix prod = a.matmul(b);
    animal::test::require(animal::test::approx(prod(0, 0), 19.0), "matrix matmul failed (0,0)");
    animal::test::require(animal::test::approx(prod(1, 1), 50.0), "matrix matmul failed (1,1)");

    const animal::Matrix t = a.transpose();
    animal::test::require(animal::test::approx(t(1, 0), 2.0), "matrix transpose failed");

    const animal::Matrix row = {{10.0, -1.0}};
    const animal::Matrix shifted = a.add_row_vector(row);
    animal::test::require(animal::test::approx(shifted(1, 0), 13.0), "matrix add_row_vector failed");

    const animal::Matrix mean_cols = a.column_mean();
    animal::test::require(animal::test::approx(mean_cols(0, 0), 2.0), "matrix column_mean col0 failed");
    animal::test::require(animal::test::approx(mean_cols(0, 1), 3.0), "matrix column_mean col1 failed");

    bool threw = false;
    try {
        (void)a.matmul(animal::Matrix(3, 1));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    animal::test::require(threw, "matrix matmul mismatch should throw");
}

}  // namespace

void register_matrix_tests(std::vector<animal::test::TestCase>& tests) {
    tests.push_back({"matrix_ops", test_matrix_ops});
}
