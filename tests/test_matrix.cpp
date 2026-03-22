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

void test_matrix_row_col() {
    animal::Matrix m = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    const animal::Matrix r = m.row(1);
    animal::test::require(r.rows() == 1 && r.cols() == 3, "row shape wrong");
    animal::test::require(animal::test::approx(r(0, 0), 4.0), "row(1) value wrong");
    animal::test::require(animal::test::approx(r(0, 2), 6.0), "row(1) last value wrong");

    const animal::Matrix c = m.col(1);
    animal::test::require(c.rows() == 2 && c.cols() == 1, "col shape wrong");
    animal::test::require(animal::test::approx(c(0, 0), 2.0), "col(1) first value wrong");
    animal::test::require(animal::test::approx(c(1, 0), 5.0), "col(1) second value wrong");

    bool threw = false;
    try { (void)m.row(2); } catch (const std::runtime_error&) { threw = true; }
    animal::test::require(threw, "row out of range should throw");

    threw = false;
    try { (void)m.col(3); } catch (const std::runtime_error&) { threw = true; }
    animal::test::require(threw, "col out of range should throw");
}

void test_matrix_slice() {
    animal::Matrix m = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    const animal::Matrix s = m.slice(0, 2, 1, 3);
    animal::test::require(s.rows() == 2 && s.cols() == 2, "slice shape wrong");
    animal::test::require(animal::test::approx(s(0, 0), 2.0), "slice (0,0) wrong");
    animal::test::require(animal::test::approx(s(1, 1), 6.0), "slice (1,1) wrong");

    bool threw = false;
    try { (void)m.slice(2, 1, 0, 1); } catch (const std::runtime_error&) { threw = true; }
    animal::test::require(threw, "invalid slice range should throw");
}

void test_matrix_stack() {
    animal::Matrix a = {{1.0, 2.0}, {3.0, 4.0}};
    animal::Matrix b = {{5.0}, {6.0}};

    const animal::Matrix h = animal::Matrix::hstack(a, b);
    animal::test::require(h.rows() == 2 && h.cols() == 3, "hstack shape wrong");
    animal::test::require(animal::test::approx(h(0, 2), 5.0), "hstack value wrong");

    animal::Matrix c = {{7.0, 8.0}};
    const animal::Matrix v = animal::Matrix::vstack(a, c);
    animal::test::require(v.rows() == 3 && v.cols() == 2, "vstack shape wrong");
    animal::test::require(animal::test::approx(v(2, 0), 7.0), "vstack value wrong");

    bool threw = false;
    try { (void)animal::Matrix::hstack(a, animal::Matrix(3, 1)); }
    catch (const std::runtime_error&) { threw = true; }
    animal::test::require(threw, "hstack row mismatch should throw");

    threw = false;
    try { (void)animal::Matrix::vstack(a, animal::Matrix(1, 3)); }
    catch (const std::runtime_error&) { threw = true; }
    animal::test::require(threw, "vstack col mismatch should throw");
}

void test_matrix_reshape() {
    animal::Matrix m = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const animal::Matrix r = m.reshape(3, 2);
    animal::test::require(r.rows() == 3 && r.cols() == 2, "reshape shape wrong");
    animal::test::require(animal::test::approx(r(0, 0), 1.0), "reshape value (0,0) wrong");
    animal::test::require(animal::test::approx(r(2, 1), 6.0), "reshape value (2,1) wrong");

    bool threw = false;
    try { (void)m.reshape(4, 4); } catch (const std::runtime_error&) { threw = true; }
    animal::test::require(threw, "reshape size mismatch should throw");
}

void test_matrix_plus_eq_and_free_scalar() {
    animal::Matrix a = {{1.0, 2.0}, {3.0, 4.0}};
    animal::Matrix b = {{10.0, 20.0}, {30.0, 40.0}};
    a += b;
    animal::test::require(animal::test::approx(a(0, 0), 11.0), "+= failed (0,0)");
    animal::test::require(animal::test::approx(a(1, 1), 44.0), "+= failed (1,1)");

    animal::Matrix c = {{2.0, 3.0}};
    const animal::Matrix d = 5.0 * c;
    animal::test::require(animal::test::approx(d(0, 0), 10.0), "free scalar * failed (0,0)");
    animal::test::require(animal::test::approx(d(0, 1), 15.0), "free scalar * failed (0,1)");
}

void test_matrix_sum_min_max_argmax() {
    animal::Matrix m = {{1.0, 5.0, 3.0}, {7.0, 2.0, 4.0}};

    animal::test::require(animal::test::approx(m.sum(), 22.0), "sum failed");

    const animal::Matrix cs = m.column_sum();
    animal::test::require(cs.rows() == 1 && cs.cols() == 3, "column_sum shape wrong");
    animal::test::require(animal::test::approx(cs(0, 0), 8.0), "column_sum col0 wrong");
    animal::test::require(animal::test::approx(cs(0, 1), 7.0), "column_sum col1 wrong");

    animal::test::require(animal::test::approx(m.min(), 1.0), "min failed");
    animal::test::require(animal::test::approx(m.max(), 7.0), "max failed");

    animal::test::require(m.argmax(0) == 1, "argmax row 0 wrong");
    animal::test::require(m.argmax(1) == 0, "argmax row 1 wrong");

    bool threw = false;
    try { (void)m.argmax(2); } catch (const std::runtime_error&) { threw = true; }
    animal::test::require(threw, "argmax out of range should throw");
}

}  // namespace

void register_matrix_tests(std::vector<animal::test::TestCase>& tests) {
    tests.push_back({"matrix_ops", test_matrix_ops});
    tests.push_back({"matrix_row_col", test_matrix_row_col});
    tests.push_back({"matrix_slice", test_matrix_slice});
    tests.push_back({"matrix_stack", test_matrix_stack});
    tests.push_back({"matrix_reshape", test_matrix_reshape});
    tests.push_back({"matrix_plus_eq_and_free_scalar", test_matrix_plus_eq_and_free_scalar});
    tests.push_back({"matrix_sum_min_max_argmax", test_matrix_sum_min_max_argmax});
}
