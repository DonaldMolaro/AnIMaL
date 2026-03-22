#include <vector>

#include "animal/metrics.hpp"
#include "test_common.hpp"
#include "test_registry.hpp"

namespace {

void test_accuracy_binary() {
    animal::Matrix pred = {{0.9}, {0.1}, {0.8}, {0.3}};
    animal::Matrix labels = {{1.0}, {0.0}, {1.0}, {0.0}};
    animal::test::require(animal::test::approx(animal::accuracy(pred, labels), 1.0),
                          "binary accuracy should be 1.0 for correct predictions");

    animal::Matrix bad_pred = {{0.1}, {0.9}, {0.2}, {0.7}};
    animal::test::require(animal::test::approx(animal::accuracy(bad_pred, labels), 0.0),
                          "binary accuracy should be 0.0 for all wrong");
}

void test_accuracy_multiclass() {
    animal::Matrix pred = {{0.1, 0.8, 0.1}, {0.7, 0.2, 0.1}, {0.1, 0.1, 0.8}};
    animal::Matrix labels = {{0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}};
    animal::test::require(animal::test::approx(animal::accuracy(pred, labels), 1.0),
                          "multiclass accuracy should be 1.0");

    animal::Matrix mixed_pred = {{0.1, 0.8, 0.1}, {0.2, 0.7, 0.1}, {0.1, 0.1, 0.8}};
    animal::test::require(animal::test::approx(animal::accuracy(mixed_pred, labels), 2.0 / 3.0, 1e-5),
                          "multiclass accuracy should be 2/3");
}

void test_confusion_matrix() {
    animal::Matrix pred = {{0.9}, {0.1}, {0.8}, {0.6}};
    animal::Matrix labels = {{1.0}, {0.0}, {0.0}, {1.0}};
    const animal::Matrix cm = animal::confusion_matrix(pred, labels, 2);

    animal::test::require(cm.rows() == 2 && cm.cols() == 2, "confusion matrix shape wrong");
    animal::test::require(animal::test::approx(cm(0, 0), 1.0), "true negatives wrong");
    animal::test::require(animal::test::approx(cm(0, 1), 1.0), "false positives wrong");
    animal::test::require(animal::test::approx(cm(1, 0), 0.0), "false negatives wrong");
    animal::test::require(animal::test::approx(cm(1, 1), 2.0), "true positives wrong");

    double total = cm.sum();
    animal::test::require(animal::test::approx(total, 4.0), "confusion matrix total should equal sample count");
}

void test_confusion_matrix_multiclass() {
    animal::Matrix pred = {{0.8, 0.1, 0.1}, {0.1, 0.1, 0.8}, {0.1, 0.8, 0.1}};
    animal::Matrix labels = {{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}};
    const animal::Matrix cm = animal::confusion_matrix(pred, labels, 3);

    animal::test::require(cm.rows() == 3 && cm.cols() == 3, "3-class confusion matrix shape wrong");
    animal::test::require(animal::test::approx(cm(0, 0), 1.0), "class 0 correct");
    animal::test::require(animal::test::approx(cm(1, 1), 1.0), "class 1 correct");
    animal::test::require(animal::test::approx(cm(2, 2), 1.0), "class 2 correct");
    animal::test::require(animal::test::approx(cm.sum(), 3.0), "total should be 3");
}

}  // namespace

void register_metrics_tests(std::vector<animal::test::TestCase>& tests) {
    tests.push_back({"accuracy_binary", test_accuracy_binary});
    tests.push_back({"accuracy_multiclass", test_accuracy_multiclass});
    tests.push_back({"confusion_matrix", test_confusion_matrix});
    tests.push_back({"confusion_matrix_multiclass", test_confusion_matrix_multiclass});
}
