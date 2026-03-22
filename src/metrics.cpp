#include "animal/metrics.hpp"

#include <stdexcept>

namespace animal {

double accuracy(const Matrix& predictions, const Matrix& labels) {
    if (predictions.rows() != labels.rows()) {
        throw std::runtime_error("accuracy: row count mismatch");
    }
    if (predictions.rows() == 0) {
        throw std::runtime_error("accuracy: inputs must not be empty");
    }

    std::size_t correct = 0;
    const std::size_t n = predictions.rows();

    if (predictions.cols() == 1 && labels.cols() == 1) {
        for (std::size_t r = 0; r < n; ++r) {
            const std::size_t pred_class = predictions(r, 0) >= 0.5 ? 1 : 0;
            const std::size_t label_class = labels(r, 0) >= 0.5 ? 1 : 0;
            if (pred_class == label_class) ++correct;
        }
    } else {
        if (predictions.cols() != labels.cols()) {
            throw std::runtime_error("accuracy: column count mismatch");
        }
        for (std::size_t r = 0; r < n; ++r) {
            if (predictions.argmax(r) == labels.argmax(r)) ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(n);
}

Matrix confusion_matrix(const Matrix& predictions, const Matrix& labels,
                        std::size_t num_classes) {
    if (predictions.rows() != labels.rows()) {
        throw std::runtime_error("confusion_matrix: row count mismatch");
    }
    if (num_classes == 0) {
        throw std::runtime_error("confusion_matrix: num_classes must be > 0");
    }

    Matrix cm(num_classes, num_classes, 0.0);
    const std::size_t n = predictions.rows();

    for (std::size_t r = 0; r < n; ++r) {
        std::size_t pred_class, true_class;

        if (predictions.cols() == 1) {
            pred_class = predictions(r, 0) >= 0.5 ? 1 : 0;
            true_class = labels(r, 0) >= 0.5 ? 1 : 0;
        } else {
            pred_class = predictions.argmax(r);
            true_class = labels.argmax(r);
        }

        if (true_class >= num_classes || pred_class >= num_classes) {
            throw std::runtime_error("confusion_matrix: class index exceeds num_classes");
        }

        cm(true_class, pred_class) += 1.0;
    }

    return cm;
}

}  // namespace animal
