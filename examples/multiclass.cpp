#include <iomanip>
#include <iostream>
#include <memory>

#include "animal/layers.hpp"
#include "animal/losses.hpp"
#include "animal/sequential.hpp"

namespace {

std::size_t argmax_row(const animal::Matrix& m, std::size_t row) {
    std::size_t best_idx = 0;
    double best_val = m(row, 0);
    for (std::size_t c = 1; c < m.cols(); ++c) {
        if (m(row, c) > best_val) {
            best_val = m(row, c);
            best_idx = c;
        }
    }
    return best_idx;
}

double accuracy(const animal::Matrix& prediction, const animal::Matrix& target) {
    std::size_t correct = 0;
    for (std::size_t r = 0; r < prediction.rows(); ++r) {
        if (argmax_row(prediction, r) == argmax_row(target, r)) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(prediction.rows());
}

}  // namespace

int main() {
    using namespace animal;

    // Three separable classes in 2D.
    Matrix x = {
        {-1.6, -1.2},
        {-1.2, -1.8},
        {-1.0, -1.1},
        {1.4, -1.0},
        {1.8, -1.6},
        {1.1, -1.3},
        {0.0, 1.2},
        {0.5, 1.8},
        {-0.6, 1.5},
    };

    // One-hot labels for classes 0, 1, and 2.
    Matrix y = {
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 1.0},
    };

    Sequential model;
    model.add(std::make_unique<Dense>(2, 12));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Dense>(12, 3));
    model.add(std::make_unique<Softmax>());

    CrossEntropyLoss loss;

    const int epochs = 2500;
    const double learning_rate = 0.08;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        Matrix prediction = model.forward(x);
        const double train_loss = loss.forward(prediction, y);

        Matrix grad = loss.backward(prediction, y);
        model.backward(grad);
        model.step(learning_rate);

        if (epoch % 250 == 0) {
            const double train_acc = accuracy(prediction, y);
            std::cout << "Epoch " << std::setw(4) << epoch
                      << " | Loss: " << train_loss
                      << " | Acc: " << std::fixed << std::setprecision(3) << train_acc << '\n';
        }
    }

    Matrix prediction = model.forward(x);
    std::cout << "\nFinal class probabilities:\n" << prediction << "\n";
    std::cout << "Final accuracy: " << std::fixed << std::setprecision(3) << accuracy(prediction, y) << '\n';

    return 0;
}
