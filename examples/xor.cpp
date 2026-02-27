#include <iomanip>
#include <iostream>
#include <memory>

#include "animal/layers.hpp"
#include "animal/losses.hpp"
#include "animal/sequential.hpp"

int main() {
    using namespace animal;

    Matrix x = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
    };

    Matrix y = {
        {0.0},
        {1.0},
        {1.0},
        {0.0},
    };

    Sequential model;
    model.add(std::make_unique<Dense>(2, 8));
    model.add(std::make_unique<Sigmoid>());
    model.add(std::make_unique<Dense>(8, 1));
    model.add(std::make_unique<Sigmoid>());

    MSELoss loss;

    const int epochs = 20000;
    const double learning_rate = 0.5;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        Matrix prediction = model.forward(x);
        double train_loss = loss.forward(prediction, y);

        Matrix grad = loss.backward(prediction, y);
        model.backward(grad);
        model.step(learning_rate);

        if (epoch % 2000 == 0) {
            std::cout << "Epoch " << std::setw(4) << epoch << " | Loss: " << train_loss << '\n';
        }
    }

    Matrix prediction = model.forward(x);
    std::cout << "\nFinal predictions (XOR):\n";
    std::cout << prediction << '\n';

    return 0;
}
