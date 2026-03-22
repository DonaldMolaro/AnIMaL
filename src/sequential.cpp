#include "animal/sequential.hpp"

#include <stdexcept>

namespace animal {

void Sequential::add(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

Matrix Sequential::forward(const Matrix& input) {
    Matrix output = input;
    for (const auto& layer : layers_) {
        output = layer->forward(output);
    }
    return output;
}

Matrix Sequential::backward(const Matrix& grad_output) {
    Matrix grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
    return grad;
}

void Sequential::step(double learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::runtime_error("learning_rate must be > 0");
    }

    for (const auto& layer : layers_) {
        layer->update(learning_rate);
    }
}

Matrix Sequential::predict(const Matrix& input) {
    return forward(input);
}

double Sequential::evaluate(const Matrix& input, const Matrix& target, Loss& loss) {
    const Matrix output = forward(input);
    return loss.forward(output, target);
}

std::vector<double> Sequential::train(const Matrix& input, const Matrix& target,
                                       Loss& loss, int epochs, double learning_rate) {
    if (epochs <= 0) {
        throw std::runtime_error("train: epochs must be > 0");
    }
    if (learning_rate <= 0.0) {
        throw std::runtime_error("train: learning_rate must be > 0");
    }

    std::vector<double> history;
    history.reserve(static_cast<std::size_t>(epochs));

    for (int e = 0; e < epochs; ++e) {
        const Matrix output = forward(input);
        const double loss_val = loss.forward(output, target);
        history.push_back(loss_val);

        const Matrix grad = loss.backward(output, target);
        backward(grad);
        step(learning_rate);
    }

    return history;
}

}  // namespace animal
