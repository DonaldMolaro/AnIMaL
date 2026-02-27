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

}  // namespace animal
