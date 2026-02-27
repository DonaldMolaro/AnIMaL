#pragma once

#include <memory>
#include <vector>

#include "animal/layers.hpp"

namespace animal {

class Sequential {
public:
    void add(std::unique_ptr<Layer> layer);
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);
    void step(double learning_rate);

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace animal
