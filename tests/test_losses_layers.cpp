#include <memory>
#include <vector>

#include "animal/layers.hpp"
#include "animal/losses.hpp"
#include "animal/sequential.hpp"
#include "test_common.hpp"
#include "test_registry.hpp"

namespace {

void test_losses_and_layers() {
    animal::MSELoss mse;
    animal::Matrix pred = {{1.0}, {3.0}};
    animal::Matrix target = {{2.0}, {1.0}};
    animal::test::require(animal::test::approx(mse.forward(pred, target), 2.5), "MSE forward failed");

    const animal::Matrix grad = mse.backward(pred, target);
    animal::test::require(animal::test::approx(grad(0, 0), -1.0), "MSE backward elem 0 failed");
    animal::test::require(animal::test::approx(grad(1, 0), 2.0), "MSE backward elem 1 failed");

    animal::Softmax s;
    animal::Matrix logits = {{1.0, 2.0, 3.0}};
    const animal::Matrix probs = s.forward(logits);
    const double sum = probs(0, 0) + probs(0, 1) + probs(0, 2);
    animal::test::require(animal::test::approx(sum, 1.0), "softmax probabilities should sum to 1");

    animal::CrossEntropyLoss ce;
    animal::Matrix y = {{0.0, 0.0, 1.0}};
    const double loss = ce.forward(probs, y);
    animal::test::require(loss > 0.0, "cross entropy should be positive");
    const animal::Matrix ce_grad = ce.backward(probs, y);
    animal::test::require(ce_grad.rows() == 1 && ce_grad.cols() == 3, "cross entropy grad shape invalid");

    animal::Sequential model;
    model.add(std::make_unique<animal::Dense>(2, 4));
    model.add(std::make_unique<animal::ReLU>());
    model.add(std::make_unique<animal::Dense>(4, 1));

    animal::Matrix x = {{1.0, 2.0}, {0.5, -1.0}};
    const animal::Matrix out = model.forward(x);
    animal::test::require(out.rows() == 2 && out.cols() == 1, "sequential output shape invalid");

    const animal::Matrix grad_out(out.rows(), out.cols(), 1.0);
    const animal::Matrix grad_in = model.backward(grad_out);
    animal::test::require(grad_in.rows() == 2 && grad_in.cols() == 2, "sequential backward shape invalid");
    model.step(0.01);
}

}  // namespace

void register_losses_layers_tests(std::vector<animal::test::TestCase>& tests) {
    tests.push_back({"losses_and_layers", test_losses_and_layers});
}
