#include <cmath>
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

void test_tanh_layer() {
    animal::Tanh tanh_layer;
    animal::Matrix input = {{-1.0, 0.0, 1.0}};
    const animal::Matrix out = tanh_layer.forward(input);

    animal::test::require(animal::test::approx(out(0, 1), 0.0), "tanh(0) should be 0");
    animal::test::require(out(0, 0) < 0.0, "tanh(-1) should be negative");
    animal::test::require(out(0, 2) > 0.0, "tanh(1) should be positive");
    animal::test::require(animal::test::approx(out(0, 2), std::tanh(1.0)), "tanh(1) value wrong");

    animal::Matrix grad = {{1.0, 1.0, 1.0}};
    const animal::Matrix grad_in = tanh_layer.backward(grad);
    const double y1 = std::tanh(1.0);
    animal::test::require(animal::test::approx(grad_in(0, 1), 1.0), "tanh grad at 0 should be 1");
    animal::test::require(animal::test::approx(grad_in(0, 2), 1.0 - y1 * y1, 1e-5),
                          "tanh grad at 1 wrong");
}

void test_dropout_layer() {
    animal::Dropout drop(0.5);
    animal::Matrix input(1, 100, 1.0);

    drop.set_training(true);
    const animal::Matrix out_train = drop.forward(input);
    double train_sum = 0.0;
    int zeros = 0;
    for (std::size_t c = 0; c < 100; ++c) {
        train_sum += out_train(0, c);
        if (out_train(0, c) == 0.0) ++zeros;
    }
    animal::test::require(zeros > 10, "dropout should zero out some values during training");

    drop.set_training(false);
    const animal::Matrix out_eval = drop.forward(input);
    for (std::size_t c = 0; c < 100; ++c) {
        animal::test::require(animal::test::approx(out_eval(0, c), 1.0),
                              "dropout should pass through during eval");
    }
}

void test_batchnorm_layer() {
    animal::BatchNorm bn(2);
    animal::Matrix input = {{1.0, 10.0}, {3.0, 20.0}, {5.0, 30.0}, {7.0, 40.0}};

    bn.set_training(true);
    const animal::Matrix out = bn.forward(input);
    animal::test::require(out.rows() == 4 && out.cols() == 2, "batchnorm output shape wrong");

    double col0_sum = 0.0;
    for (std::size_t r = 0; r < out.rows(); ++r) {
        col0_sum += out(r, 0);
    }
    animal::test::require(animal::test::approx(col0_sum / 4.0, 0.0, 1e-5),
                          "batchnorm output should have mean ~0");

    animal::Matrix grad(4, 2, 1.0);
    const animal::Matrix grad_in = bn.backward(grad);
    animal::test::require(grad_in.rows() == 4 && grad_in.cols() == 2,
                          "batchnorm backward shape wrong");
}

void test_binary_cross_entropy() {
    animal::BinaryCrossEntropyLoss bce;

    animal::Matrix pred = {{0.9}, {0.1}};
    animal::Matrix target = {{1.0}, {0.0}};
    const double loss = bce.forward(pred, target);
    animal::test::require(loss > 0.0, "BCE loss should be positive");
    animal::test::require(loss < 0.3, "BCE loss for good predictions should be small");

    animal::Matrix bad_pred = {{0.1}, {0.9}};
    const double bad_loss = bce.forward(bad_pred, target);
    animal::test::require(bad_loss > loss, "BCE loss for bad predictions should be larger");

    const animal::Matrix grad = bce.backward(pred, target);
    animal::test::require(grad.rows() == 2 && grad.cols() == 1, "BCE gradient shape wrong");
}

void test_sequential_train_api() {
    animal::Sequential model;
    model.add(std::make_unique<animal::Dense>(2, 4));
    model.add(std::make_unique<animal::ReLU>());
    model.add(std::make_unique<animal::Dense>(4, 1));
    model.add(std::make_unique<animal::Sigmoid>());

    animal::Matrix x = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    animal::Matrix y = {{0.0}, {1.0}, {1.0}, {0.0}};
    animal::BinaryCrossEntropyLoss bce;

    const std::vector<double> history = model.train(x, y, bce, 100, 0.5);
    animal::test::require(history.size() == 100, "train history should have 100 entries");
    animal::test::require(history.back() < history.front(), "loss should decrease during training");

    const animal::Matrix pred = model.predict(x);
    animal::test::require(pred.rows() == 4 && pred.cols() == 1, "predict shape wrong");

    const double eval_loss = model.evaluate(x, y, bce);
    animal::test::require(eval_loss >= 0.0, "evaluate loss should be non-negative");
}

}  // namespace

void register_losses_layers_tests(std::vector<animal::test::TestCase>& tests) {
    tests.push_back({"losses_and_layers", test_losses_and_layers});
    tests.push_back({"tanh_layer", test_tanh_layer});
    tests.push_back({"dropout_layer", test_dropout_layer});
    tests.push_back({"batchnorm_layer", test_batchnorm_layer});
    tests.push_back({"binary_cross_entropy", test_binary_cross_entropy});
    tests.push_back({"sequential_train_api", test_sequential_train_api});
}
