# AnIMaL

An educational C++ framework for learning machine learning fundamentals from scratch.

## What is included

- `Matrix` math utilities (basic tensor-like operations)
- Fully connected (`Dense`) neural network layer
- Activation layers: `ReLU`, `Sigmoid`, `Softmax`
- Loss functions: `MSELoss`, `CrossEntropyLoss`
- `Sequential` model container
- Regression models: `LinearRegression`, `PolynomialRegression`
- Statistics functions: `mean`, `median`, `mode`
- Relationship statistics: `covariance`, `pearson_correlation`
- Bayes rule utilities: posterior and total probability
- End-to-end XOR training example
- End-to-end multiclass classification example
- End-to-end linear regression example
- End-to-end polynomial regression example
- CSV dataset loader utility (`load_xy_csv`)
- Interactive teaching shell for guided regression experiments

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run the examples

```bash
./build/animal_xor
./build/animal_multiclass
./build/animal_linear_regression
./build/animal_polynomial_regression
./build/animal_shell
./build/animal_statistics_basics
./build/animal_bayes_examples
./build/animal_correlation_covariance
```

Regression examples train from CSV files in `/Users/donaldmolaro/src/AnIMaL/data` by default:

- `/Users/donaldmolaro/src/AnIMaL/data/linear_regression.csv`
- `/Users/donaldmolaro/src/AnIMaL/data/polynomial_regression.csv`

You can also pass a custom CSV path:

```bash
./build/animal_linear_regression /path/to/your_linear.csv
./build/animal_polynomial_regression /path/to/your_poly.csv
```

## Interactive shell

Start:

```bash
./build/animal_shell
```

Typical workflow:

```text
load linear data/linear_regression.csv
fit linear
show linear
predict linear 13
load poly data/polynomial_regression.csv 2
fit poly
show poly
predict poly 2.5
```

Lesson mode workflow:

```text
lesson linear
lesson poly
```

Optional custom lesson inputs:

```text
lesson linear /path/to/linear.csv
lesson poly /path/to/poly.csv 3
```

Statistics commands in shell:

```text
stats 2 4 4 4 5 5 7 9
stats y linear
stats y poly
stats relation linear
stats relation poly
```

Bayes commands in shell:

```text
bayes posterior 0.95 0.01 0.059
bayes from_likelihood 0.95 0.01 0.05
lesson bayes
lesson correlation
```

## Suggested next steps

- Add mini-batch training and data shuffling
- Add model save/load (checkpointing)
- Add unit tests for matrix math and gradient checks
