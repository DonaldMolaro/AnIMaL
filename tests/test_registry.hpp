#pragma once

#include <vector>

#include "test_common.hpp"

void register_matrix_tests(std::vector<animal::test::TestCase>& tests);
void register_dataset_tests(std::vector<animal::test::TestCase>& tests);
void register_statistics_tests(std::vector<animal::test::TestCase>& tests);
void register_bayes_tests(std::vector<animal::test::TestCase>& tests);
void register_losses_layers_tests(std::vector<animal::test::TestCase>& tests);
void register_regression_tests(std::vector<animal::test::TestCase>& tests);
