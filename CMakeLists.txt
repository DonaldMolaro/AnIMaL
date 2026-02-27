cmake_minimum_required(VERSION 3.16)
project(AnIMaL VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(animal STATIC
    src/sequential.cpp
    src/regression.cpp
    src/dataset.cpp
    src/statistics.cpp
    src/bayes.cpp
)

target_include_directories(animal PUBLIC include)

target_compile_options(animal PRIVATE -Wall -Wextra -Wpedantic)

add_executable(animal_xor examples/xor.cpp)
target_link_libraries(animal_xor PRIVATE animal)

add_executable(animal_multiclass examples/multiclass.cpp)
target_link_libraries(animal_multiclass PRIVATE animal)

add_executable(animal_linear_regression examples/linear_regression.cpp)
target_link_libraries(animal_linear_regression PRIVATE animal)

add_executable(animal_polynomial_regression examples/polynomial_regression.cpp)
target_link_libraries(animal_polynomial_regression PRIVATE animal)

add_executable(animal_shell examples/animal_shell.cpp)
target_link_libraries(animal_shell PRIVATE animal)

add_executable(animal_statistics_basics examples/statistics_basics.cpp)
target_link_libraries(animal_statistics_basics PRIVATE animal)

add_executable(animal_bayes_examples examples/bayes_examples.cpp)
target_link_libraries(animal_bayes_examples PRIVATE animal)

add_executable(animal_correlation_covariance examples/correlation_covariance.cpp)
target_link_libraries(animal_correlation_covariance PRIVATE animal)
