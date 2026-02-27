cmake_minimum_required(VERSION 3.16)
project(AnIMaL VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ANIMAL_ENABLE_TESTS "Build unit tests" ON)
option(ANIMAL_ENABLE_COVERAGE "Enable gcov coverage instrumentation" OFF)

add_library(animal STATIC
    src/sequential.cpp
    src/regression.cpp
    src/dataset.cpp
    src/statistics.cpp
    src/bayes.cpp
)

target_include_directories(animal PUBLIC include)

target_compile_options(animal PRIVATE -Wall -Wextra -Wpedantic)

if (ANIMAL_ENABLE_COVERAGE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(animal PRIVATE -O0 -g --coverage)
        add_compile_options(-O0 -g --coverage)
        add_link_options(--coverage)
    else()
        message(FATAL_ERROR "Coverage is only supported with GCC/Clang-style toolchains")
    endif()
endif()

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

if (ANIMAL_ENABLE_TESTS)
    enable_testing()

    add_executable(animal_tests
        tests/test_main.cpp
        tests/test_matrix.cpp
        tests/test_dataset.cpp
        tests/test_statistics.cpp
        tests/test_bayes.cpp
        tests/test_losses_layers.cpp
        tests/test_regression.cpp
    )
    target_link_libraries(animal_tests PRIVATE animal)
    target_compile_options(animal_tests PRIVATE -Wall -Wextra -Wpedantic)

    add_test(NAME animal_tests COMMAND animal_tests)
    set_tests_properties(animal_tests PROPERTIES WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

    if (ANIMAL_ENABLE_COVERAGE)
        add_custom_target(coverage
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            COMMAND gcov -b -c -r
                ${CMAKE_BINARY_DIR}/CMakeFiles/animal.dir/src/sequential.cpp.o
                ${CMAKE_BINARY_DIR}/CMakeFiles/animal.dir/src/regression.cpp.o
                ${CMAKE_BINARY_DIR}/CMakeFiles/animal.dir/src/dataset.cpp.o
                ${CMAKE_BINARY_DIR}/CMakeFiles/animal.dir/src/statistics.cpp.o
                ${CMAKE_BINARY_DIR}/CMakeFiles/animal.dir/src/bayes.cpp.o
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPENDS animal_tests
            COMMENT "Running tests and generating gcov coverage reports"
        )
    endif()
endif()
