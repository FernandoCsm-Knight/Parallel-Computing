#ifndef TENSORLIB_HPP
#define TENSORLIB_HPP   

#include <random>
#include <omp.h>

#include "tensor.hpp"
#include "numeric.hpp"

namespace tensor {
    template <Numeric T, Integral... Dims>
    Tensor<T> zeros(Dims... dims);

    template <Numeric T, Integral... Dims>
    Tensor<T> ones(Dims... dims);

    template <Numeric T, Integral... Dims>
    Tensor<T> rand(Dims... dims);

    template <Numeric T, Integral... Dims>
    Tensor<T> randn(Dims... dims, T mean = 0, T stddev = 1);

    template <Numeric T>
    Tensor<T> arange(T start, T end, T step = 1);

    template <Numeric T, Integral... Dims>
    Tensor<T> range(Dims... dims);

    template <Numeric T>
    Tensor<T> eye(int n);

    template <Numeric T>
    Tensor<T> linspace(T start, T end, int num = 50);

    template <Numeric T, Integral... Dims>
    Tensor<T> full(T value, Dims... dims);
}

#include "../src/tensorlib.tpp"

#endif