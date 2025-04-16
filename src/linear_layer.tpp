#ifndef LINEARLAYER_TPP 
#define LINEARLAYER_TPP

#include "../inc/linear_layer.hpp"
#include "../inc/tensorlib.hpp"

#include <random>

template <Numeric T>
Tensor<T> relu(const Tensor<T>& input) {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = std::max(static_cast<T>(0), input[i]);
    }

    return result;
}

template <Numeric T>
Tensor<T> relu_derivative(const Tensor<T>& input) {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = (input[i] > static_cast<T>(0)) ? static_cast<T>(1) : static_cast<T>(0);
    }

    return result;
}

template <Numeric T>
LinearLayer<T>::LinearLayer(int in_features, int out_features, float lr)
    : in_features(in_features), out_features(out_features), learning_rate(lr) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist(0, std::sqrt(2.0 / in_features));
    
    weights = Tensor<T>(in_features, out_features);
    
    #pragma omp parallel for
    for (size_t i = 0; i < weights.length(); ++i) {
        weights[i] = dist(gen);
    }

    bias = tensor::ones<T>(1, out_features);
}

template <Numeric T>
Tensor<T> LinearLayer<T>::forward(const Tensor<T>& input) {
    if (input.shape(input.ndim() - 1) != in_features) {
        throw std::invalid_argument("Input features don't match layer's in_features");
    }
    
    this->last_input = input;
    Tensor<T> dot_result = input.dot(weights) + bias;
    this->last_output = relu(dot_result);
    
    return this->last_output;
}

template <Numeric T>
Tensor<T> LinearLayer<T>::backward(const Tensor<T>& grad_weights) {
    Tensor<T> delta = grad_weights * relu_derivative(last_output);
    Tensor<T> previous_grad = delta.dot(weights.transpose());
    
    weights -= (last_input.transpose().dot(delta) * learning_rate);
    bias -= (delta.sum(0, true) * learning_rate);

    return previous_grad;
}

#endif
