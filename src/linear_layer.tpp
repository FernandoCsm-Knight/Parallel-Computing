#ifndef LINEARLAYER_TPP 
#define LINEARLAYER_TPP

#include "../inc/linear_layer.hpp"
#include <random>

template <Numeric T>
LinearLayer<T>::LinearLayer(int in_features, int out_features, bool bias)
    : in_features(in_features), out_features(out_features), use_bias(bias) {
    
    // Initialize weights using He initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist(0, std::sqrt(2.0 / in_features));
    
    // Create weight tensor
    weights = Tensor<T>(in_features, out_features);
    
    // Initialize weights with random values
    for (size_t i = 0; i < weights.length(); ++i) {
        weights[i] = dist(gen);
    }
    
    // Create and initialize bias if needed
    if (use_bias) {
        this->bias = Tensor<T>(out_features);
        // Initialize bias with zeros
        for (size_t i = 0; i < this->bias.length(); ++i) {
            this->bias[i] = 0.0;
        }
    }
}

template <Numeric T>
Tensor<T> LinearLayer<T>::forward(const Tensor<T>& input) {
    // Check input dimensions
    if (input.shape(input.ndim() - 1) != in_features) {
        throw std::invalid_argument("Input features don't match layer's in_features");
    }
    
    // Linear transformation: y = xW + b
    Tensor<T> output = input.dot(weights);
    
    // Add bias if needed
    if (use_bias) {
        // Broadcast and add bias to each output
        for (size_t i = 0; i < output.length(); ++i) {
            output[i] += bias[i % bias.length()];
        }
    }
    
    return output;
}

template <Numeric T>
void LinearLayer<T>::update_parameters(T learning_rate, const Tensor<T>& grad_weights, const Tensor<T>& grad_bias) {
    // Update weights: w = w - lr * grad_w
    for (size_t i = 0; i < weights.length(); ++i) {
        weights[i] -= learning_rate * grad_weights[i];
    }
    
    // Update bias if used
    if (use_bias) {
        for (size_t i = 0; i < bias.length(); ++i) {
            bias[i] -= learning_rate * grad_bias[i];
        }
    }
}

template <Numeric T>
const Tensor<T>& LinearLayer<T>::get_weights() const {
    return weights;
}

template <Numeric T>
const Tensor<T>& LinearLayer<T>::get_bias() const {
    if (!use_bias) {
        throw std::runtime_error("This layer does not use bias");
    }
    return bias;
}

#endif