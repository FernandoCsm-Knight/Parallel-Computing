#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <random>
#include <algorithm>
#include <utility>
#include <vector>
#include <cmath>

#include "linear_layer.hpp"
#include "tensor.hpp"

template <Numeric T> class NeuralNetwork {
    private:
        LinearLayer<T> layer1;
        LinearLayer<T> layer2;
        LinearLayer<T> layer3;

        Tensor<T> softmax(const Tensor<T>& input) const;

        T cross_entropy_loss(const Tensor<T>& predictions, const Tensor<T>& targets) const;
        
        Tensor<T> softmax_cross_entropy_grad(const Tensor<T>& predictions, const Tensor<T>& targets) const;

    public:
        NeuralNetwork(int input_features, int hidden_size1, int hidden_size2, int num_classes, float learning_rate);
        
        Tensor<T> forward(const Tensor<T>& input);
        
        void backward(const Tensor<T>& input, const Tensor<T>& targets, const Tensor<T>& predictions);
        
        void train(const Tensor<T>& X, const Tensor<T>& y, int epochs, int batch_size = 32);
        
        T evaluate(const Tensor<T>& X, const Tensor<T>& y);
};

#include "../src/neural_network.tpp"

#endif
