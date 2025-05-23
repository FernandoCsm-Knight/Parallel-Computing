#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <random>
#include <utility>
#include <vector>
#include <cmath>

#include "abstract/loss_function.hpp"
#include "abstract/activation.hpp"
#include "abstract/layer.hpp"
#include "layer/sequential.hpp"

#include "tensor/tensor.hpp"

template <Numeric T> class NeuralNetwork {
    private:
        Sequential<T>* model;
        LossFunction<T>* loss_function;

    public:
        NeuralNetwork(Sequential<T>* model, LossFunction<T>* loss_function);
        ~NeuralNetwork();

        Tensor<T> forward(const Tensor<T>& input);
        
        void backward(const Tensor<T>& input, const Tensor<T>& targets, const Tensor<T>& predictions);
        
        void train(const Tensor<T>& X, const Tensor<T>& y, int epochs, int batch_size = 32);
        
        T evaluate(const Tensor<T>& X, const Tensor<T>& y);

        friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork<T>& nn) {
            os << "Summary of Neural Network:" << std::endl;
            os << *nn.model << std::endl;
            os << "Loss Function: " << *nn.loss_function << std::endl;
            return os;
        }
};

#include "../src/neural_network.tpp"

#endif
