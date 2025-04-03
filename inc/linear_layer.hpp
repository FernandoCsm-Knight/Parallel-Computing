#ifndef LINEARLAYER_HPP
#define LINEARLAYER_HPP

#include "numeric.hpp"
#include "tensor.hpp"

template <Numeric T>
class LinearLayer {
    private:
        int in_features;
        int out_features;
        float learning_rate;
        Tensor<T> last_input;
        Tensor<T> last_output;
        Tensor<T> weights;
        Tensor<T> bias;

    public:
        LinearLayer(int in_features, int out_features, float lr);

        Tensor<T> forward(const Tensor<T>& input);

        Tensor<T> backward(const Tensor<T>& grad_weights);

};

#include "../src/linear_layer.tpp"

#endif
