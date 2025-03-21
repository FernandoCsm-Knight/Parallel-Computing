#ifndef LINEARLAYER_HPP
#define LINEARLAYER_HPP

#include "numeric.hpp"
#include "tensor.hpp"

template <Numeric T>
class LinearLayer {
    private:
        int in_features;
        int out_features;
        bool use_bias;
        Tensor<T> weights;
        Tensor<T> bias;

    public:
        LinearLayer(int in_features, int out_features, bool bias = true);

        Tensor<T> forward(const Tensor<T>& input);

        void update_parameters(T learning_rate, const Tensor<T>& grad_weights, const Tensor<T>& grad_bias);

        const Tensor<T>& get_weights() const;

        const Tensor<T>& get_bias() const;
};

#include "../src/linear_layer.tpp"

#endif
