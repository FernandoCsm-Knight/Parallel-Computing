#ifndef CROSS_ENTROPY_LOSS_TPP
#define CROSS_ENTROPY_LOSS_TPP

#include "../../inc/loss/cross_entropy_loss.hpp"

template <Numeric T>
T CrossEntropyLoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    int batch_size = predictions.shape(0);
    
    #pragma omp parallel for collapse(2) reduction(-:loss)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.shape(1); ++j) {
            if (targets(i, j).value() > 0) {
                T pred = std::max(predictions(i, j).value(), static_cast<T>(1e-7));
                loss -= targets(i, j).value() * std::log(pred);
            }
        }
    }
    
    return loss / batch_size;
}

template <Numeric T>
Tensor<T> CrossEntropyLoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> grad(predictions.shape());
    int batch_size = predictions.shape(0);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.shape(1); ++j) {
            grad(i, j) = (predictions(i, j).value() - targets(i, j).value()) / batch_size;
        }
    }
    
    return grad;
}

#endif