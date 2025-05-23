#ifndef BINARY_CROSS_ENTROPY_LOSS_TPP
#define BINARY_CROSS_ENTROPY_LOSS_TPP

#include "../../inc/loss/binary_cross_entropy_loss.hpp"

template <Numeric T>
T BinaryCrossEntropyLoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    int batch_size = predictions.shape(0);
    int total_elements = predictions.length();
    
    #pragma omp parallel for collapse(2) reduction(-:loss)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.shape(1); ++j) {
            T pred = std::max(std::min(predictions(i, j).value(), static_cast<T>(1 - 1e-7)), static_cast<T>(1e-7));
            T target = targets(i, j).value();
            
            loss -= target * std::log(pred) + (1 - target) * std::log(1 - pred);
        }
    }
    
    return loss / total_elements;
}

template <Numeric T>
Tensor<T> BinaryCrossEntropyLoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> grad(predictions.shape());
    int batch_size = predictions.shape(0);
    int total_elements = predictions.length();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.shape(1); ++j) {
            T pred = std::max(std::min(predictions(i, j).value(), static_cast<T>(1 - 1e-7)), static_cast<T>(1e-7));
            T target = targets(i, j).value();
            
            grad(i, j) = (-target / pred + (1 - target) / (1 - pred)) / total_elements;
        }
    }
    
    return grad;
}

#endif