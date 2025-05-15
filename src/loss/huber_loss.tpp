#ifndef HUBER_LOSS_TPP
#define HUBER_LOSS_TPP

#include "../../inc/loss/huber_loss.hpp"

template <Numeric T>
T HuberLoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    int batch_size = predictions.shape(0);
    int total_elements = predictions.length();
    
    #pragma omp parallel for collapse(2) reduction(+:loss)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.shape(1); ++j) {
            T diff = std::abs(predictions(i, j).value() - targets(i, j).value());
            
            if (diff < delta) {
                loss += 0.5 * diff * diff;
            } else {
                loss += delta * (diff - 0.5 * delta);
            }
        }
    }
    
    return loss / total_elements;
}

template <Numeric T>
Tensor<T> HuberLoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> grad(predictions.shape());
    int batch_size = predictions.shape(0);
    int total_elements = predictions.length();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.shape(1); ++j) {
            T diff = predictions(i, j).value() - targets(i, j).value();
            T abs_diff = std::abs(diff);
            
            if (abs_diff < delta) {
                grad(i, j) = diff / total_elements;
            } else {
                grad(i, j) = delta * (diff > 0 ? 1 : -1) / total_elements;
            }
        }
    }
    
    return grad;
}

#endif