#ifndef MAE_LOSS_TPP
#define MAE_LOSS_TPP

#include "../../inc/loss/mae_loss.hpp"

template <Numeric T>
T MAELoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;

    Tensor<T> temp;
    if (predictions.ndim() == 1) {
        temp = predictions.reshape(Shape(predictions.shape(0), 1));
    } else if (predictions.ndim() == 2) {
        temp = predictions;
    } else {
        throw std::invalid_argument("Invalid tensor shape for MAELoss");
    }

    int batch_size = temp.shape(0);
    int total_elements = temp.length();
    
    #pragma omp parallel for collapse(2) reduction(+:loss)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < temp.shape(1); ++j) {
            loss += std::abs(temp(i, j).value() - targets(i, j).value());
        }
    }
    
    return loss / total_elements;
}

template <Numeric T>
Tensor<T> MAELoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> temp;
    if (predictions.ndim() == 1) {
        temp = predictions.reshape(Shape(predictions.shape(0), 1));
    } else if (predictions.ndim() == 2) {
        temp = predictions;
    } else {
        throw std::invalid_argument("Invalid tensor shape for MAELoss");
    }

    Tensor<T> grad(temp.shape());
    int batch_size = temp.shape(0);
    int total_elements = temp.length();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < temp.shape(1); ++j) {
            T diff = temp(i, j).value() - targets(i, j).value();
            grad(i, j) = (diff > 0 ? 1 : (diff < 0 ? -1 : 0)) / total_elements;
        }
    }
    
    return grad;
}

#endif