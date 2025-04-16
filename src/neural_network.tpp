#ifndef NEURAL_NETWORK_TPP
#define NEURAL_NETWORK_TPP

#include "../inc/neural_network.hpp"

// Private methods

template <Numeric T>
Tensor<T> NeuralNetwork<T>::softmax(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());
    
    for (int i = 0; i < input.shape(0); ++i) {
        T max_val = input(i, 0).value();
        for (int j = 1; j < input.shape(1); ++j) {
            max_val = std::max(max_val, input(i, j).value());
        }
        
        T sum_exp = 0;
        for (int j = 0; j < input.shape(1); ++j) {
            T exp_val = std::exp(input(i, j).value() - max_val);
            result(i, j) = exp_val;
            sum_exp += exp_val;
        }
        
        for (int j = 0; j < input.shape(1); ++j) {
            result(i, j) = result(i, j).value() / sum_exp;
        }
    }
    
    return result;
}

template <Numeric T>
T NeuralNetwork<T>::cross_entropy_loss(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    int batch_size = predictions.shape(0);
    
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
Tensor<T> NeuralNetwork<T>::softmax_cross_entropy_grad(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> grad(predictions.shape());
    int batch_size = predictions.shape(0);
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.shape(1); ++j) {
            grad(i, j) = (predictions(i, j).value() - targets(i, j).value()) / batch_size;
        }
    }
    
    return grad;
}

// Constructors

template <Numeric T>
NeuralNetwork<T>::NeuralNetwork(int input_features, int hidden_size1, int hidden_size2, int num_classes, float learning_rate)
    : layer1(input_features, hidden_size1, learning_rate),
      layer2(hidden_size1, hidden_size2, learning_rate),
      layer3(hidden_size2, num_classes, learning_rate) {}

// Public methods

template <Numeric T>
Tensor<T> NeuralNetwork<T>::forward(const Tensor<T>& input) {
    Tensor<T> hidden1 = layer1.forward(input);
    Tensor<T> hidden2 = layer2.forward(hidden1);
    Tensor<T> logits = layer3.forward(hidden2);
    return softmax(logits);
}

template <Numeric T>
void NeuralNetwork<T>::backward(const Tensor<T>& input, const Tensor<T>& targets, const Tensor<T>& predictions) {
    Tensor<T> loss_grad = softmax_cross_entropy_grad(predictions, targets);
    
    Tensor<T> grad3 = layer3.backward(loss_grad);
    Tensor<T> grad2 = layer2.backward(grad3);
    layer1.backward(grad2);
}

template <Numeric T>
void NeuralNetwork<T>::train(const Tensor<T>& X, const Tensor<T>& y, int epochs, int batch_size) {
    int n_samples = X.shape(0);
    int n_batches = (n_samples + batch_size - 1) / batch_size;
    
    std::vector<int> indices(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), g);
        
        T total_loss = 0;
        
        for (int batch = 0; batch < n_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, n_samples);
            int current_batch_size = end_idx - start_idx;
            
            Tensor<T> batch_X(current_batch_size, X.shape(1));
            Tensor<T> batch_y(current_batch_size, y.shape(1));
            
            #pragma omp parallel for
            for (int i = 0; i < current_batch_size; ++i) {
                int idx = indices[start_idx + i];
                for (int j = 0; j < X.shape(1); ++j) {
                    batch_X(i, j) = X(idx, j).value();
                }
                for (int j = 0; j < y.shape(1); ++j) {
                    batch_y(i, j) = y(idx, j).value();
                }
            }
            
            Tensor<T> predictions = forward(batch_X);
            
            T loss = cross_entropy_loss(predictions, batch_y);
            total_loss += loss;
            
            backward(batch_X, batch_y, predictions);
        }
        
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            std::cout << "Época " << (epoch + 1) << ", Loss: " << (total_loss / n_batches) << std::endl;
        }
    }
}

template <Numeric T>
T NeuralNetwork<T>::evaluate(const Tensor<T>& X, const Tensor<T>& y) {
    Tensor<T> predictions = forward(X);
    int correct = 0;
    int total = X.shape(0);
    
    for (int i = 0; i < total; ++i) {
        int pred_class = 0;
        T max_prob = predictions(i, 0).value();
        
        for (int j = 1; j < predictions.shape(1); ++j) {
            if (predictions(i, j).value() > max_prob) {
                max_prob = predictions(i, j).value();
                pred_class = j;
            }
        }
        
        int true_class = 0;
        for (int j = 0; j < y.shape(1); ++j) {
            if (y(i, j).value() > 0.5) {
                true_class = j;
                break;
            }
        }
        
        if (pred_class == true_class) {
            correct++;
        }
    }
    
    return static_cast<T>(correct) / total;
}

#endif 
