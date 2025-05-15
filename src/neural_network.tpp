#ifndef NEURAL_NETWORK_TPP
#define NEURAL_NETWORK_TPP

#include "../inc/neural_network.hpp"

// Construtor

template <Numeric T>
NeuralNetwork<T>::NeuralNetwork(Sequential<T>* model, LossFunction<T>* loss_function)
    : model(model), loss_function(loss_function) {

    if(model == nullptr) {
        throw std::invalid_argument("Model cannot be null");
    }

    if(loss_function == nullptr) {
        throw std::invalid_argument("Loss function cannot be null");
    }

    if(model->empty()) {
        throw std::invalid_argument("Model cannot be empty");
    }

    if(model->last()->is_activation()) {
        dynamic_cast<Activation<T>*>(model->last())->set_last_activation(true);
    }
}

// Destructor

template <Numeric T>
NeuralNetwork<T>::~NeuralNetwork() {
    delete model;
    delete loss_function;
}

// Methods

template <Numeric T>
Tensor<T> NeuralNetwork<T>::forward(const Tensor<T>& input) {
    return model->forward(input);
}

template <Numeric T>
void NeuralNetwork<T>::backward(const Tensor<T>& input, const Tensor<T>& targets, const Tensor<T>& predictions) {
    Tensor<T> loss_grad = loss_function->gradient(predictions, targets);
    model->backward(loss_grad);
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

            T loss = loss_function->compute(predictions, batch_y);
            total_loss += loss;
            
            backward(batch_X, batch_y, predictions);
        }
        
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            std::cout << "Época " << (epoch + 1) << ", Loss: " << (total_loss/n_batches) << std::endl;
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