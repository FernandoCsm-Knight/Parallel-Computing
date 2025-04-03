#include "inc/tensor.hpp"
#include "inc/tensorlib.hpp"
#include "inc/linear_layer.hpp"

#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include <cmath>

// Função softmax para classificação multi-classe
template <Numeric T>
Tensor<T> softmax(const Tensor<T>& input) {
    Tensor<T> result(input.shape());
    
    // Aplicar softmax por batch (assumindo a dimensão 0 como batch)
    for (int i = 0; i < input.shape(0); ++i) {
        // Encontrar o valor máximo para estabilidade numérica
        T max_val = input(i, 0).value();
        for (int j = 1; j < input.shape(1); ++j) {
            max_val = std::max(max_val, input(i, j).value());
        }
        
        // Calcular exp(x_i - max) para cada elemento
        T sum_exp = 0;
        for (int j = 0; j < input.shape(1); ++j) {
            T exp_val = std::exp(input(i, j).value() - max_val);
            result(i, j) = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalizar dividindo pela soma
        for (int j = 0; j < input.shape(1); ++j) {
            result(i, j) = result(i, j).value() / sum_exp;
        }
    }
    
    return result;
}

// Cross-entropy loss para classificação
template <Numeric T>
T cross_entropy_loss(const Tensor<T>& predictions, const Tensor<T>& targets) {
    T loss = 0;
    int batch_size = predictions.shape(0);
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.shape(1); ++j) {
            if (targets(i, j).value() > 0) {
                // Evitar log(0) que é -inf
                T pred = std::max(predictions(i, j).value(), static_cast<T>(1e-7));
                loss -= targets(i, j).value() * std::log(pred);
            }
        }
    }
    
    return loss / batch_size;
}

// Derivada de softmax * cross entropy (combinados para eficiência)
template <Numeric T>
Tensor<T> softmax_cross_entropy_grad(const Tensor<T>& predictions, const Tensor<T>& targets) {
    Tensor<T> grad(predictions.shape());
    int batch_size = predictions.shape(0);
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < predictions.shape(1); ++j) {
            // A derivada simplificada para cross-entropy + softmax é (pred - target)
            grad(i, j) = (predictions(i, j).value() - targets(i, j).value()) / batch_size;
        }
    }
    
    return grad;
}

// Classe para rede neural de múltiplas camadas
template <Numeric T>
class NeuralNetwork {
private:
    LinearLayer<T> layer1;
    LinearLayer<T> layer2;
    LinearLayer<T> layer3;
    
public:
    NeuralNetwork(int input_features, int hidden_size1, int hidden_size2, int num_classes, T learning_rate)
        : layer1(input_features, hidden_size1, learning_rate),
          layer2(hidden_size1, hidden_size2, learning_rate),
          layer3(hidden_size2, num_classes, learning_rate) {}
    
    Tensor<T> forward(const Tensor<T>& input) {
        Tensor<T> hidden1 = layer1.forward(input);
        Tensor<T> hidden2 = layer2.forward(hidden1);
        Tensor<T> logits = layer3.forward(hidden2);
        return softmax(logits);
    }
    
    void backward(const Tensor<T>& input, const Tensor<T>& targets, const Tensor<T>& predictions) {
        // Calcular gradiente da perda
        Tensor<T> loss_grad = softmax_cross_entropy_grad(predictions, targets);
        
        // Backpropagation
        Tensor<T> grad3 = layer3.backward(loss_grad);
        Tensor<T> grad2 = layer2.backward(grad3);
        layer1.backward(grad2);
    }
    
    // Treinar a rede por um número específico de épocas
    void train(const Tensor<T>& X, const Tensor<T>& y, int epochs, int batch_size = 32) {
        int n_samples = X.shape(0);
        int n_batches = (n_samples + batch_size - 1) / batch_size; // Ceiling division
        
        std::vector<int> indices(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            indices[i] = i;
        }
        
        std::random_device rd;
        std::mt19937 g(rd());
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Embaralhar os índices para treinamento estocástico
            std::shuffle(indices.begin(), indices.end(), g);
            
            T total_loss = 0;
            
            for (int batch = 0; batch < n_batches; ++batch) {
                int start_idx = batch * batch_size;
                int end_idx = std::min(start_idx + batch_size, n_samples);
                int current_batch_size = end_idx - start_idx;
                
                // Criar batch de dados
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
                
                // Forward pass
                Tensor<T> predictions = forward(batch_X);
                
                // Calcular perda
                T loss = cross_entropy_loss(predictions, batch_y);
                total_loss += loss;
                
                // Backward pass
                backward(batch_X, batch_y, predictions);
            }
            
            // Imprimir perda média por época
            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                std::cout << "Época " << (epoch + 1) << ", Loss: " << (total_loss / n_batches) << std::endl;
            }
        }
    }
    
    // Avaliar a precisão do modelo
    T evaluate(const Tensor<T>& X, const Tensor<T>& y) {
        Tensor<T> predictions = forward(X);
        int correct = 0;
        int total = X.shape(0);
        
        for (int i = 0; i < total; ++i) {
            // Encontrar a classe predita (índice do valor máximo)
            int pred_class = 0;
            T max_prob = predictions(i, 0).value();
            
            for (int j = 1; j < predictions.shape(1); ++j) {
                if (predictions(i, j).value() > max_prob) {
                    max_prob = predictions(i, j).value();
                    pred_class = j;
                }
            }
            
            // Encontrar a classe verdadeira
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
};

// Generate n non-linearly separable data points with num_classes classes
// Returns a pair of tensors (x, y) where:
// - x is the feature tensor with shape (n, 2) containing 2D points
// - y is the one-hot encoded labels with shape (n, num_classes)
template <Numeric T>
std::pair<Tensor<T>, Tensor<T>> generate_nonlinear_data(int n, int num_classes) {
    // Create tensors for data and labels
    Tensor<T> x(n, 2);          // n samples with 2 features (x, y coordinates)
    Tensor<T> y(n, num_classes); // One-hot encoded labels

    // Initialize random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> radius_dist(0.0, 1.0);
    std::uniform_real_distribution<T> angle_dist(0.0, 2 * M_PI);
    std::uniform_real_distribution<T> noise_dist(-0.1, 0.1);
    
    // Parameters for creating non-linear separation between classes
    std::vector<T> class_radii(num_classes);
    T radius_step = 1.0 / num_classes;
    
    // Assign base radius for each class (concentric circles pattern)
    for (int c = 0; c < num_classes; ++c) {
        class_radii[c] = (c + 1) * radius_step;
    }
    
    // Generate data points
    for (int i = 0; i < n; ++i) {
        // Determine class for this point
        int class_id = i % num_classes;
        
        // Generate a point using polar coordinates
        T base_radius = class_radii[class_id];
        // Add some variation to radius
        T radius = base_radius + radius_dist(gen) * radius_step * 0.5;
        T angle = angle_dist(gen);
        
        // Convert to cartesian coordinates with some noise
        x(i, 0) = radius * std::cos(angle) + noise_dist(gen);
        x(i, 1) = radius * std::sin(angle) + noise_dist(gen);
        
        // Set one-hot encoded label
        for (int c = 0; c < num_classes; ++c) {
            y(i, c) = (c == class_id) ? 1 : 0;
        }
    }
    
    return std::make_pair(x, y);
}

int main() {
    // Configurações
    const int NUM_SAMPLES = 1000;
    const int NUM_CLASSES = 3;
    const int NUM_FEATURES = 2;
    const float LEARNING_RATE = 0.01f;
    const int EPOCHS = 1000;
    const int BATCH_SIZE = 32;
    
    // Gerar dados
    auto [features, labels] = generate_nonlinear_data<float>(NUM_SAMPLES, NUM_CLASSES);
    
    // Dividir em conjuntos de treinamento (80%) e teste (20%)
    int train_size = static_cast<int>(NUM_SAMPLES * 0.8);
    int test_size = NUM_SAMPLES - train_size;
    
    Tensor<float> X_train(train_size, NUM_FEATURES);
    Tensor<float> y_train(train_size, NUM_CLASSES);
    Tensor<float> X_test(test_size, NUM_FEATURES);
    Tensor<float> y_test(test_size, NUM_CLASSES);
    
    // Copiar dados para conjuntos de treinamento e teste
    for (int i = 0; i < train_size; ++i) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            X_train(i, j) = features(i, j).value();
        }
        for (int j = 0; j < NUM_CLASSES; ++j) {
            y_train(i, j) = labels(i, j).value();
        }
    }
    
    for (int i = 0; i < test_size; ++i) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            X_test(i, j) = features(train_size + i, j).value();
        }
        for (int j = 0; j < NUM_CLASSES; ++j) {
            y_test(i, j) = labels(train_size + i, j).value();
        }
    }
    
    // Criar e treinar o modelo
    NeuralNetwork<float> model(NUM_FEATURES, 16, 8, NUM_CLASSES, LEARNING_RATE);
    
    std::cout << "Iniciando treinamento..." << std::endl;
    model.train(X_train, y_train, EPOCHS, BATCH_SIZE);
    
    // Avaliar o modelo
    float accuracy_train = model.evaluate(X_train, y_train);
    float accuracy_test = model.evaluate(X_test, y_test);
    
    std::cout << "Acurácia de treinamento: " << (accuracy_train * 100) << "%" << std::endl;
    std::cout << "Acurácia de teste: " << (accuracy_test * 100) << "%" << std::endl;
    
    return 0;
}