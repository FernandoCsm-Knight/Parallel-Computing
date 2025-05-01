#include "inc/neural_network.hpp"
#include "inc/tensor.hpp"
#include "inc/tensorlib.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

// Generate n non-linearly separable data points with num_classes classes
// Returns a pair of tensors (x, y) where:
// - x is the feature tensor with shape (n, 2) containing 2D points
// - y is the one-hot encoded labels with shape (n, num_classes)
template <Numeric T>
std::pair<Tensor<T>, Tensor<T>> generate_nonlinear_data(int n, int num_classes) {
  // Create tensors for data and labels
  Tensor<T> x(n, 2);           // n samples with 2 features (x, y coordinates)
  Tensor<T> y(n, num_classes); // One-hot encoded labels

  // Initialize random number generators
  std::random_device rd;

  // Parameters for creating non-linear separation between classes
  std::vector<T> class_radii(num_classes);
  T radius_step = 1.0 / num_classes;

  // Assign base radius for each class (concentric circles pattern)

  for (int c = 0; c < num_classes; ++c) {
    class_radii[c] = (c + 1) * radius_step;
  }

// Generate data points
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    std::mt19937 local_gen(rd() + i);

    std::uniform_real_distribution<T> radius_dist(0.0, 1.0);
    std::uniform_real_distribution<T> angle_dist(0.0, 2 * M_PI);
    std::uniform_real_distribution<T> noise_dist(-0.1, 0.1);

    int class_id = i % num_classes;
    T base_radius = class_radii[class_id];

    T radius = base_radius + radius_dist(local_gen) * radius_step * 0.5;
    T angle = angle_dist(local_gen);

    x(i, 0) = radius * std::cos(angle) + noise_dist(local_gen);
    x(i, 1) = radius * std::sin(angle) + noise_dist(local_gen);

    for (int c = 0; c < num_classes; ++c) {
        y(i, c) = (c == class_id) ? 1 : 0;
    }
}
  return std::make_pair(x, y);
}

int main() {
  // Configurações
  const int NUM_SAMPLES = 1000;
  const int NUM_CLASSES = 2;
  const int NUM_FEATURES = 2;
  const float LEARNING_RATE = 0.01f;
  const int EPOCHS = 1000;
  const int BATCH_SIZE = 32;

  omp_set_num_threads(8);

  // Gerar dados
  auto [features, labels] =
      generate_nonlinear_data<float>(NUM_SAMPLES, NUM_CLASSES);

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

  // #pragma omp parallel for collapse(2)
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

  std::cout << "Acurácia de treinamento: " << (accuracy_train * 100) << "%"
            << std::endl;
  std::cout << "Acurácia de teste: " << (accuracy_test * 100) << "%"
            << std::endl;

  tensor::cleanup();

  return 0;
}
