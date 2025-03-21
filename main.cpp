#include <iostream>
#include <iomanip>
#include <cassert>
#include "inc/linear_layer.hpp"


template <typename T>
void print_tensor(const Tensor<T>& tensor, const std::string& name) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < tensor.length(); ++i) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << tensor[i] << " ";
        if ((i + 1) % tensor.shape(tensor.ndim() - 1) == 0) {
            std::cout << "\n";
        }
    }
    std::cout << std::endl;
}

void test_update_parameters() {
    std::cout << "Testing LinearLayer::update_parameters..." << std::endl;
    
    // Create a linear layer with 2 input features and 3 output features
    LinearLayer<float> layer(2, 3, true);
    
    // Set weights and bias to known values for testing
    Tensor<float> weights(2, 3);
    for (size_t i = 0; i < weights.length(); ++i) {
        weights[i] = 0.1f * (i + 1);
    }
    
    Tensor<float> bias(3);
    for (size_t i = 0; i < bias.length(); ++i) {
        bias[i] = 0.5f * (i + 1);
    }
    
    // Since we don't have direct setters, we'll update the parameters using gradients
    Tensor<float> initial_weights = layer.get_weights();
    Tensor<float> initial_bias = layer.get_bias();
    
    print_tensor(initial_weights, "Initial weights");
    print_tensor(initial_bias, "Initial bias");
    
    // Create gradients with known values
    Tensor<float> grad_weights(2, 3);
    for (size_t i = 0; i < grad_weights.length(); ++i) {
        grad_weights[i] = 0.01f * (i + 1);
    }
    
    Tensor<float> grad_bias(3);
    for (size_t i = 0; i < grad_bias.length(); ++i) {
        grad_bias[i] = 0.02f * (i + 1);
    }
    
    print_tensor(grad_weights, "Gradient weights");
    print_tensor(grad_bias, "Gradient bias");
    
    // Apply the update with a learning rate of 0.1
    float learning_rate = 0.1f;
    layer.update_parameters(learning_rate, grad_weights, grad_bias);
    
    // Get updated weights and bias
    Tensor<float> updated_weights = layer.get_weights();
    Tensor<float> updated_bias = layer.get_bias();
    
    print_tensor(updated_weights, "Updated weights");
    print_tensor(updated_bias, "Updated bias");
    
    // Verify correctness: w_new = w_old - lr * grad_w
    for (size_t i = 0; i < updated_weights.length(); ++i) {
        float expected = initial_weights[i] - learning_rate * grad_weights[i];
        assert(std::abs(updated_weights[i] - expected) < 1e-6f);
    }
    
    for (size_t i = 0; i < updated_bias.length(); ++i) {
        float expected = initial_bias[i] - learning_rate * grad_bias[i];
        assert(std::abs(updated_bias[i] - expected) < 1e-6f);
    }
    
    std::cout << "All tests passed for update_parameters!" << std::endl;
}

// Comprehensive test that applies multiple updates
void test_multiple_updates() {
    std::cout << "Testing multiple parameter updates..." << std::endl;
    
    // Create a linear layer with 2 input features and 2 output features
    LinearLayer<double> layer(2, 2, true);
    
    // Create manual gradients
    Tensor<double> grad_weights(2, 2);
    grad_weights[0] = 0.1;
    grad_weights[1] = 0.2;
    grad_weights[2] = 0.3;
    grad_weights[3] = 0.4;
    
    Tensor<double> grad_bias(2);
    grad_bias[0] = 0.05;
    grad_bias[1] = 0.15;
    
    // Get initial weights and bias
    Tensor<double> initial_weights = layer.get_weights();
    Tensor<double> initial_bias = layer.get_bias();
    
    print_tensor(initial_weights, "Initial weights");
    print_tensor(initial_bias, "Initial bias");
    
    // Apply multiple updates with different learning rates
    double learning_rates[] = {0.01, 0.05, 0.1};
    
    for (int i = 0; i < 3; ++i) {
        double lr = learning_rates[i];
        
        std::cout << "Update " << (i+1) << " with learning rate " << lr << std::endl;
        layer.update_parameters(lr, grad_weights, grad_bias);
        
        Tensor<double> updated_weights = layer.get_weights();
        Tensor<double> updated_bias = layer.get_bias();
        
        print_tensor(updated_weights, "Updated weights");
        print_tensor(updated_bias, "Updated bias");
    }
    
    std::cout << "Multiple update test completed!" << std::endl;
}

int main() {
    test_update_parameters();
    std::cout << "--------------------------------" << std::endl;
    test_multiple_updates();
    
    return 0;
}