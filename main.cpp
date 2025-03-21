#include <iostream>

#include "inc/tensor.hpp"
#include "inc/tensorlib.hpp"

int main() {
   Tensor<int> t = tensor::rand<int>(2, 2);

   std::cout << "Tensor t:\n" << t << std::endl;

   Tensor<int> t2 = t.transpose();
   std::cout << "Transpose of t:\n" << t2 << std::endl;

   Tensor<int> t3 = t.reshape(Shape(4));
   std::cout << "Reshaped t:\n" << t3 << std::endl;

   Tensor<int> t4 = t.abs();
   std::cout << "Absolute values of t:\n" << t4 << std::endl;

   std::cout << "Mean of t: " << t.mean() << std::endl;
   std::cout << "Variance of t: " << t.var() << std::endl;
   std::cout << "Standard deviation of t: " << t.std() << std::endl;

   Tensor<float> t5 = tensor::rand<float>(2, 2);

   std::cout << "Tensor t5:\n" << t5 << std::endl;
   std::cout << "Sum t + t5\n" << t + t5 << std::endl;
   std::cout << "Difference t - t5\n" << t - t5 << std::endl;
   std::cout << "Product t * t5\n" << t * t5 << std::endl;
   std::cout << "Quotient t / t5\n" << t / t5 << std::endl;

   Tensor<int> t6 = tensor::zeros<int>(2, 2, 3);
   std::cout << "Tensor t6:\n" << t6 << std::endl;

   t6(0) = 20;
   std::cout << "Tensor t6(0) + 20:\n" << t6 << std::endl;

   return 0;
}