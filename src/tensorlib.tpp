#ifndef TENSORLIB_TPP
#define TENSORLIB_TPP

#include "../inc/tensorlib.hpp"

namespace tensor {
    template<Numeric T, Integral... Dims>
    Tensor<T> zeros(Dims... dims) {
        return Tensor<T>(dims...);
    }

    template<Numeric T, Integral... Dims>
    Tensor<T> ones(Dims... dims) {
        Tensor<T> result(dims...);
        #pragma omp parallel for
        for(size_t i = 0; i < result.length(); ++i) {
            result[i] = static_cast<T>(1);
        }
        return result;
    }

    template<Numeric T, Integral... Dims>
    Tensor<T> rand(Dims... dims) {
        Tensor<T> result(dims...);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if constexpr(std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dist(0, 100);
            #pragma omp parallel
            {
                std::mt19937 local_gen(rd() + omp_get_thread_num());
                #pragma omp for
                for(size_t i = 0; i < result.length(); ++i) {
                    result[i] = dist(local_gen);
                }
            }
        } else {
            std::uniform_real_distribution<T> dist(0.0, 1.0);
            #pragma omp parallel
            {
                std::mt19937 local_gen(rd() + omp_get_thread_num());
                #pragma omp for
                for(size_t i = 0; i < result.length(); ++i) {
                    result[i] = dist(local_gen);
                }
            }
        }
        
        return result;
    }

    template<Numeric T, Integral... Dims>
    Tensor<T> randn(Dims... dims, T mean, T stddev) {
        Tensor<T> result(dims...);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(mean, stddev);
        
        #pragma omp parallel
        {
            std::mt19937 local_gen(rd() + omp_get_thread_num());
            #pragma omp for
            for(size_t i = 0; i < result.length(); ++i) {
                result[i] = dist(local_gen);
            }
        }
        
        return result;
    }

    template<Numeric T>
    Tensor<T> arange(T start, T end, T step) {
        if (step == 0) {
            throw std::invalid_argument("Step cannot be zero");
        }
        
        int size = static_cast<int>(std::ceil((end - start) / step));
        size = size < 0 ? 0 : size;
        
        Tensor<T> result(size);
        #pragma omp parallel for
        for(int i = 0; i < size; ++i) {
            result[i] = start + i * step;
        }
        
        return result;
    }

    template<Numeric T>
    Tensor<T> eye(int n) {
        Tensor<T> result(n, n);
        #pragma omp parallel for
        for(int i = 0; i < n; ++i) {
            result(i, i) = static_cast<T>(1);
        }
        return result;
    }

    template<Numeric T>
    Tensor<T> linspace(T start, T end, int num) {
        if (num < 2) {
            throw std::invalid_argument("Number of samples must be at least 2");
        }
        
        Tensor<T> result(num);
        T step = (end - start) / (num - 1);
        
        #pragma omp parallel for
        for(int i = 0; i < num; ++i) {
            result[i] = start + step * i;
        }
        
        result[num - 1] = end;
        
        return result;
    }

    template<Numeric T, Integral... Dims>
    Tensor<T> full(T value, Dims... dims) {
        Tensor<T> result(dims...);
        #pragma omp parallel for
        for(size_t i = 0; i < result.length(); ++i) {
            result[i] = value;
        }
        return result;
    }
}

#endif