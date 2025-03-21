#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <type_traits>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <cmath>

#include "numeric.hpp"
#include "shapeable.hpp"
#include "views.hpp"

template <Numeric T> class Tensor: public Shapeable {
    template <Numeric> friend class Tensor;

    using Shapeable::shape;

    private:
        T* data = nullptr;
        int* stride = nullptr;

        // Private methods

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> create_with_tensor(
            const Tensor<U>& other, 
            std::function<void(std::common_type_t<T, U>&, const T&, const U&)> func
        ) const;

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> create_with_scalar(
            const U& scalar, 
            std::function<void(std::common_type_t<T, U>&, const T&, const U&)> func
        ) const;

    public:
        // Constructors
    
        template<Integral... Dims>
        Tensor(Dims... dims);

        Tensor(Shape shape);

        Tensor(const Tensor<T>& other);

        Tensor(Tensor<T>&& other) noexcept;
        
        template <Numeric U>
        Tensor(const Tensor<U>& other);

        // Destructor

        ~Tensor();

        // Assingment operators

        Tensor<T>& operator=(const Tensor<T>& other);
        Tensor<T>& operator=(Tensor<T>&& other) noexcept;

        template <Numeric U>
        Tensor<T>& operator=(const Tensor<U>& other);

        // Operators

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator+(const Tensor<U>& other);

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator+(const U& scalar);

        template <Numeric U>
        Tensor<std::common_reference_t<T, U>> operator-(const Tensor<U>& other);

        template <Numeric U>
        Tensor<std::common_reference_t<T, U>> operator-(const U& scalar);

        template <Numeric U>
        Tensor<std::common_reference_t<T, U>> operator*(const Tensor<U>& other);

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator*(const U& scalar);

        template <Numeric U>
        Tensor<std::common_reference_t<T, U>> operator/(const Tensor<U>& other);

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator/(const U& scalar);

        // Accessors

        template<Integral... Indices>
        TensorView<T> operator()(Indices... indices);

        template<Integral... Indices>
        TensorView<const T> operator()(Indices... indices) const;

        T& operator[](size_t idx);
        const T& operator[](size_t idx) const;

        T& value();
        const T& value() const;

        // Helper methods

        T min() const;
        T max() const;
        T sum() const;
        T mean() const;
        T var() const;
        T std() const;
        Tensor<T> abs() const;

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> dot(const Tensor<U>& other) const;

        // Formatted tensors

        Tensor<T> transpose() const;
        Tensor<T> reshape(Shape new_shape) const;
        Tensor<T> flatten() const;
        Tensor<T> slice(int start, int end, int step = 1) const;

        // Iterators

        Iterator<T> begin();
        Iterator<T> end();

        Iterator<const T> begin() const;
        Iterator<const T> end() const;

        // Modifiers

        void clear();

        // Formatted output

        friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
            if(tensor.is_scalar()) {
                os << "Tensor(" << tensor.value() << ")";
            } else {
                os << "Tensor(" << std::endl;
                os << "[";
                for(size_t i = 0; i < tensor.length(); ++i) {
                    for(int j = 0;  j < tensor.ndim() - 1; ++j) {
                        if(i % tensor.stride[j] == 0) {
                            os << "[";
                        }
                    }
    
                    os << tensor.data[i];
                
    
                    for(int j = 0;  j < tensor.ndim() - 1; ++j) {
                        if((i + 1) % tensor.stride[j] == 0) {
                            os << "]";
                        }
                    }
                    
                    if (i < tensor.length() - 1) {
                        os << ", ";
                    }
                }
                os << "]" << std::endl;
                os << tensor.shape() << std::endl << ")";
            }

            return os;
        }
};

#include "../src/tensor.tpp"

#endif 