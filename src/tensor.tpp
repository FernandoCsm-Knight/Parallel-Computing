#ifndef TENSOR_TPP
#define TENSOR_TPP

#include "../inc/tensor.hpp"

// Constructors

template <Numeric T>
template <Integral... Dims>
Tensor<T>::Tensor(Dims... dims): Shapeable(dims...) {
    stride = nullptr;
    data = nullptr;
    
    if (this->ndim() > 0) {
        stride = new int[this->ndim()];
        
        if (length() > 0) {
            data = new T[length()]();
            
            stride[this->ndim() - 1] = 1;
            for(int i = this->ndim() - 2; i >= 0; --i) {
                stride[i] = stride[i + 1] * this->shape(i + 1);
            }
        } else {
            data = new T[1]();
        }
    } else {
        data = new T[1]();
        stride = new int[1]();
    }
}

template <Numeric T>
Tensor<T>::Tensor(Shape shape): Shapeable(shape) {
    stride = nullptr;
    data = nullptr;
    
    if (this->ndim() > 0) {
        stride = new int[this->ndim()];
        
        if (length() > 0) {
            data = new T[length()]();
            
            stride[this->ndim() - 1] = 1;
            for(int i = this->ndim() - 2; i >= 0; --i) {
                stride[i] = stride[i + 1] * this->shape(i + 1);
            }
        } else {
            data = new T[1]();
        }
    } else {
        data = new T[1]();
        stride = new int[1]();
    }
}

template <Numeric T>
Tensor<T>::Tensor(const Tensor<T>& other) { 
    this->sh = other.shape();
    data = new T[length()]();
    stride = new int[this->ndim()];

    for(int i = 0; i < this->ndim(); ++i) {
        stride[i] = other.stride[i];
    }

    for(size_t i = 0; i < length(); ++i) {
        data[i] = static_cast<T>(other[i]);
    }
}

template <Numeric T>
Tensor<T>::Tensor(Tensor<T>&& other) noexcept {
    this->sh = std::move(other.sh);
    data = other.data;
    stride = other.stride;

    other.data = nullptr;
    other.stride = nullptr;
}

template <Numeric T>
template <Numeric U>
Tensor<T>::Tensor(const Tensor<U>& other): Shapeable(other.shape()) { 
    data = new T[length()]();
    stride = new int[this->ndim()];

    for(int i = 0; i < this->ndim(); ++i) {
        stride[i] = other.stride[i];
    }

    for(size_t i = 0; i < length(); ++i) {
        data[i] = static_cast<T>(other[i]);
    }
}

// Destructor

template <Numeric T>
Tensor<T>::~Tensor() {
    delete[] data;
    delete[] stride;
    data = nullptr;
    stride = nullptr;
}

// Assingment operators

template <Numeric T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other) {
    if(this != &other) {
        delete[] data;
        delete[] stride;

        this->sh = other.shape();
        data = new T[length()]();
        stride = new int[this->ndim()];

        for(int i = 0; i < this->ndim(); ++i){
            stride[i] = other.stride[i];
        }

        for(size_t i = 0; i < length(); ++i) {
            data[i] = static_cast<T>(other.data[i]);
        }
    }

    return *this;
}

template <Numeric T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& other) noexcept {
    if(this != &other) {
        delete[] data;
        delete[] stride;
        
        data = other.data;
        stride = other.stride;
        this->sh = std::move(other.sh);
        
        other.data = nullptr;
        other.stride = nullptr;
    }

    return *this;
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator=(const Tensor<U>& other) {
    if(this != &other) {
        delete[] data;
        delete[] stride;

        this->sh = other.shape();
        data = new T[length()]();
        stride = new int[this->ndim()];

        for(int i = 0; i < this->ndim(); ++i){
            stride[i] = other.stride[i];
        }

        for(size_t i = 0; i < length(); ++i) {
            data[i] = static_cast<T>(other.data[i]);
        }
    }

    return *this;
}

// Private methods

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::create_with_tensor(
    const Tensor<U>& other, 
    std::function<void(std::common_type_t<T, U>&, const T&, const U&)> func
) const {
    if(this->shape() != other.shape()) {
        throw std::invalid_argument("Cannot apply function to tensors of different shapes");
    }

    using R = std::common_type_t<T, U>;

    Tensor<R> result(this->shape());
    for(size_t i = 0; i < length(); ++i) {
        func(result.data[i], data[i], other.data[i]);
    }

    return result;
}


template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::create_with_scalar(
    const U& scalar, 
    std::function<void(std::common_type_t<T, U>&, const T&, const U&)> func
) const {
    using R = std::common_type_t<T, U>;

    Tensor<R> result(this->shape());
    for(size_t i = 0; i < length(); ++i) {
        func(result.data[i], data[i], scalar);
    }

    return result;
}

// Operators

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator+(const Tensor<U>& other) {
    if(other.is_scalar()) {
        return this->operator+(other.value());
    }

    return create_with_tensor(other, std::function<void(std::common_type_t<T, U>&, const T&, const U&)>(
        [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
            result = a + b; 
        }
    ));
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator+(const U& scalar) {
    return create_with_scalar(scalar, std::function<void(std::common_type_t<T, U>&, const T&, const U&)>(
        [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
            result = a + b; 
        }
    ));
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_reference_t<T, U>> Tensor<T>::operator-(const Tensor<U>& other) {
    if(other.is_scalar()) {
        return this->operator-(other.value());
    }

    return create_with_tensor(other, std::function<void(std::common_type_t<T, U>&, const T&, const U&)>(
        [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
            result = a - b; 
        }
    ));
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_reference_t<T, U>> Tensor<T>::operator-(const U& scalar) {
    return create_with_scalar(scalar, std::function<void(std::common_type_t<T, U>&, const T&, const U&)>(
        [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
            result = a - b; 
        }
    ));
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_reference_t<T, U>> Tensor<T>::operator*(const Tensor<U>& other) {
    if(other.is_scalar()) {
        return this->operator*(other.value());
    }

    return create_with_tensor(other, std::function<void(std::common_type_t<T, U>&, const T&, const U&)>(
        [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
            result = a * b; 
        }
    ));
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator*(const U& scalar) {
    return create_with_scalar(scalar, std::function<void(std::common_type_t<T, U>&, const T&, const U&)>(
        [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
            result = a * b; 
        }
    ));
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_reference_t<T, U>> Tensor<T>::operator/(const Tensor<U>& other) {
    if(other.is_scalar()) {
        return this->operator/(other.value());
    }

    return create_with_tensor(other, std::function<void(std::common_type_t<T, U>&, const T&, const U&)>(
        [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
            result = a / b; 
        }
    ));
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator/(const U& scalar) {
    return create_with_scalar(scalar, std::function<void(std::common_type_t<T, U>&, const T&, const U&)>(
        [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
            result = a / b; 
        }
    ));
}

// Accessors

template <Numeric T>
template <Integral... Indices>
TensorView<T> Tensor<T>::operator()(Indices... indices) {
    int dims = sizeof...(indices);
    if (dims > this->ndim()) {
        throw std::invalid_argument("Too many indices for tensor dimensions");
    }
    
    int idx_array[] = { static_cast<int>(indices)... };
    
    size_t offset = 0;
    for(int i = 0; i < dims; ++i) {
        if(idx_array[i] < 0 || idx_array[i] >= this->shape(i)) {
            std::stringstream ss;
            ss << "Array index " << i << " out of bounds for dimension " << this->shape(i);
            throw std::out_of_range(ss.str());
        }
        
        offset += idx_array[i] * stride[i];
    }
    
    Shape subview_shape;
    if (dims == this->ndim()) {
        return TensorView<T>(data + offset, subview_shape, new int[1]());
    } else {
        for(int i = dims; i < this->ndim(); ++i) {
            subview_shape.add_dimension(this->shape(i));
        }
        
        int* subview_stride = new int[this->ndim() - dims];
        for(int i = dims; i < this->ndim(); ++i) {
            subview_stride[i - dims] = stride[i];
        }
        
        return TensorView<T>(data + offset, subview_shape, subview_stride);
    }
}

template <Numeric T>
template <Integral... Indices>
TensorView<const T> Tensor<T>::operator()(Indices... indices) const {
    int dims = sizeof...(indices);
    if (dims > this->ndim()) {
        throw std::invalid_argument("Too many indices for tensor dimensions");
    }
    
    int idx_array[] = { static_cast<int>(indices)... };
    
    size_t offset = 0;
    for(int i = 0; i < dims; ++i) {
        if(idx_array[i] < 0 || idx_array[i] >= this->shape(i)) {
            std::stringstream ss;
            ss << "Array index " << i << " out of bounds for dimension " << this->shape(i);
            throw std::out_of_range(ss.str());
        }

        offset += idx_array[i] * stride[i];
    }
    
    Shape subview_shape;
    if (dims == this->ndim()) {
        return TensorView<const T>(data + offset, subview_shape, new int[1]());
    } else {
        for(int i = dims; i < this->ndim(); ++i) {
            subview_shape.add_dimension(this->shape(i));
        }
        
        int* subview_stride = new int[this->ndim() - dims];
        for(int i = dims; i < this->ndim(); ++i) {
            subview_stride[i - dims] = stride[i];
        }
        
        return TensorView<const T>(data + offset, subview_shape, subview_stride);
    }
}

template <Numeric T>
T& Tensor<T>::operator[](size_t idx) {
    if(this->ndim() == 0) {
        if(idx != 0) throw std::out_of_range("Index out of bounds for scalar value");
        return *data;
    }

    if(idx >= length()) {
        throw std::out_of_range("Index out of bounds for tensor");
    }

    return data[idx];
}

template <Numeric T>
const T& Tensor<T>::operator[](size_t idx) const {
    if(this->ndim() == 0) {
        if(idx != 0) throw std::out_of_range("Index out of bounds for scalar value");
        return *data;
    }

    if(idx >= length()) {
        throw std::out_of_range("Index out of bounds for tensor");
    }

    return data[idx];
}

template <Numeric T>
T& Tensor<T>::value() {
    if(this->ndim() != 0)
        throw std::invalid_argument("value() can only be called on scalar (0D) views");
    
    return *data;
}

template <Numeric T>
const T& Tensor<T>::value() const {
    if(this->ndim() != 0)
        throw std::invalid_argument("value() can only be called on scalar (0D) views");
    
    return *data;
}

// Helper methods

template <Numeric T>
T Tensor<T>::min() const {
    T min_element = data[0];
    for(size_t i = 1; i < length(); ++i) {
        min_element = std::min(min_element, data[i]);
    }

    return min_element;
}

template <Numeric T>
T Tensor<T>::max() const {
    T max_element = data[0];
    for(size_t i = 1; i < length(); ++i) {
        max_element = std::max(max_element, data[i]);
    }

    return max_element;
}

template <Numeric T>
T Tensor<T>::sum() const {
    T sum = 0;
    for(size_t i = 0; i < length(); ++i) {
        sum += data[i];
    }

    return sum;
}

template <Numeric T>
T Tensor<T>::mean() const {
    return sum() / length();
}

template <Numeric T>
T Tensor<T>::var() const {
    T mean_value = mean();
    T sum = 0;
    for(size_t i = 0; i < length(); ++i) {
        sum += (data[i] - mean_value) * (data[i] - mean_value);
    }

    return sum / length();
}

template <Numeric T>
T Tensor<T>::std() const {
    return std::sqrt(var());
}

template <Numeric T>
Tensor<T> Tensor<T>::abs() const {
    Tensor<T> result(this->shape());
    for(size_t i = 0; i < length(); ++i) {
        result.data[i] = std::abs(data[i]);
    }

    return result;
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::dot(const Tensor<U>& other) const {
    if (this->ndim() > 2 || other.ndim() > 2) {
        throw std::invalid_argument("Tensors must have at most 2 dimensions");
    }
    
    int m = (this->ndim() == 1) ? 1 : this->shape(0);
    int n = (this->ndim() == 1) ? this->shape(0) : this->shape(1);
    int p = (other.ndim() == 1) ? 1 : other.shape(1);
    
    if ((this->ndim() == 2 && other.ndim() == 2 && this->shape(1) != other.shape(0)) ||
        (this->ndim() == 1 && other.ndim() == 2 && this->shape(0) != other.shape(0)) ||
        (this->ndim() == 2 && other.ndim() == 1 && this->shape(1) != other.shape(0))) {
        throw std::invalid_argument("Cannot multiply tensors with incompatible shapes");
    }
    
    using R = std::common_type_t<T, U>;
    
    Tensor<R> result((m == 1) ? Shape(p) : (p == 1) ? Shape(m) : Shape(m, p));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            R sum = 0;
            
            for (int k = 0; k < n; ++k) {
                int idx1 = (this->ndim() == 1) ? k : i * stride[0] + k * stride[1];
                int idx2 = (other.ndim() == 1) ? k : k * other.stride[0] + j * other.stride[1];
                
                sum += static_cast<R>(data[idx1]) * static_cast<R>(other.data[idx2]);
            }
            
            if (m == 1) {
                result.data[j] = sum;
            } else if (p == 1) {
                result.data[i] = sum;
            } else {
                result.data[i * result.stride[0] + j * result.stride[1]] = sum;
            }
        }
    }
    
    return result;
}

// Formatted output

template <Numeric T>
Tensor<T> Tensor<T>::transpose() const {
    if(this->ndim() != 2) {
        throw std::invalid_argument("Tensor must have 2 dimensions");
    }

    Tensor<T> result(this->shape(1), this->shape(0));
    for(int i = 0; i < this->shape(0); ++i) {
        for(int j = 0; j < this->shape(1); ++j) {
            result(j, i) = (*this)(i, j).value();
        }
    }

    return result;
}

template <Numeric T>
Tensor<T> Tensor<T>::reshape(Shape new_shape) const {
    if(new_shape.length() != length()) {
        throw std::invalid_argument("New shape must have the same number of elements as the original shape");
    }

    Tensor<T> result(new_shape);
    for(size_t i = 0; i < length(); ++i) {
        result.data[i] = data[i];
    }

    return result;
}

template <Numeric T>
Tensor<T> Tensor<T>::flatten() const {
    return reshape(Shape(length()));
}

template <Numeric T>
Tensor<T> Tensor<T>::slice(int start, int end, int step) const {
    if(start < 0 || end > this->shape(0)) {
        throw std::invalid_argument("Slice indices out of bounds");
    }

    int new_size = (end - start) / step;
    Tensor<T> result(new_size);
    for(int i = 0; i < new_size; ++i) {
        result.data[i] = data[start + i * step];
    }

    return result;
}

// Iterators

template <Numeric T>
Iterator<T> Tensor<T>::begin() {
    return Iterator<T>(data);
}

template <Numeric T>
Iterator<T> Tensor<T>::end() {
    return Iterator<T>(data + length());
}

template <Numeric T>
Iterator<const T> Tensor<T>::begin() const {
    return Iterator<const T>(data);
}

template <Numeric T>
Iterator<const T> Tensor<T>::end() const {
    return Iterator<const T>(data + length());
}

// Modifiers

template <Numeric T>
void Tensor<T>::clear() {
    if(data != nullptr) {
        for(size_t i = 0; i < length(); ++i) 
            data[i] = T();
    }
}

#endif