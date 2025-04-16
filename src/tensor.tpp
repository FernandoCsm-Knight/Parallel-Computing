#ifndef TENSOR_TPP
#define TENSOR_TPP

#include "../inc/tensor.hpp"

// Constructors

template <Numeric T>
template <Integral... Dims>
Tensor<T>::Tensor(Dims... dims): Shapeable(dims...) {
    data = new T[this->length()]();
    stride = new int[this->ndim()]();

    if(this->ndim() != 0) {
        stride[this->ndim() - 1] = 1;
        for(int i = this->ndim() - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * this->shape(i + 1);
        }
    } 
}

template <Numeric T>
Tensor<T>::Tensor(Shape shape): Shapeable(shape) {
    data = new T[this->length()]();
    stride = new int[this->ndim()]();

    if(this->ndim() != 0) {
        stride[this->ndim() - 1] = 1;
        for(int i = this->ndim() - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * this->shape(i + 1);
        }
    } 
}

template <Numeric T>
Tensor<T>::Tensor(const Tensor<T>& other): Shapeable(other) { 
    data = new T[this->length()]();
    stride = new int[this->ndim()];

    for(int i = 0; i < this->ndim(); ++i) {
        stride[i] = other.stride[i];
    }

    #pragma omp parallel for
    for(size_t i = 0; i < this->length(); ++i) {
        data[i] = other.data[i];
    }
}

template <Numeric T>
Tensor<T>::Tensor(Tensor<T>&& other) noexcept: Shapeable(std::move(other)) {
    data = other.data;
    stride = other.stride;
    other.stride = nullptr;
    other.data = nullptr;
}

template <Numeric T>
template <Numeric U>
Tensor<T>::Tensor(const Tensor<U>& other): Shapeable(other) { 
    data = new T[this->length()]();
    stride = new int[this->ndim()];

    for(int i = 0; i < this->ndim(); ++i) {
        stride[i] = other.stride[i];
    }

    #pragma omp parallel for
    for(size_t i = 0; i < this->length(); ++i) {
        data[i] = static_cast<T>(other.data[i]);
    }
}

// Destructor

template <Numeric T>
Tensor<T>::~Tensor() {
    delete[] data;
    delete[] stride;
    stride = nullptr;
    data = nullptr;
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
        return TensorView<T>(this->data + offset, subview_shape, new int[1]());
    } else {
        for(int i = dims; i < this->ndim(); ++i) {
            subview_shape.add_dimension(this->shape(i));
        }
        
        int* subview_stride = new int[this->ndim() - dims];
        for(int i = dims; i < this->ndim(); ++i) {
            subview_stride[i - dims] = stride[i];
        }
        
        return TensorView<T>(this->data + offset, subview_shape, subview_stride);
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
        return TensorView<const T>(this->data + offset, subview_shape, new int[1]());
    } else {
        for(int i = dims; i < this->ndim(); ++i) {
            subview_shape.add_dimension(this->shape(i));
        }
        
        int* subview_stride = new int[this->ndim() - dims];
        for(int i = dims; i < this->ndim(); ++i) {
            subview_stride[i - dims] = stride[i];
        }
        
        return TensorView<const T>(this->data + offset, subview_shape, subview_stride);
    }
}

template <Numeric T>
T& Tensor<T>::operator[](size_t idx) {
    return data[idx];
}

template <Numeric T>
const T& Tensor<T>::operator[](size_t idx) const {
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
    T min_element = this->data[0];

    #pragma omp parallel for
    for(size_t i = 1; i < this->length(); ++i) {
        min_element = std::min(min_element, this->data[i]);
    }

    return min_element;
}

template <Numeric T>
T Tensor<T>::max() const {
    T max_element = this->data[0];
    
    #pragma omp parallel for
    for(size_t i = 1; i < this->length(); ++i) {
        max_element = std::max(max_element, this->data[i]);
    }

    return max_element;
}

template <Numeric T>
Tensor<T> Tensor<T>::sum(int axis, bool keep_dimension) const {
    Tensor<T> result;

    if (axis == -1) {
        if (keep_dimension) {
            Shape result_shape;
            for (int i = 0; i < this->ndim(); ++i) {
                result_shape.add_dimension(1);
            }
            
            result = Tensor<T>(result_shape);
            T total = 0;
            
            #pragma omp parallel for
            for (size_t i = 0; i < this->length(); ++i) {
                total += this->data[i];
            }
            
            result[0] = total;
        } else {
            result = Tensor<T>(Shape());
            T total = 0;
            
            #pragma omp parallel for
            for (size_t i = 0; i < this->length(); ++i) {
                total += this->data[i];
            }
            
            result.value() = total;
        }
    } else {        
        if (axis < 0 || axis >= this->ndim()) {
            throw std::invalid_argument("Invalid axis value for sum operation");
        }
        
        Shape result_shape;
        for (int i = 0; i < this->ndim(); ++i) {
            if (i == axis) {
                if (keep_dimension) result_shape.add_dimension(1);
            } else {
                result_shape.add_dimension(this->shape()[i]);
            }
        }
        
        result = Tensor<T>(result_shape);
        
        int axis_stride = 1;
        for (int i = this->ndim() - 1; i > axis; --i) {
            axis_stride *= this->shape()[i];
        }
    
        int axis_size = this->shape()[axis];
        int outer_block_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_block_size *= this->shape()[i];
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < result.length(); ++i) {
            result[i] = 0;
        }
        
        #pragma omp parallel for collapse(2)
        for (int outer = 0; outer < outer_block_size; ++outer) {
            for (int inner = 0; inner < axis_stride; ++inner) {
                size_t result_idx = (keep_dimension) ?
                    outer * axis_stride * (result_shape[axis]) + inner :
                    outer * axis_stride + inner;
                
                for (int a = 0; a < axis_size; ++a) {
                    size_t src_idx = outer * axis_size * axis_stride + a * axis_stride + inner;
                    result[result_idx] += this->data[src_idx];
                }
            }
        }
    }
    
    return result;
}

template <Numeric T>
T Tensor<T>::mean() const {
    return sum() / this->length();
}

template <Numeric T>
T Tensor<T>::var() const {
    T mean_value = mean();
    T sum = 0;

    #pragma omp parallel for
    for(size_t i = 0; i < this->length(); ++i) {
        sum += (this->data[i] - mean_value) * (this->data[i] - mean_value);
    }

    return sum / this->length();
}

template <Numeric T>
T Tensor<T>::std() const {
    return std::sqrt(var());
}

template <Numeric T>
Tensor<T> Tensor<T>::abs() const {
    Tensor<T> result(this->shape());

    #pragma omp parallel for
    for(size_t i = 0; i < this->length(); ++i) {
        result.data[i] = std::abs(this->data[i]);
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
                
                sum += static_cast<R>(this->data[idx1]) * static_cast<R>(other.data[idx2]);
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

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < this->shape(0); ++i) {
        for(int j = 0; j < this->shape(1); ++j) {
            result.data[j * result.stride[0] + i] = this->data[i * this->stride[0] + j];
        }
    }

    return result;
}

template <Numeric T>
Tensor<T> Tensor<T>::reshape(Shape new_shape) const {
    if(new_shape.length() != this->length()) {
        throw std::invalid_argument("New shape must have the same number of elements as the original shape");
    }

    Tensor<T> result(new_shape);

    #pragma omp parallel for
    for(size_t i = 0; i < this->length(); ++i) {
        result.data[i] = this->data[i];
    }

    return result;
}

template <Numeric T>
Tensor<T> Tensor<T>::flatten() const {
    return reshape(Shape(this->length()));
}

template <Numeric T>
Tensor<T> Tensor<T>::slice(int start, int end, int step) const {
    if(start < 0 || end > this->shape(0)) {
        throw std::invalid_argument("Slice indices out of bounds");
    }

    int new_size = (end - start) / step;
    Tensor<T> result(new_size);
    for(int i = 0; i < new_size; ++i) {
        result.data[i] = this->data[start + i * step];
    }

    return result;
}

// Iterators

template <Numeric T>
Iterator<T> Tensor<T>::begin() {
    return Iterator<T>(this->data);
}

template <Numeric T>
Iterator<T> Tensor<T>::end() {
    return Iterator<T>(this->data + this->length());
}

template <Numeric T>
Iterator<const T> Tensor<T>::begin() const {
    return Iterator<const T>(this->data);
}

template <Numeric T>
Iterator<const T> Tensor<T>::end() const {
    return Iterator<const T>(this->data + this->length());
}

// Modifiers

template <Numeric T>
void Tensor<T>::clear() {
    if(this->data != nullptr) {
        #pragma omp parallel for
        for(size_t i = 0; i < this->length(); ++i) 
            this->data[i] = T();
    }
}

#endif