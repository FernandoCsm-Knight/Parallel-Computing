#ifndef VIEWS_TPP
#define VIEWS_TPP

#include "../inc/views.hpp"

// Private methods

template <Numeric T>
void TensorView<T>::fill(const T& value) {
    size_t total_elements = length();
    
    if (this->ndim() == 0) {
        *data = value;
    } else {
        int indices[this->ndim()];
        for(int i = 0; i < this->ndim(); ++i) {
            indices[i] = 0;
        }
    
        for(size_t i = 0; i < total_elements; ++i) {
            size_t flat_idx = 0;
            for(int d = 0; d < this->ndim(); ++d) {
                flat_idx += indices[d] * stride[d];
            }
            
            data[flat_idx] = value;
            
            for(int d = this->ndim() - 1; d >= 0; --d) {
                indices[d]++;
                
                if(indices[d] < this->shape(d)) d = -1;
                else indices[d] = 0;
            }
        }
    }
}

// Constructors

template <Numeric T>
TensorView<T>::TensorView() {}

template <Numeric T>
TensorView<T>::TensorView(T* data_ptr, const Shape& shape, int* strides): Shapeable(shape) {
    data = data_ptr;
    stride = strides;
    owns_stride = true;
}

template <Numeric T>
TensorView<T>::TensorView(TensorView<T>&& other) noexcept {
    data = other.data;
    this->sh = std::move(other.sh);
    stride = other.stride;
    owns_stride = other.owns_stride;

    other.stride = nullptr;
    other.owns_stride = false;
}

// Destructor

template <Numeric T>
TensorView<T>::~TensorView() {
    if (owns_stride) delete[] stride;
}

// Assignment operators

template <Numeric T>
TensorView<T>& TensorView<T>::operator=(TensorView<T>&& other) noexcept {
    if (this != &other) {
        if (owns_stride) delete[] stride;
        
        data = other.data;
        this->sh = std::move(other.sh);
        stride = other.stride;
        owns_stride = other.owns_stride;
        
        other.stride = nullptr;
        other.owns_stride = false;
    }

    return *this;
}

template <Numeric T>
TensorView<T>& TensorView<T>::operator=(const T& value) {
    if (this->ndim() == 0) *data = value;
    else fill(value);
    
    return *this;
}

// Accessors

template <Numeric T>
template <Numeric... Indices>
TensorView<T> TensorView<T>::operator()(Indices... indices) {
    int dims = sizeof...(indices);
    if (dims > this->ndim()) {
        throw std::invalid_argument("Too many indices for view dimensions");
    }
    
    int idx_array[] = { static_cast<int>(indices)... };
    
    size_t offset = 0;
    for(size_t i = 0; i < dims; ++i) {
        if(idx_array[i] < 0 || idx_array[i] >= this->shape(i)) {
            std::stringstream ss;
            ss << "View index " << i << " out of bounds for dimension " << this->shape(i);
            throw std::out_of_range(ss.str());
        }

        offset += idx_array[i] * stride[i];
    }
    
    TensorView<T> view;
    Shape subview_shape;
    if (dims == this->ndim()) {
        view = TensorView<T>(data + offset, subview_shape, new int[1]());
    } else {
        for(int i = dims; i < this->ndim(); ++i) {
            subview_shape.add_dimension(this->shape(i));
        }
        
        int* subview_stride = new int[this->ndim() - dims];
        for(int i = dims; i < this->ndim(); ++i) {
            subview_stride[i - dims] = stride[i];
        }
        
        view = TensorView<T>(data + offset, subview_shape, subview_stride);
    }

    return view;
}

template <Numeric T>
template <typename... Indices>
TensorView<const T> TensorView<T>::operator()(Indices... indices) const {
    int dims = sizeof...(indices);
    if (dims > this->ndim()) {
        throw std::invalid_argument("Too many indices for view dimensions");
    }
    
    int idx_array[] = { static_cast<int>(indices)... };
    
    size_t offset = 0;
    for(size_t i = 0; i < dims; ++i) {
        if(idx_array[i] < 0 || idx_array[i] >= this->shape(i)) {
            std::stringstream ss;
            ss << "View index " << i << " out of bounds for dimension " << this->shape(i);
            throw std::out_of_range(ss.str());
        }

        offset += idx_array[i] * stride[i];
    }
    
    TensorView<const T> view;
    Shape subview_shape;
    if (dims == this->ndim()) {
        view = TensorView<const T>(data + offset, subview_shape, new int[1]());
    } else {
        for(int i = dims; i < this->ndim(); ++i) {
            subview_shape.add_dimension(this->shape(i));
        }
        
        int* subview_stride = new int[this->ndim() - dims];
        for(int i = dims; i < this->ndim(); ++i) {
            subview_stride[i - dims] = stride[i];
        }
        
        view = TensorView<const T>(data + offset, subview_shape, subview_stride);
    }

    return view;
}

template <Numeric T>
T& TensorView<T>::operator[](size_t idx) {
    if (this->ndim() == 0) {
        if (idx > 0) throw std::out_of_range("Index out of range for scalar view");
        return *data;
    }
    
    if (idx >= length()) {
        throw std::out_of_range("Index out of range");
    }

    int indices[this->ndim()];
    size_t remaining = idx;
    
    for (int i = 0; i < this->ndim() - 1; ++i) {
        size_t stride_length = 1;
        for (int j = i + 1; j < this->ndim(); ++j) {
            stride_length *= this->shape(j);
        }
        
        indices[i] = remaining / stride_length;
        remaining %= stride_length;
    }
    
    indices[this->ndim() - 1] = remaining;
    
    size_t offset = 0;
    for (int i = 0; i < this->ndim(); ++i) {
        offset += indices[i] * stride[i];
    }
    
    return data[offset];
}

template <Numeric T>
const T& TensorView<T>::operator[](size_t idx) const {
    if (this->ndim() == 0) {
        if (idx > 0) throw std::out_of_range("Index out of range for scalar view");
        return *data;
    }
    
    if (idx >= length()) {
        throw std::out_of_range("Index out of range");
    }
    
    int indices[this->ndim()];
    size_t remaining = idx;
    
    for (int i = 0; i < this->ndim() - 1; ++i) {
        size_t stride_length = 1;
        for (int j = i + 1; j < this->ndim(); ++j) {
            stride_length *= this->shape(j);
        }
        
        indices[i] = remaining / stride_length;
        remaining %= stride_length;
    }
    
    indices[this->ndim() - 1] = remaining;
    
    size_t offset = 0;
    for (int i = 0; i < this->ndim(); ++i) {
        offset += indices[i] * stride[i];
    }
    
    return data[offset];
}

template <Numeric T>
T& TensorView<T>::value() {
    if (this->ndim() != 0) 
        throw std::invalid_argument("value() can only be called on scalar (0D) views");
    
    return *data;
}

template <Numeric T>
const T& TensorView<T>::value() const {
    if (this->ndim() != 0) 
        throw std::invalid_argument("value() can only be called on scalar (0D) views");
    
    return *data;
}

#endif
