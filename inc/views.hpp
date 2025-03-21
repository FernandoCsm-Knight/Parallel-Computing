#ifndef VIEWS_HPP
#define VIEWS_HPP

#include <stdexcept>
#include <sstream>

#include "shapeable.hpp"
#include "numeric.hpp"
#include "tensor.hpp"

template <Numeric T>
class TensorView: Shapeable {
    template <Numeric> friend class TensorView;

    private:
        T* data = nullptr;
        int* stride = nullptr;
        bool owns_stride = false;

        // Private methods

        void fill(const T& value);

    public:
        // Constructors

        TensorView();

        TensorView(T* data_ptr, const Shape& shape, int* strides);
        
        TensorView(const TensorView<T>& other) = delete;
        TensorView<T>& operator=(const TensorView<T>& other) = delete;
        
        TensorView(TensorView<T>&& other) noexcept;
        
        // Destructor 

        ~TensorView();
        
        // Assignment operators
        
        TensorView<T>& operator=(TensorView<T>&& other) noexcept;
        TensorView<T>& operator=(const T& value);

        // Accessors

        template<Numeric... Indices>
        TensorView<T> operator()(Indices... indices);
        
        template<typename... Indices>
        TensorView<const T> operator()(Indices... indices) const;

        T& operator[](size_t idx);
        const T& operator[](size_t idx) const;
        
        T& value();
        const T& value() const;
};

#include "../src/views.tpp"

#endif
