#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <iostream>

#include "numeric.hpp"

class Shape {
    private:
        int* buff = nullptr;
        int dimensions;
        size_t len;

    public:
        // Constructors

        template <Integral... Dims>
        Shape(Dims... dims);

        Shape(const Shape& other);
        Shape(Shape&& other) noexcept;

        // Destructor

        ~Shape();

        // Assignment operators

        Shape& operator=(const Shape& other);
        Shape& operator=(Shape&& other) noexcept;

        // Modifiers

        void add_dimension(int dim);

        // Accessors
        
        int operator[](int i) const;
        int& operator[](int i);
        
        int ndim() const;
        size_t length() const;
        bool is_scalar() const;

        // Boolean operators

        bool operator==(const Shape& other) const;
        bool operator!=(const Shape& other) const;

        // Iterators

        int* begin();
        int* end();

        const int* begin() const;
        const int* end() const;

        // Formatted output

        friend std::ostream& operator<<(std::ostream& os, const Shape& sh) {
            os << "(";
            for(int i = 0; i < sh.ndim(); ++i) {
                os << sh[i];
                if(i < sh.ndim() - 1) {
                    os << ", ";
                }
            }
            os << ")";
            
            return os;
        }
};

#include "../src/shape.tpp"

#endif 