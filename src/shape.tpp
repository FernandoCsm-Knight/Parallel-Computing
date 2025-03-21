#ifndef SHAPE_TPP
#define SHAPE_TPP

#include "../inc/shape.hpp"

// Constructors

template <Integral... Dims>
Shape::Shape(Dims... dims) {
    len = 1;
    dimensions = sizeof...(dims);

    if(dimensions == 0) {
        buff = nullptr;
    } else {
        buff = new int[dimensions]{ static_cast<int>(dims)... };
    
        for(int i = 0; i < dimensions; ++i) {
            len *= buff[i];
        }
    }
}

Shape::Shape(const Shape& other) {
    len = other.len;
    dimensions = other.dimensions;

    buff = new int[dimensions];
    for(int i = 0; i < dimensions; ++i) {
        buff[i] = other.buff[i];
    }
}

Shape::Shape(Shape&& other) noexcept {
    len = other.len;
    buff = other.buff;
    dimensions = other.dimensions;
    
    other.len = 0;
    other.dimensions = 0;
    other.buff = nullptr;
}

// Destructor

Shape::~Shape() {
    delete[] buff;
}

// Assignment operators

Shape& Shape::operator=(const Shape& other) {
    if(this != &other) {
        delete[] buff;
        len = other.len;
        dimensions = other.dimensions;

        buff = new int[dimensions];
        for(int i = 0; i < dimensions; ++i) {
            buff[i] = other.buff[i];
        }
    }

    return *this;
}

Shape& Shape::operator=(Shape&& other) noexcept {
    if (this != &other) {
        delete[] buff;
        len = other.len;
        buff = other.buff;
        dimensions = other.dimensions;

        other.len = 0;
        other.dimensions = 0;
        other.buff = nullptr;
    }

    return *this;
}

// Modifiers

void Shape::add_dimension(int dim) {
    int* new_buff = new int[dimensions + 1];
    for (int i = 0; i < dimensions; ++i) {
        new_buff[i] = buff[i];
    }

    new_buff[dimensions] = dim;
    len *= dim;

    delete[] buff;
    buff = new_buff;
    
    dimensions++;
}

// Accessors

int Shape::operator[](int i) const {
    return buff[i];
}

int& Shape::operator[](int i) {
    return buff[i];
}

int Shape::ndim() const {
    return dimensions;
}

size_t Shape::length() const {
    return (dimensions == 0) ? 1 : len;
}

bool Shape::is_scalar() const {
    return dimensions == 0;
}

// Boolean operators

bool Shape::operator==(const Shape& other) const {
    bool equal = dimensions == other.dimensions && len == other.len;

    for(int i = 0; equal && i < dimensions; ++i) 
        equal = buff[i] == other.buff[i];

    return equal;
}

bool Shape::operator!=(const Shape& other) const {
    return !(*this == other);
}

// Iterators

int* Shape::begin() {
    return buff;
}

int* Shape::end() {
    return buff + dimensions;
}

const int* Shape::begin() const {
    return buff;
}

const int* Shape::end() const {
    return buff + dimensions;
}

#endif
