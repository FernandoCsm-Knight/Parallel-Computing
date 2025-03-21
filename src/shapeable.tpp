#ifndef SHAPABLE_TPP
#define SHAPABLE_TPP

#include "../inc/shapeable.hpp"

template <Integral... Dims>
Shapeable::Shapeable(Dims... dims) {
    sh = Shape(dims...);
}

Shapeable::Shapeable(Shape shape) {
    sh = shape;
}

Shapeable::Shapeable(Shape&& shape) {
    sh = shape;
}

int Shapeable::ndim() const {
    return sh.ndim();
}

size_t Shapeable::length() const {
    return sh.length();
}

bool Shapeable::is_scalar() const {
    return sh.ndim() == 0;
}

Shape Shapeable::shape() const {
    return sh;
}

int Shapeable::shape(int i) const {
    return sh[i];
}

#endif
