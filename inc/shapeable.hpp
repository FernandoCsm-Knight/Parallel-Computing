#ifndef SHAPEABLE_HPP
#define SHAPEABLE_HPP

#include "shape.hpp"

class Shapeable {
    protected:
        Shape sh;

    public:
        // Constructors

        template <Integral... Dims>
        Shapeable(Dims... dims);

        Shapeable(Shape shape);

        // Destructor

        virtual ~Shapeable() = default;

        // Accessors

        inline virtual int ndim() const;
        inline virtual size_t length() const;
        inline virtual bool is_scalar() const;
        inline virtual Shape shape() const;
        inline virtual int shape(int i) const;
};

#include "../src/shapeable.tpp"

#endif
