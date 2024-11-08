#include "tensor.h"
#include <omp.h>

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape, T initial) : shape(shape)
{
    size_t total_size = 1;
    for (size_t dim : shape)
    {
        total_size *= dim;
    }
    data.resize(total_size, initial);
}

template <typename T>
Tensor<T> Tensor<T>::t() const
{
    if (shape.size() != 2)
    {
        throw std::invalid_argument("Transpose is only defined for 2D matrices.");
    }

    std::vector<size_t> transposed_shape = {shape[1], shape[0]};

    Tensor<T> result(transposed_shape);

    for (size_t i = 0; i < shape[0]; ++i)
    {
        for (size_t j = 0; j < shape[1]; ++j)
        {
            result({j, i}) = (*this)({i, j});
        }
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::getRow(size_t rowIndex) const
{
    if (shape.size() != 2)
    {
        throw std::invalid_argument("getRow is only defined for 2D matrices.");
    }
    if (rowIndex >= shape[0])
    {
        throw std::out_of_range("Row index is out of range.");
    }

    Tensor<T> row({shape[1]});

    for (size_t col = 0; col < shape[1]; ++col)
    {
        row.data[col] = (*this)({rowIndex, col});
    }

    return row;
}

template <typename T>
Tensor<T> Tensor<T>::getCol(size_t colIndex) const
{
    if (shape.size() != 2)
    {
        throw std::invalid_argument("getCol is only defined for 2D matrices.");
    }
    if (colIndex >= shape[1])
    {
        throw std::out_of_range("Column index is out of range.");
    }

    Tensor<T> col({shape[0]});

    for (size_t row = 0; row < shape[0]; ++row)
    {
        col.data[row] = (*this)({row, colIndex});
    }

    return col;
}

template <typename T>
size_t Tensor<T>::getIndex(const std::vector<size_t> &indices) const
{
    size_t index = 0;
    size_t stride = 1;
    for (size_t i = shape.size() - 1; i > 0; i--)
    {
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return index;
}

template <typename T>
inline const T &Tensor<T>::operator()(const std::vector<size_t> &indices) const
{
    size_t index = 0;
    size_t stride = 1;
    for (size_t i = shape.size() - 1; i > 0; i--)
    {
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return data[index];
}

template <typename T>
inline T &Tensor<T>::operator()(const std::vector<size_t> &indices)
{
    size_t index = 0;
    size_t stride = 1;
    for (size_t i = shape.size() - 1; i > 0; i--)
    {
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return data[index];
}

template <typename T>
Tensor<T> Tensor<T>::dot(const Tensor<T> &other) const
{
    if (shape.size() != 2 || other.shape.size() != 2 || shape[1] != other.shape[0])
    {
        throw std::invalid_argument("Incompatible shape for matrix dot multiplication.");
    }

    Tensor<T> result({shape[0], other.shape[1]});

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; i++)
    {
        for (size_t j = 0; j < other.shape[1]; j++)
        {
            T sum = 0;
            for (size_t k = 0; k < shape[1]; k++)
            {
                size_t index1 = shape[1] * i + k;       
                size_t index2 = other.shape[1] * k + j; 

                sum += (*this).data[index1] * other.data[index2];
            }
            result.data[result.shape[1]*i+j] = sum; 
        }
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::elementWiseProduct(const Tensor<T> &other) const
{
    if (shape != other.shape)
    {
        throw std::invalid_argument("Tensors must have the same shape for element-wise product.");
    }

    Tensor<T> result(shape);

#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
    {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::outerProduct(const Tensor<T> &other) const
{
    if (shape.size() != 1 || other.shape.size() != 1)
    {
        throw std::invalid_argument("Outer product is only defined for vectors.");
    }

    Tensor<T> result({shape[0], other.shape[0]});

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i)
    {
        for (size_t j = 0; j < other.shape[0]; ++j)
        {
            result({i, j}) = (*this)({i}) * other({j});
        }
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) const
{
    return this->elementWiseProduct(other);
}

template <typename T>
Tensor<T> Tensor<T>::cross(const Tensor<T> &other) const
{
    if (shape.size() != 1 || other.shape.size() != 1 || shape[0] != 3 || other.shape[0] != 3)
    {
        throw std::invalid_argument("Cross product is only defined for 3D vectors.");
    }

    Tensor<T> result({3});

    result({0}) = (*this)({1}) * other({2}) - (*this)({2}) * other({1});
    result({1}) = (*this)({2}) * other({0}) - (*this)({0}) * other({2});
    result({2}) = (*this)({0}) * other({1}) - (*this)({1}) * other({0});

    return result;
}

template <typename T>
size_t Tensor<T>::size() const
{
    return data.size();
}

template <typename T>
const std::vector<size_t> &Tensor<T>::getShape() const
{
    return shape;
}

template <typename T>
void Tensor<T>::print() const
{

    if (shape.size() == 1)
    {
        if (shape[0] > 1000)
        {
            printf("shape is too large to print\n");
            return;
        }
        for (size_t i = 0; i < shape[0]; ++i)
        {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    else if (shape.size() == 2)
    {
        if (shape[0] || shape[1] > 1000)
        {
            printf("shape is too large to print\n");
            return;
        }
        for (size_t i = 0; i < shape[0]; ++i)
        {
            for (size_t j = 0; j < shape[1]; ++j)
            {
                std::cout << (*this)({i, j}) << " ";
            }
            std::cout << std::endl;
        }
    }
}

template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;
