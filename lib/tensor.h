#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>

template <typename T>
class Tensor
{
private:
    std::vector<T> data;
    std::vector<size_t> shape;
    size_t getIndex(const std::vector<size_t>& indices) const;

public:
    Tensor(const std::vector<size_t>& shape, T initial = T());
    const T& operator()(const std::vector<size_t>& indices) const;
    T& operator()(const std::vector<size_t>& indices);
    Tensor<T> t() const;
    Tensor<T> dot(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T>& other) const;

    Tensor<T> getRow(size_t rowIndex) const;
    Tensor<T> getCol(size_t colIndex) const;

    Tensor<T> elementWiseProduct(const Tensor<T>& other) const;
    Tensor<T> outerProduct(const Tensor<T>& other) const;
    Tensor<T> cross(const Tensor<T>& other) const;

    size_t size() const;
    const std::vector<size_t>& getShape() const;
    void print() const;
};

#endif
