#include "Print.h"
#include <iostream>
#include<vector>
using namespace std;
template <typename T>
void Print::print(const T* str) {
    std::cout << str << std::endl;
}
template <typename T>
void Print::print(T* num) {
    std::cout << num << std::endl;
}



template <typename T, int N>
void print(const T(&arr)[N]) {
    for (int i = 0; i < N; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T, int Rows, int Cols>
void print(const T(&arr)[Rows][Cols]) {
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            std::cout << arr[i][j] << " ";
        }
        std::cout << std::endl;
    }
}


void print(const vector<vector<float>>& vec) {
    for (const auto& row : vec) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
}