
#ifndef PRINT_H
#define PRINT_H
#include <vector>

class Print {
public:
    
    template <typename T>
    void print(const T* str);
    template <typename T>
    void print(T* num);
    
    template <typename T, int N>
    void print(const T(&arr)[N]);

    template <typename T, int Rows, int Cols>
    void print(const T(&arr)[Rows][Cols]);
    //void print(const vector<vector<float>>& vec);
};

#endif // PRINT_H