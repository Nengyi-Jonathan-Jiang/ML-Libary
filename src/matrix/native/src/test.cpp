#include "matrix.h"
#include <iostream>

int main() {
    constexpr int SIZE = 700;

    auto* buf = new double[SIZE * SIZE];
    auto* outbuf = new double[SIZE * SIZE];

    std::cout << "calculating...\n";

    matrix m(buf, SIZE * SIZE, SIZE * SIZE);
    matrix out(outbuf, SIZE * SIZE, SIZE * SIZE);

    matrix::multiply(out, m, m, SIZE, SIZE, SIZE);

    std::cout << "done.\n";

    delete[] buf;
    delete[] outbuf;
}