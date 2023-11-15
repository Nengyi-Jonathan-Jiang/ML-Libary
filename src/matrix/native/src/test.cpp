#include "matrix.h"
#include <iostream>
#include <random>

int main() {
    constexpr int SIZE = 700;

    auto* buf = new double[SIZE * SIZE];
    auto* outbuf = new double[SIZE * SIZE];

    for(int i = 0; i < SIZE * SIZE; i++) {
        buf[i] = rand() / 1000.;
    }

    std::cout << "calculating...\n";

    matrix m(buf, SIZE * SIZE, SIZE * SIZE);
    matrix out(outbuf, SIZE * SIZE, SIZE * SIZE);

    matrix::multiply(out, m, m, SIZE, SIZE, SIZE);

    std::cout << "done.\n";

    delete[] buf;
    delete[] outbuf;
}