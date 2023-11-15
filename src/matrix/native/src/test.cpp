#include "matrix.h"
#include <iostream>
#include <random>

int main() {
    int SIZE = 1000;
    int ITERATIONS = 10;

    auto *buf = new double[SIZE * SIZE];
    auto *outbuf = new double[SIZE * SIZE];
    matrix m(buf, SIZE * SIZE, SIZE * SIZE);
    matrix out(outbuf, SIZE * SIZE, SIZE * SIZE);

    std::cout << "calculating..." << std::endl;

    for(int n = 0; n < ITERATIONS; n++) {
        std::cout << "iteration " << n << ": initializing; " << std::flush;

        for (int i = 0; i < SIZE * SIZE; i++) buf[i] = rand() / 1000.;

        std::cout << "multiplying; " << std::endl;

        matrix::multiply(out, m, m, SIZE, SIZE, SIZE);
    }

    std::cout << "done." << std::endl;

    delete[] buf;
    delete[] outbuf;
}