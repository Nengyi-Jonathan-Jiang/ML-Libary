#pragma once

struct matrix {
    double* const data;
    [[maybe_unused]] const int M, N;

    matrix(double* data, int M, int N) : data(data), M(M), N(N) { }

    inline double& operator[](int x) const {
        return data[x];
    }

    inline double& operator()(int i, int j) const {
        return data[i * N + j];
    }

    static void transpose(const matrix& out, const matrix& X, int M, int N);

    static void multiply(const matrix& out, const matrix& A, const matrix& B, int M, int N, int K);

    static void multiply(const matrix& out, const matrix& A, const matrix& B, int M, int N);

    static void multiply(const matrix& out, const matrix& X, double c, double M, double N);

    static void multiplyAndAddOptimized(const matrix &out, const matrix &A, const matrix &B, int M, int N, int K);

    static void add(const matrix& out, const matrix& A, const matrix& B, int M, int N);

    static void add(const matrix& out, const matrix& X, double c, int M, int N);

    static void subtract(const matrix& out, const matrix& A, const matrix& B, int M, int N);

    static void multiplyAndAdd(const matrix &out, const matrix &A, const matrix &B, int M, int N, int K);

    static double dot(const matrix &A, const matrix &B, int K);
};
