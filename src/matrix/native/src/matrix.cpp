#include <cstring>
#include "matrix.h"


void matrix::transpose(const matrix &out, const matrix &X, int M, int N) {
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            out(j, i) = X(i, j);
}

void matrix::multiplyAndAdd(const matrix &out, const matrix &A, const matrix &B, int M, int N, int K) {
    for(int i = 0; i < M; i++)
        for(int k = 0; k < K; k++)
            for(int j = 0; j < N; j++)
                out[i * N + j] += A[i * K + k] * B[k * N + j];
}

void matrix::multiply(const matrix &out, const matrix &A, const matrix &B, int M, int N, int K) {
    memset(out.data, 0, M * N * 8);
    multiplyAndAdd(out, A, B, M, N, K);
}

void matrix::multiply(const matrix &out, const matrix &A, const matrix &B, int M, int N) {
    for(int i = 0; i < M * N; i++)
        out[i] = A[i] * B[i];
}

void matrix::multiply(const matrix &out, const matrix &X, double c, double M, double N) {
    for(int i = 0; i < M * N; i++)
        out[i] = X[i] * c;
}

void matrix::add(const matrix &out, const matrix &A, const matrix &B, int M, int N) {
    for(int i = 0; i < M * N; i++)
        out[i] = A[i] + B[i];
}

void matrix::add(const matrix &out, const matrix &X, double c, int M, int N) {
    for(int i = 0; i < M * N; i++)
        out[i] = X[i] + c;
}

void matrix::subtract(const matrix &out, const matrix &A, const matrix &B, int M, int N) {
    for(int i = 0; i < M * N; i++)
        out[i] = A[i] - B[i];
}

double matrix::dot(const matrix &A, const matrix &B, int N) {
    double res = 0;
    for(int i = 0; i < N; i++) res += A[i] * B[i];
    return res;
}

