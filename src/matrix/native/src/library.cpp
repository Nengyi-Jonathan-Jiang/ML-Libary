#include "library.h"
#include "matrix.h"

matrix matrix_from_buffer(JNIEnv* env, jobject buf, int M, int N) {
    return {static_cast<double *>(env->GetDirectBufferAddress(buf)), M, N};
}

void JNICALL Java_matrix_MatrixOperations_transpose
        (JNIEnv *env, jclass, jobject out, jobject X, jint M, jint N) {
    matrix out_m = matrix_from_buffer(env, out, N, M);
    matrix X_m = matrix_from_buffer(env, X, M, N);
    matrix::transpose(out_m, X_m, M, N);
}

void JNICALL Java_matrix_MatrixOperations_multiply__Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2III
        (JNIEnv *env, jclass, jobject out, jobject A, jobject B, jint M, jint N, jint K) {
    const matrix &out_m = matrix_from_buffer(env, out, M, N);
    const matrix &A_m = matrix_from_buffer(env, A, M, K);
    const matrix &B_m = matrix_from_buffer(env, B, K, N);
    matrix::multiply(out_m, A_m, B_m, M, N, K);
}

void JNICALL Java_matrix_MatrixOperations_multiply__Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2II
        (JNIEnv *env, jclass cls, jobject out, jobject A, jobject B, jint M, jint N) {
    const matrix &out_m = matrix_from_buffer(env, out, M, N);
    const matrix &A_m = matrix_from_buffer(env, A, M, N);
    const matrix &B_m = matrix_from_buffer(env, B, M, N);
    matrix::multiply(out_m, A_m, B_m, M, N);
}

void JNICALL Java_matrix_MatrixOperations_multiply__Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2DDD
        (JNIEnv *env, jclass cls, jobject out, jobject X, jdouble c, jdouble M, jdouble N) {
    const matrix &out_m = matrix_from_buffer(env, out, M, N);
    const matrix &X_m = matrix_from_buffer(env, X, M, N);
    matrix::multiply(out_m, X_m, c, M, N);
}

void JNICALL Java_matrix_MatrixOperations_add__Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2II
        (JNIEnv *env, jclass cls, jobject out, jobject A, jobject B, jint M, jint N) {
    const matrix &out_m = matrix_from_buffer(env, out, M, N);
    const matrix &A_m = matrix_from_buffer(env, A, M, N);
    const matrix &B_m = matrix_from_buffer(env, B, M, N);
    matrix::add(out_m, A_m, B_m, M, N);
}

void JNICALL Java_matrix_MatrixOperations_add__Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2DII
        (JNIEnv *env, jclass cls, jobject out, jobject X, jdouble c, jint M, jint N) {
    const matrix &out_m = matrix_from_buffer(env, out, M, N);
    const matrix &X_m = matrix_from_buffer(env, X, M, N);
    matrix::add(out_m, X_m, c, M, N);
}

void JNICALL Java_matrix_MatrixOperations_subtract
        (JNIEnv *env, jclass cls, jobject out, jobject A, jobject B, jint M, jint N) {
    const matrix &out_m = matrix_from_buffer(env, out, M, N);
    const matrix &A_m = matrix_from_buffer(env, A, M, N);
    const matrix &B_m = matrix_from_buffer(env, B, M, N);
    matrix::subtract(out_m, A_m, B_m, M, N);
}

void JNICALL Java_matrix_MatrixOperations_multiplyAndAdd
        (JNIEnv *env, jclass, jobject out, jobject A, jobject B, jint M, jint N, jint K) {
    const matrix &out_m = matrix_from_buffer(env, out, M, N);
    const matrix &A_m = matrix_from_buffer(env, A, M, K);
    const matrix &B_m = matrix_from_buffer(env, B, K, N);
    matrix::multiplyAndAdd(out_m, A_m, B_m, M, N, K);
}

jdouble JNICALL Java_matrix_MatrixOperations_dot
        (JNIEnv *env, jclass, jobject A, jobject B, jint K) {
    const matrix &A_m = matrix_from_buffer(env, A, 1, K);
    const matrix &B_m = matrix_from_buffer(env, B, 1, K);
    return matrix::dot(A_m, B_m, K);
}

void JNICALL Java_matrix_MatrixOperations_outer
        (JNIEnv *env, jclass, jobject out, jobject A, jobject B, jint M, jint N) {
    const matrix &out_m = matrix_from_buffer(env, out, M, N);
    const matrix &A_m = matrix_from_buffer(env, A, M, 1);
    const matrix &B_m = matrix_from_buffer(env, B, 1, N);
    matrix::multiply(out_m, A_m, B_m, M, N, 1);
}
