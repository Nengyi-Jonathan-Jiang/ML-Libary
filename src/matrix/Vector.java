package matrix;

sealed interface Vector extends Matrix permits ColumnVector, RowVector, VectorBase {
    double getElementAt(int index);
    void setElementAt(int index, double value);

    double dot(Vector other);
    Matrix outer(Vector other);
    Matrix outer_to(Vector other, Matrix res);
}
