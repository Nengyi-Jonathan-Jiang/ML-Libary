package matrix;

public sealed interface RowVector extends Vector permits RowVectorImplementation, DegenerateMatrix {
    @Override
    default int rows() {
        return 1;
    }

    @Override
    RowVector copy();

    @Override
    ColumnVector transpose();


}