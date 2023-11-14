package matrix;

public sealed interface ColumnVector extends Vector permits ColumnVectorImplementation, DegenerateMatrix {
    @Override
    default int columns() {
        return 1;
    }

    @Override
    ColumnVector copy();

    @Override
    RowVector transpose();
}
