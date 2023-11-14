package matrix;

import java.nio.DoubleBuffer;

final class RectangularMatrix extends MatrixBaseWithOperations {
    private final int rows, columns;

    protected RectangularMatrix(int rows, int columns) {
        super(allocateDoubleBuffer(rows * columns));

        assert(rows > 1 && columns > 1);

        this.rows = rows;
        this.columns = columns;
    }

    /**
     * Constructs a matrix from raw data in column major order <br>
     */
    protected RectangularMatrix(DoubleBuffer buf, int rows, int columns) {
        super(buf);

        assert(rows > 1 && columns > 1);

        this.rows = rows;
        this.columns = columns;
    }

    @Override
    public int rows() {
        return rows;
    }

    @Override
    public int columns() {
        return columns;
    }

    @Override
    public double getElementAt(int row, int column) {
        return buffer.get(row * columns + column);
    }

    @Override
    public void setElementAt(int row, int column, double value) {
        buffer.put(row * columns + column, value);
    }

    @Override
    public Matrix copy() {
        return new RectangularMatrix(copyDoubleBuffer(buffer), rows, columns);
    }


    public RowVector getRow(int row) {
        DoubleBuffer columnData = buffer.slice(row * columns, columns);
        return new RowVectorImplementation(columnData, columns);
    }

    public ColumnVector getColumn(int column) {
        DoubleBuffer res = allocateDoubleBuffer(rows);
        for (int r = 0; r < rows; r++) res.put(r, getElementAt(r, column));
        return new ColumnVectorImplementation(res, rows);
    }

}
