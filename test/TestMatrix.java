import matrix.Matrix;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TestMatrix {
    @Test
    public void testMatrixSum() {
        Matrix m1 = new Matrix(new double[][] {
            {1, 2},
            {3, 4},
            {5, 6}
        });

        Matrix m2 = new Matrix(new double[][] {
            {7, 8},
            {9, 10},
            {11, 12},
        });

        Matrix sum = new Matrix(new double[][] {
            {8, 10},
            {12, 14},
            {16, 18}
        });

        assertEquals(sum, m1.plus(m2));
    }

    @Test
    public void testMatrixProduct() {
        Matrix m1 = new Matrix(new double[][] {
            {1, 2},
            {3, 4},
            {5, 6}
        });

        Matrix m2 = new Matrix(new double[][] {
            {7, 8, 9, 10},
            {11, 12, 13, 14}
        });

        Matrix product = new Matrix(new double[][] {
            {29, 32, 35, 38},
            {65, 72, 79, 86},
            {101, 112, 123, 134}
        });

        assertEquals(product, m1.times(m2));
    }

    @Test
    public void testMatrixAccess() {
        Matrix m1 = new Matrix(new double[][]{
            {1, 2},
            {3, 4},
            {5, 6}
        });

        Matrix col0 = new Matrix(new double[][]{{1}, {3}, {5}});
        Matrix col1 = new Matrix(new double[][]{{2}, {4}, {6}});
        Matrix row0 = new Matrix(new double[][]{{1, 2}});
        Matrix row2 = new Matrix(new double[][]{{5, 6}});

        assertEquals(col0, m1.getColumn(0));
        assertEquals(col1, m1.getColumn(1));
        assertEquals(row0, m1.getRow(0));
        assertEquals(row2, m1.getRow(2));
    }
}
