package matrix;

public class MatrixException extends RuntimeException {
    public MatrixException() { this(""); }
    public MatrixException(String why) {
        super(why);
    }
}
