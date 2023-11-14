package matrix.exception;

public class MatrixException extends RuntimeException {
    public MatrixException() { this(""); }
    public MatrixException(String why) {
        super(why);
    }
}
