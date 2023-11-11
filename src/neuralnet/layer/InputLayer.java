package neuralnet.layer;

import matrix.Matrix;

public class InputLayer extends Layer {
    private Matrix values;

    @Override
    public void acceptInput(Matrix input) {
        values = input;
    }

    @Override
    public Matrix getOutput() {
        return values;
    }
}
