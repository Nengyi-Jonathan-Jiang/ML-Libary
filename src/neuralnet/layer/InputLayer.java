package neuralnet.layer;

import matrix.Matrix;

public class InputLayer implements Layer {
    private Matrix values;

    @Override
    public void acceptInput(Matrix input) {
        values = input;
    }

    @Override
    public Matrix getOutputs() {
        return values;
    }

    @Override
    public Matrix updateWeightsAndBackpropagate(Matrix gradient, double learningRate) {
        return null;
    }
}
