package neuralnet.layer;

import matrix.Matrix;

public interface Layer {

    void acceptInput(Matrix input);
    Matrix getOutputs();

    Matrix updateWeightsAndBackpropagate(Matrix gradient, double learningRate);
}
