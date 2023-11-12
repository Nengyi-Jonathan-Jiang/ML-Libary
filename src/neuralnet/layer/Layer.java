package neuralnet.layer;

import matrix.Matrix;

public interface Layer {

    void acceptInput(Matrix input);
    Matrix getOutputs();

    void updateWeights(Matrix amount);

    record BackpropogationResult(Matrix gradient_wrt_inputs, Matrix gradient_wrt_weights){}

    BackpropogationResult backpropagate(Matrix gradient);
}
