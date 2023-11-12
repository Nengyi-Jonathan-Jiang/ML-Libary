package neuralnet.layer;

import matrix.Matrix;

public abstract class Layer {
    protected Matrix weights;

    public abstract void acceptInput(Matrix input);
    public abstract Matrix getOutputs();

    public abstract void updateWeights(Matrix amount);

    public static record BackpropogationResult(Matrix gradient_wrt_inputs, Matrix gradient_wrt_weights){}

    public abstract Matrix backpropagate(Matrix gradient_wrt_output, Matrix gradient_wrt_weights_accumulator);

    public Matrix getWeights() {
        return weights;
    }
}
