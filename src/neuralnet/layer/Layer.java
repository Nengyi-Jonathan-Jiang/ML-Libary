package neuralnet.layer;

import matrix.Matrix;
import matrix.RowVector;

public abstract class Layer {
    protected Matrix weights;

    public abstract void acceptInput(RowVector input);
    public abstract RowVector getOutputs();

    public abstract void updateWeights(Matrix amount);

    public record BackpropogationResult(RowVector gradient_wrt_inputs, Matrix gradient_wrt_weights){}

    public abstract RowVector backpropagate(RowVector gradient_wrt_output, Matrix gradient_wrt_weights_accumulator);

    public Matrix getWeights() {
        return weights;
    }
}
