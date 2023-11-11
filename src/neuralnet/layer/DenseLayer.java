package neuralnet.layer;

import matrix.Matrix;
import neuralnet.neuron.ActivationFunction;

public class DenseLayer implements Layer {
    private final int numInputNodes;
    private final int numOutputNodes;
    private final ActivationFunction activationFunction;
    /** <1, numInputNodes> */
    public Matrix inputs;
    /** <1, numOutputNodes> */
    public Matrix weightedSums;
    /** <1, numOutputNodes> */
    public Matrix outputs;
    /** &lt;numInputNodes, numOutputNodes&gt; */
    public Matrix weights;

    public DenseLayer(int numInputNodes, int numOutputNodes, ActivationFunction activationFunction) {
        this.numInputNodes = numInputNodes;
        this.numOutputNodes = numOutputNodes;
        this.activationFunction = activationFunction;

        this.weights = Matrix.randomGaussian(numInputNodes, numOutputNodes).times(1 / Math.sqrt(numInputNodes));
    }

    /**
     * @param input a row vector <1, rows>
     */
    @Override
    public void acceptInput(Matrix input) {
        this.inputs = input;
        this.weightedSums = input.times(weights).transpose();
        this.outputs = Matrix.applyOperation(weightedSums, activationFunction::apply);
    }



    @Override
    public Matrix getOutputs() {
        return outputs;
    }

    /**
     * @param gradient_wrt_outputs     A row vector with size = numOutputs
     * @param learningRate Learning rate
     * @returns A row vector with size = numInputs
     */
    @Override
    public Matrix updateWeightsAndBackpropagate(Matrix gradient_wrt_outputs, double learningRate) {
        // <1, outputNodes>
        Matrix gradient_wrt_weightedSums =
                Matrix.applyOperation(weightedSums, activationFunction::applyDerivative)
                .times_elementwise(gradient_wrt_outputs);

        // <1, outputNodes> x <outputNodes, inputNodes> = <1, inputNodes>
        Matrix gradient_wrt_inputs = gradient_wrt_weightedSums.times(weights.transpose());

        // <outputNodes, 1> x <1, inputNodes> = <outputNodes, inputNodes>
        Matrix gradient_wrt_weights = gradient_wrt_weightedSums.transpose().times(inputs);

        // Update weights
        weights = weights.minus(gradient_wrt_weights.times(-learningRate));

        return gradient_wrt_inputs;
    }
}
