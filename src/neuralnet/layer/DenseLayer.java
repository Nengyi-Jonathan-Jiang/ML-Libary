package neuralnet.layer;

import matrix.Matrix;
import neuralnet.neuron.ActivationFunction;

public class DenseLayer extends Layer {
    private final int numInputNodes;
    private final int numOutputNodes;

    private final ActivationFunction activationFunction;
    /** <1, numInputNodes> */
    public Matrix inputs;
    /** <1, numOutputNodes> */
    public Matrix weightedSums;
    /** <1, numOutputNodes> */
    public Matrix outputs;

    /** <1, numOutputNodes> */
    private Matrix gradient_wrt_weightedSums;
    /** <1, numInputNodes> */
    private Matrix gradient_wrt_inputs;


    public DenseLayer(int numInputNodes, int numOutputNodes, ActivationFunction activationFunction) {
        this(
            numInputNodes,
            numOutputNodes,
            activationFunction,
            Matrix.randomUniform(numInputNodes, numOutputNodes).times(1 / Math.sqrt(numInputNodes))
        );
    }

    public DenseLayer(int numInputNodes, int numOutputNodes, ActivationFunction activationFunction, Matrix weights) {
        this.numInputNodes = numInputNodes;
        this.numOutputNodes = numOutputNodes;
        this.activationFunction = activationFunction;

        this.inputs = Matrix.create(1, numInputNodes);
        this.weightedSums = Matrix.create(1, numOutputNodes);
        this.outputs = Matrix.create(1, numOutputNodes);

        this.weights = weights;

        this.gradient_wrt_weightedSums = Matrix.create(1, numOutputNodes);
        this.gradient_wrt_inputs = Matrix.create(1, numInputNodes);
    }

    /**
     * @param input a row vector <1, rows>
     */
    @Override
    public void acceptInput(Matrix input) {
        this.inputs = input;
        Matrix.multiply(input, weights, this.weightedSums);
        Matrix.applyOperation(weightedSums, activationFunction::apply, this.outputs);
    }

    @Override
    public Matrix getOutputs() {
        return outputs;
    }

    @Override
    public void updateWeights(Matrix amount) {
        weights = weights.plus(amount);
    }

    /**
     * @param gradient_wrt_output              A row vector with size = numOutputs
     * @param gradient_wrt_weights_accumulator
     * @return
     */
    @Override
    public Matrix backpropagate(Matrix gradient_wrt_output, Matrix gradient_wrt_weights_accumulator) {
        Matrix.applyOperation(weightedSums, activationFunction::applyDerivative, gradient_wrt_weightedSums);
        Matrix.multiply_elementWise(gradient_wrt_weightedSums, gradient_wrt_output, gradient_wrt_weightedSums);

        gradient_wrt_inputs = gradient_wrt_inputs.transpose();
        Matrix.multiply(weights, gradient_wrt_weightedSums.transpose(), gradient_wrt_inputs);
        gradient_wrt_inputs = gradient_wrt_inputs.transpose();

        Matrix.addProductTo(inputs.transpose(), gradient_wrt_weightedSums, gradient_wrt_weights_accumulator);

//        new BackpropogationResult(gradient_wrt_inputs.copy(), gradient_wrt_weights.copy());
        return gradient_wrt_inputs;
    }
}