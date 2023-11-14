package neuralnet.layer;

import matrix.Matrix;
import matrix.MatrixFactory;
import matrix.RowVector;
import neuralnet.neuron.ActivationFunction;

public class DenseLayer extends Layer {
    private final int numInputNodes;
    private final int numOutputNodes;

    private final ActivationFunction activationFunction;
    /** <1, numInputNodes> */
    public RowVector inputs;
    /** <1, numOutputNodes> */
    public final RowVector weightedSums;
    /** <1, numOutputNodes> */
    public RowVector outputs;

    /** <1, numOutputNodes> */
    private final RowVector gradient_wrt_weightedSums;
    /** <1, numInputNodes> */
    private final RowVector gradient_wrt_inputs;


    public DenseLayer(int numInputNodes, int numOutputNodes, ActivationFunction activationFunction) {
        this(
            numInputNodes,
            numOutputNodes,
            activationFunction,
            MatrixFactory.randomUniform(numInputNodes, numOutputNodes).times(1 / Math.sqrt(numInputNodes))
        );
    }

    public DenseLayer(int numInputNodes, int numOutputNodes, ActivationFunction activationFunction, Matrix weights) {
        this.numInputNodes = numInputNodes;
        this.numOutputNodes = numOutputNodes;
        this.activationFunction = activationFunction;

        this.inputs = MatrixFactory.rowVector(numInputNodes);
        this.weightedSums = MatrixFactory.rowVector(numOutputNodes);
        this.outputs = MatrixFactory.rowVector(numOutputNodes);

        this.weights = weights;

        this.gradient_wrt_weightedSums = MatrixFactory.rowVector(numOutputNodes);
        this.gradient_wrt_inputs = MatrixFactory.rowVector(numInputNodes);
    }

    /**
     * @param input a row vector <1, rows>
     */
    @Override
    public void acceptInput(RowVector input) {
        this.inputs = input;
        input.multiply_to(weights, weightedSums);
        Matrix.applyOperation(weightedSums, activationFunction::apply, this.outputs);
    }

    @Override
    public RowVector getOutputs() {
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
    public RowVector backpropagate(RowVector gradient_wrt_output, Matrix gradient_wrt_weights_accumulator) {
        Matrix.applyOperation(weightedSums, activationFunction::applyDerivative, gradient_wrt_weightedSums);
        gradient_wrt_weightedSums.multiply_elementwise_to(gradient_wrt_output, gradient_wrt_weightedSums);

        weights.multiply_to(gradient_wrt_weightedSums.transpose(), gradient_wrt_inputs.transpose());

        inputs.transpose().multiply_and_add_to(gradient_wrt_weightedSums, gradient_wrt_weights_accumulator);

        return gradient_wrt_inputs;
    }
}