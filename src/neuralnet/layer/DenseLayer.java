package neuralnet.layer;

import matrix.Matrix;

public class DenseLayer extends Layer {
    private final int numInputNodes, numOutputNodes;
    public Matrix input;
    public Matrix output;
    public Matrix weights;

    public DenseLayer(int numInputNodes, int numOutputNodes) {
        this.numInputNodes = numInputNodes;
        this.numOutputNodes = numOutputNodes;

        this.weights = Matrix.randomGaussian(numInputNodes, numOutputNodes).times(1 / Math.sqrt(numInputNodes));
    }

    /**
     * @param input a row vector <1, rows>
     */
    @Override
    public void acceptInput(Matrix input) {
        this.input = input;
        this.output = input.times(weights);
    }



    @Override
    public Matrix getOutput() {
        return output;
    }
}
