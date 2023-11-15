package neuralnet;

import matrix.Matrix;
import matrix.MatrixFactory;
import matrix.RowVector;
import neuralnet.layer.Layer;
import neuralnet.neuron.LossFunction;

import java.util.ArrayList;
import java.util.List;

public class Model {
    private final List<Layer> layers = new ArrayList<>();
    private final LossFunction lossFunction;
    private final double learningRate;

    public record DataPoint(RowVector input, RowVector output) {}

    public Model(LossFunction lossFunction, double learningRate) {
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public double train(RowVector input, RowVector expected) {
        return train(new DataPoint(input, expected));
    }

    public double train(Matrix input, Matrix output) {
        // TODO: check dimensions

        Matrix[] total_gradients_wrt_weights = new Matrix[getNumLayers()];
        RowVector original_backpropagation_gradient = MatrixFactory.rowVector(output.columns());

        for(int i = 0; i < getNumLayers(); i++) {
            total_gradients_wrt_weights[i] = MatrixFactory.matrix(layers.get(i).getWeights());
        }
        double totalLoss = 0;

//        System.out.println("New batch================");

        int numDataPoints = input.rows();
        for(int row = 0; row < numDataPoints; row++) {

            // Feed data point into network
            RowVector predicted = input.getRow(row);
            for(Layer l : layers) {
                l.acceptInput(predicted);
                predicted = l.getOutputs();
            }

            // Get loss
            double loss = lossFunction.apply(predicted, output.getRow(row));

            totalLoss += loss;

            RowVector backpropagation_gradient = lossFunction.applyGradient(predicted, output.getRow(row), original_backpropagation_gradient);
            for(int i = getNumLayers() - 1; i >= 0; i--) {
                backpropagation_gradient = layers.get(i).backpropagate(backpropagation_gradient, total_gradients_wrt_weights[i]);
            }
        }

        for(int i = 0; i < getNumLayers(); i++) {
            layers.get(i).updateWeights(total_gradients_wrt_weights[i].times(-learningRate / numDataPoints));
        }

        return totalLoss / numDataPoints;
    }

    public double train(DataPoint data) {
        return train(new DataPoint[]{data});
    }

    public double train(DataPoint[] data) {
        // TODO: convert to throw error
        int numDataPoints = data.length;
        if(numDataPoints == 0) return Double.NaN;

        Matrix[] total_gradients_wrt_weights = new Matrix[getNumLayers()];
        RowVector original_backpropagation_gradient = MatrixFactory.rowVector(data[0].output.columns());

        for(int i = 0; i < getNumLayers(); i++) {
            total_gradients_wrt_weights[i] = MatrixFactory.matrix(layers.get(i).getWeights());
        }
        double totalLoss = 0;

//        System.out.println("New batch================");

        for(DataPoint dataPoint : data) {

            // Feed data point into network
            RowVector predicted = dataPoint.input;
            for(Layer l : layers) {
                l.acceptInput(predicted);
                predicted = l.getOutputs();
            }

            // Get loss
            double loss = lossFunction.apply(predicted, dataPoint.output);

            totalLoss += loss;

            RowVector backpropagation_gradient = lossFunction.applyGradient(predicted, dataPoint.output, original_backpropagation_gradient);
            for(int i = getNumLayers() - 1; i >= 0; i--) {
                backpropagation_gradient = layers.get(i).backpropagate(backpropagation_gradient, total_gradients_wrt_weights[i]);
            }
        }

        for(int i = 0; i < getNumLayers(); i++) {
            layers.get(i).updateWeights(total_gradients_wrt_weights[i].times(-learningRate / numDataPoints));
        }

        return totalLoss / numDataPoints;
    }

    private int getNumLayers() {
        return layers.size();
    }

    public RowVector run(RowVector input) {
        RowVector predicted = input;
        for(Layer l : layers) {
            l.acceptInput(predicted);
            predicted = l.getOutputs();
        }
        return predicted;
    }
}
