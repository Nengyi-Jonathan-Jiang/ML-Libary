package neuralnet;

import matrix.Matrix;
import neuralnet.layer.Layer;
import neuralnet.neuron.LossFunction;

import java.util.ArrayList;
import java.util.List;

import static neuralnet.layer.Layer.BackpropogationResult;

public class Model {
    private final List<Layer> layers = new ArrayList<>();
    private final LossFunction lossFunction;
    private final double learningRate;

    public record DataPoint(Matrix input, Matrix output) {}

    public Model(LossFunction lossFunction, double learningRate) {
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public double train(Matrix input, Matrix expected) {
        return train(new DataPoint(input, expected));
    }

    public double train(DataPoint data) {
        return train(new DataPoint[]{data});
    }

    public double train(DataPoint[] data) {
        // TODO: convert to throw error
        int numDataPoints = data.length;
        if(numDataPoints == 0) return Double.NaN;

        Matrix[] totalGradient = new Matrix[getNumLayers()];
        double totalLoss = 0;

        for(DataPoint dataPoint : data) {

            // Feed data point into network
            Matrix predicted = dataPoint.input;
            for(Layer l : layers) {
                l.acceptInput(predicted);
                predicted = l.getOutputs();
            }

            // Get loss
            double loss = lossFunction.apply(predicted, dataPoint.output);

            totalLoss += loss;

            // Backpropagate
            Matrix backpropagation_gradient = lossFunction.applyGradient(predicted, dataPoint.output);
            for(int i = getNumLayers() - 1; i >= 0; i--) {
                BackpropogationResult backpropogationResult = layers.get(i).backpropagate(backpropagation_gradient);
                backpropagation_gradient = backpropogationResult.gradient_wrt_inputs();
                Matrix weights_gradient = backpropogationResult.gradient_wrt_weights();

                if(totalGradient[i] == null) {
                    totalGradient[i] = weights_gradient;
                }
                else {
                    totalGradient[i] = totalGradient[i].plus(weights_gradient);
                }
            }
        }

        for(int i = 0; i < getNumLayers(); i++) {
            Matrix gradient = totalGradient[i].times(1. / numDataPoints);
            layers.get(i).updateWeights(gradient.times(-learningRate));
        }

        return totalLoss / numDataPoints;
    }

    private int getNumLayers() {
        return layers.size();
    }

    public Matrix run(Matrix input) {
        Matrix predicted = input;
        for(Layer l : layers) {
            l.acceptInput(predicted);
            predicted = l.getOutputs();
        }
        return predicted;
    }
}
