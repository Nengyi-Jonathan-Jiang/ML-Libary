package neuralnet;

import matrix.Matrix;
import neuralnet.layer.InputLayer;
import neuralnet.layer.Layer;
import neuralnet.neuron.LossFunction;

import java.util.ArrayList;
import java.util.List;

public class Model {
    private final List<Layer> layers = new ArrayList<>();
    private final LossFunction lossFunction;

    public Model(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
        layers.add(new InputLayer());
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public double train(Matrix input, Matrix expected, double learningRate) {
        Matrix predicted = input;
        for(Layer l : layers) {
            l.acceptInput(predicted);
            predicted = l.getOutputs();
        }
        double loss = lossFunction.apply(predicted, expected);
        Matrix gradient = lossFunction.applyGradient(predicted, expected);
        for(int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers.get(i).updateWeightsAndBackpropagate(gradient, learningRate);
        }

        return loss;
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
