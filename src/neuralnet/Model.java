package neuralnet;

import matrix.Matrix;
import neuralnet.layer.InputLayer;
import neuralnet.layer.Layer;

import java.util.ArrayList;
import java.util.List;

public class Model {
    private final List<Layer> layers = new ArrayList<>();

    public Model() {
        layers.add(new InputLayer());
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public void train(Matrix input, Matrix expected, double learningRate) {
        Matrix predicted = input;
        for(Layer l : layers) {
            l.acceptInput(predicted);
            predicted = l.getOutput();
        }

    }
}
