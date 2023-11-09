package neuralnet.neuron;

import neuralnet.NeuralNetException;

public class NeuronInputLengthException extends NeuralNetException {

    public NeuronInputLengthException(int expected, int actual) {
        super("Neuron input length mismatched: %d should be %d".formatted(actual, expected));
    }
}
