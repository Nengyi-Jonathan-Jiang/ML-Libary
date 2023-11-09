package neuralnet.neuron;

import java.util.Random;

public class Neuron {
    private static final Random random = new Random();

    private final int numInputs;
    private double output;
    private final double[] weights;

    public Neuron(int numInputs, ActivationFunction activationFunction) {
        this.numInputs = numInputs;

        // Xavier initialization of weights:
        weights = new double[numInputs];
        for(int i = 0; i < numInputs; i++) {
            weights[i] = random.nextGaussian() / Math.sqrt(numInputs);
        }
    }

    public double acceptInputs(double... inputs) {
        if(inputs.length != numInputs) {
            throw new NeuronInputLengthException(numInputs, inputs.length);
        }
        double total = 0;
        for(int i = 0; i < numInputs; i++) {
            total += inputs[i] * weights[i];
        }

        return output = total;
    }

    public double getOutput() {
        return output;
    }

    public double runGradientDescent() {

    }
}
