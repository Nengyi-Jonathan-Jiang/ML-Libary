package neuralnet.neuron;

import neuralnet.util.ArrayUtil;

import java.util.Random;

public class Neuron {
    private static final Random random = new Random();

    private final int numInputs;
    private double weightedSumOfInputs;
    private double output;
    private double[] inputs;

    private final double[] partialDerivativesWRTInputs;
    private final double[] partialDerivativesWRTWeights;
    private final double[] weights;
    private final ActivationFunction activationFunction;

    public Neuron(int numInputs, ActivationFunction activationFunction) {
        this.numInputs = numInputs;

        // Xavier initialization of weights:
        weights = new double[numInputs];
        this.activationFunction = activationFunction;
        for(int i = 0; i < numInputs; i++) {
            weights[i] = random.nextGaussian() / Math.sqrt(numInputs);
        }

        // Initialize all result arrays
        inputs = new double[numInputs];
        partialDerivativesWRTInputs = new double[numInputs];
        partialDerivativesWRTWeights = new double[numInputs];
    }

    public void acceptInputs(double... inputs) {
        validateArrayLength(inputs);
        this.inputs = inputs;

        recalculateWeightedSumOfInputs();
        recalculateNeuronOutput();
        recalculatePartialDerivativesWRTInputs();
        recalculatePartialDerivativesWRTWeights();
    }

    private void recalculateNeuronOutput() {
        output = activationFunction.apply(weightedSumOfInputs);
    }

    private void recalculateWeightedSumOfInputs() {
        double total = 0;

        for(int i = 0; i < numInputs; i++) {
            total += inputs[i] * weights[i];
        }

        weightedSumOfInputs = total;
    }

    private void recalculatePartialDerivativesWRTInputs() {
        double activationFunctionDerivative = activationFunction.applyDerivative(weightedSumOfInputs);
        ArrayUtil.map(weights, i -> i * activationFunctionDerivative, partialDerivativesWRTInputs);
    }

    private void recalculatePartialDerivativesWRTWeights() {
        double activationFunctionDerivative = activationFunction.applyDerivative(weightedSumOfInputs);
        ArrayUtil.map(inputs, i -> i * activationFunctionDerivative, partialDerivativesWRTWeights);
    }

    public double[] getDerivative_OutputWRTInputs() {
        return partialDerivativesWRTInputs;
    }

    public double[] getDerivative_OutputWRTWeights() {
        return partialDerivativesWRTWeights;
    }

    public void frobnicateWeights(double[] tweaks) {
        validateArrayLength(tweaks);
        ArrayUtil.apply(weights, tweaks, Double::sum);
    }

    public double getWeightedSumOfInputs() {
        return weightedSumOfInputs;
    }

    public double getOutput() {
        return output;
    }

//    public double runGradientDescent() {
//
//    }

    private void validateArrayLength(double[] arr) {
        if(arr.length != numInputs) {
            throw new NeuronInputLengthException(numInputs, arr.length);
        }
    }

    public double[] getWeights() {
        return weights;
    }
}
