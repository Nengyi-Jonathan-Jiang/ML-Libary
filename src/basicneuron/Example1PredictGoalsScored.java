package basicneuron;

import neuralnet.neuron.ActivationFunction;
import neuralnet.neuron.Neuron;
import neuralnet.util.ArrayUtil;

public class Example1PredictGoalsScored {
    private static final double LEARNING_RATE = 0.1, BATCH_SIZE = 100;

    private static double[] __generate_test_case() {
        double WR = Math.random() * 0.6 + 0.2;
        double AG = Math.random() * 2 + 0.5;
        double result = (4.2 - 5 * WR + AG * 1.2 + (Math.random() - 0.5) * 0.2);

        return new double[]{WR, AG, result};
    }

    public static void main(String[] args) {
        Neuron neuron = new Neuron(3, ActivationFunction.ReLU);

        for (int i = 0; i < 1000; i++) {
            double[] gradient = new double[3];
            double totalLoss = 0;
            for(int j = 0; j < BATCH_SIZE; j++) {
                double[] testCase = __generate_test_case();
                double[] inputs = new double[]{1, testCase[0], testCase[1]};
                double expected = testCase[2];
                neuron.acceptInputs(inputs);
                double predicted = neuron.getOutput();

                double error = predicted - expected;
                double loss = error * error;
                totalLoss += loss;
                double loss_derivative = 2 * error;
                double[] weights_gradient = neuron.getDerivative_OutputWRTWeights();

                gradient = ArrayUtil.sumElementwise(gradient, ArrayUtil.multiplyByConstant(weights_gradient, loss_derivative));
            }

            double[] tweaks = ArrayUtil.multiplyByConstant(gradient, -LEARNING_RATE / BATCH_SIZE);

            double avgLoss = totalLoss / BATCH_SIZE;

            System.out.printf(
                "Weights = %.3f    %.3f    %.3f,    loss = %.3f\n",
                neuron.getWeights()[0], neuron.getWeights()[1], neuron.getWeights()[2],
                avgLoss
            );

            neuron.frobnicateWeights(tweaks);
        }
    }
}