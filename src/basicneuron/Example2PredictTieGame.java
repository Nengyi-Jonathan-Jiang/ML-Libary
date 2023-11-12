package basicneuron;

import chart.LineChart;
import neuralnet.neuron.ActivationFunction;
import neuralnet.neuron.Neuron;
import neuralnet.util.ArrayUtil;

public class Example2PredictTieGame {
    private static final double LEARNING_RATE = 0.1, BATCH_SIZE = 100;

    private static double[] __generate_test_case() {
        double WR = Math.random() * 0.6 + 0.2;
        double AG = Math.random() * 2 + 0.5;
        double WR_eva = 0.7;
        double AG_eva = 3.5;

        double __skill_difference = AG - AG_eva + 5 * (WR - WR_eva) + Math.random() - 0.5;
        double didHaveTie = Math.abs(__skill_difference) <= 0.2 ? 1 : 0;
        double didTeamScoreHigh = Math.max(AG * (WR_eva + 0.5) / 2, AG_eva * (WR + 0.5) / 2) >= 4 ? 1 : 0;

        return new double[]{WR, AG, didHaveTie, didTeamScoreHigh};
    }

    public static void main(String[] args) {
        final Neuron n11 = new Neuron(3, ActivationFunction.ReLU),
               n12 = new Neuron(3, ActivationFunction.ReLU),
               n13 = new Neuron(3, ActivationFunction.ReLU),
               n21 = new Neuron(3, ActivationFunction.Sigmoid),
               n22 = new Neuron(3, ActivationFunction.Sigmoid);

        LineChart c = new LineChart();

        for (int i = 0; i < 1000; i++) {
            double[] totalGradient21 = new double[3];
            double[] totalGradient11 = new double[3];
            double[] totalGradient12 = new double[3];
            double[] totalGradient13 = new double[3];
            double totalLoss = 0;
            for(int j = 0; j < BATCH_SIZE; j++) {
                double[] testCase = __generate_test_case();
                double[] inputs = new double[]{1, testCase[0], testCase[1]};
                double expected_team_score_high = testCase[2];

                n11.acceptInputs(inputs);
                n12.acceptInputs(inputs);
                n13.acceptInputs(inputs);

                double[] layer1Outputs = new double[]{n11.getOutput(), n12.getOutput(), n13.getOutput()};

                n21.acceptInputs(layer1Outputs);

                double predicted = n21.getOutput();

                double loss = expected_team_score_high == 0 ? -Math.log(1 - predicted) : -Math.log(predicted);
                totalLoss += loss;
                double loss_derivative = (expected_team_score_high == 0 ? 1 / (1 - predicted) : -1 / predicted);

                // These gradients need only be computed once per training cycle
                double[] d_output_weights_n21 = n21.getDerivative_OutputWRTWeights();
                double[] d_output_weights_n11 = n11.getDerivative_OutputWRTWeights();
                double[] d_output_weights_n12 = n12.getDerivative_OutputWRTWeights();
                double[] d_output_weights_n13 = n13.getDerivative_OutputWRTWeights();
                // We don't need the partial derivative of output WRT input for the neurons in the first layer
                // since they are not used for backpropagation
                double[] d_output_input_n21 = n21.getDerivative_OutputWRTInputs();

                // We can reuse these values! These are the derivatives of the loss wrt each input to n21
                double[] backPropagation_n21 = ArrayUtil.multiplyByConstant(d_output_input_n21, loss_derivative);

                // Now we are ready to compute the gradients
                double[] gradient_n21 = ArrayUtil.multiplyByConstant(d_output_weights_n21, loss_derivative);
                double[] gradient_n11 = ArrayUtil.multiplyByConstant(d_output_weights_n11, backPropagation_n21[0]);
                double[] gradient_n12 = ArrayUtil.multiplyByConstant(d_output_weights_n12, backPropagation_n21[1]);
                double[] gradient_n13 = ArrayUtil.multiplyByConstant(d_output_weights_n13, backPropagation_n21[2]);

                totalGradient21 = ArrayUtil.sumElementwise(totalGradient21, gradient_n21);
                totalGradient11 = ArrayUtil.sumElementwise(totalGradient11, gradient_n11);
                totalGradient12 = ArrayUtil.sumElementwise(totalGradient12, gradient_n12);
                totalGradient13 = ArrayUtil.sumElementwise(totalGradient13, gradient_n13);
            }

            double[] tweaks_n21 = ArrayUtil.multiplyByConstant(totalGradient21, -LEARNING_RATE / BATCH_SIZE);
            double[] tweaks_n11 = ArrayUtil.multiplyByConstant(totalGradient11, -LEARNING_RATE / BATCH_SIZE);
            double[] tweaks_n12 = ArrayUtil.multiplyByConstant(totalGradient12, -LEARNING_RATE / BATCH_SIZE);
            double[] tweaks_n13 = ArrayUtil.multiplyByConstant(totalGradient13, -LEARNING_RATE / BATCH_SIZE);

            n21.frobnicateWeights(tweaks_n21);
            n11.frobnicateWeights(tweaks_n11);
            n12.frobnicateWeights(tweaks_n12);
            n13.frobnicateWeights(tweaks_n13);

            double avg_loss_iteration = totalLoss / BATCH_SIZE;

            c.addPoint(i, avg_loss_iteration, "Normal");
        }
    }
}
