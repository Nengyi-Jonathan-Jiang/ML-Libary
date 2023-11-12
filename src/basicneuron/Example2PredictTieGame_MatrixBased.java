package basicneuron;

import matrix.Matrix;
import neuralnet.Model;
import neuralnet.layer.DenseLayer;
import neuralnet.neuron.ActivationFunction;
import neuralnet.neuron.LossFunction;
import neuralnet.neuron.Neuron;
import neuralnet.util.ArrayUtil;

public class Example2PredictTieGame_MatrixBased {
    private static final double LEARNING_RATE = 0.1;
    private static final int BATCH_SIZE = 100;

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
        Model model = new Model(LossFunction.LogLoss, LEARNING_RATE);
        model.addLayer(new DenseLayer(3, 3, ActivationFunction.ReLU));
        model.addLayer(new DenseLayer(3, 2, ActivationFunction.Sigmoid));

        for (int i = 0; i < 1000; i++) {
            Model.DataPoint[] dataPoints = new Model.DataPoint[BATCH_SIZE];
            for(int j = 0; j < BATCH_SIZE; j++) {
                double[] testCase = __generate_test_case();

                Matrix inputs = new Matrix(new double[][]{{1, testCase[0], testCase[1]}});
                Matrix outputs = new Matrix(new double[][]{{testCase[2], testCase[3]}});
                Model.DataPoint dataPoint = new Model.DataPoint(inputs, outputs);
                dataPoints[j] = dataPoint;
            }

            double avgLoss = model.train(dataPoints);

            {
                Model.DataPoint dataPoint = dataPoints[0];
                model.run(dataPoint.input());
            }

            System.out.println(avgLoss);
        }
    }
}
