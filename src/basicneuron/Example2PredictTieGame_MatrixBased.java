package basicneuron;

import chart.LineChart;
import matrix.Matrix;
import matrix.MatrixFactory;
import matrix.RowVector;
import neuralnet.Model;
import neuralnet.layer.DenseLayer;
import neuralnet.neuron.ActivationFunction;
import neuralnet.neuron.LossFunction;

public class Example2PredictTieGame_MatrixBased {
    private static final double LEARNING_RATE = 0.2;
    private static final int BATCH_SIZE = 1000;

    private static double[] __generate_test_case() {
        double WR = Math.random() * 0.6 + 0.2;
        double AG = Math.random() * 3.5 + 0.5;
        double WR_eva = 0.5;
        double AG_eva = 3;

        double __skill_difference = AG - AG_eva + 0 * 2 * (WR - WR_eva);// + Math.random() - 0.5;
        double didHaveTie = Math.abs(__skill_difference) <= 2 ? 1 : 0;
        double didTeamScoreHigh = Math.max(AG * (WR_eva + 0.5) / 2, AG_eva * (WR + 0.5) / 2) >= 4 ? 1 : 0;

        return new double[]{WR, AG, didHaveTie, didTeamScoreHigh};
    }

    public static void main(String[] args) {
        Model model = new Model(LossFunction.LogLoss, LEARNING_RATE);
        model.addLayer(new DenseLayer(3, 3, ActivationFunction.ReLU));
        model.addLayer(new DenseLayer(3, 3, ActivationFunction.ReLU));
        model.addLayer(new DenseLayer(3, 2, ActivationFunction.Sigmoid));
//        model.addLayer(new DenseLayer(3, 2, ActivationFunction.Sigmoid));

        LineChart chart = new LineChart();

        for (int i = 0; i < 10000; i++) {
            Model.DataPoint[] dataPoints = new Model.DataPoint[BATCH_SIZE];
            for(int j = 0; j < BATCH_SIZE; j++) {
                double[] testCase = __generate_test_case();

                RowVector inputs = MatrixFactory.rowVector(1, testCase[0], testCase[1]);
                RowVector outputs = MatrixFactory.rowVector(testCase[2], testCase[3]);
                Model.DataPoint dataPoint = new Model.DataPoint(inputs, outputs);
                dataPoints[j] = dataPoint;
            }

            double avgLoss = model.train(dataPoints);
            int accuracy_tie = 0;
            int accuracy_highscore = 0;


            chart.addPoint(i, avgLoss, "Matrix");
        }
    }
}