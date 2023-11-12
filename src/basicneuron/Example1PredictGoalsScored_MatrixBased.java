package basicneuron;

import chart.LineChart;
import matrix.Matrix;
import neuralnet.Model;
import neuralnet.Model.DataPoint;
import neuralnet.layer.DenseLayer;
import neuralnet.neuron.ActivationFunction;
import neuralnet.neuron.LossFunction;

public class Example1PredictGoalsScored_MatrixBased {
    private static final double LEARNING_RATE = 0.1;
    private static final int BATCH_SIZE = 100;

    private static double[] __generate_test_case() {
        double WR = Math.random() * 0.6 + 0.2;
        double AG = Math.random() * 2 + 0.5;
        double result = (4.2 - 5 * WR + AG * 1.2 + (Math.random() - 0.5) * 0.2);

        return new double[]{WR, AG, result};
    }

    public static void main(String[] args) {
        Model m = new Model(LossFunction.MeanSquaredLoss, LEARNING_RATE);
        m.addLayer(new DenseLayer(3, 1, ActivationFunction.ReLU, Matrix.fromData(new double[][]{{1}, {1}, {1}})));

        LineChart chart = new LineChart();

        for (int i = 0; i < 1000; i++) {
            DataPoint[] dataPoints = new DataPoint[BATCH_SIZE];
            for(int j = 0; j < BATCH_SIZE; j++) {
                double[] testCase = __generate_test_case();

                Matrix inputs = Matrix.fromData(new double[][]{{1, testCase[0], testCase[1]}});
                Matrix outputs = Matrix.fromData(new double[][]{{testCase[2]}});
                DataPoint dataPoint = new DataPoint(inputs, outputs);
                dataPoints[j] = dataPoint;
            }

            double avgLoss = m.train(dataPoints);
            chart.addPoint(i, avgLoss, "Matrix");
        }
    }
}