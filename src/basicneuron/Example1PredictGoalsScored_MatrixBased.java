package basicneuron;

import chart.LineChart;
import chart.LineLogChart;
import matrix.Matrix;
import matrix.MatrixFactory;
import matrix.RowVector;
import neuralnet.Model;
import neuralnet.Model.DataPoint;
import neuralnet.layer.DenseLayer;
import neuralnet.neuron.ActivationFunction;
import neuralnet.neuron.LossFunction;

public class Example1PredictGoalsScored_MatrixBased {
    private static final double LEARNING_RATE = 0.05;
    private static final int BATCH_SIZE = 1000;

    private static double[] __generate_test_case() {
        double WR = Math.random() * 0.6 + 0.2;
        double AG = Math.random() * 2 + 0.5;
        double result = (4.2 - 5 * WR + AG * 1.2);

        return new double[]{WR, AG, result};
    }

    public static void main(String[] args) {
        Model m = new Model(LossFunction.MeanSquaredLoss, LEARNING_RATE);
        DenseLayer layer = new DenseLayer(3, 1, ActivationFunction.ReLU, MatrixFactory.columnVector(1, 1, 1));
        m.addLayer(layer);

        LineLogChart chart = new LineLogChart();
        LineChart weightsChart = new LineChart();

        for (int i = 0; i < 10000; i++) {
            Matrix inputs = MatrixFactory.matrix(BATCH_SIZE, 3);
            Matrix outputs = MatrixFactory.matrix(BATCH_SIZE, 1);
            for(int j = 0; j < BATCH_SIZE; j++) {
                double[] testCase = __generate_test_case();

                inputs.setElementAt(j, 0, 1);
                inputs.setElementAt(j, 1, testCase[0]);
                inputs.setElementAt(j, 2, testCase[1]);
                outputs.setElementAt(j, 0, testCase[2]);
            }

            double avgLoss = m.train(inputs, outputs);
            chart.addPoint(i, avgLoss, "Matrix", i % 100 == 0);
            weightsChart.addPoint(i, layer.getWeights().getElementAt(0, 0), "w1", i % 100 == 0);
            weightsChart.addPoint(i, layer.getWeights().getElementAt(1, 0), "w2", i % 100 == 0);
            weightsChart.addPoint(i, layer.getWeights().getElementAt(2, 0), "w3", i % 100 == 0);
        }
    }
}