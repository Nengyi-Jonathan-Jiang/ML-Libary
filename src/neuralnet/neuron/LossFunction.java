package neuralnet.neuron;

import matrix.Matrix;
import matrix.MatrixFactory;
import matrix.RowVector;

public interface LossFunction {
    /**
     * @param predicted A row vector
     * @param expected  A row vector
     */
    double apply(RowVector predicted, RowVector expected);

    RowVector applyGradient(RowVector predicted, RowVector expected, RowVector out);

    default Matrix applyGradient(RowVector predicted, RowVector expected) {
        return applyGradient(predicted, expected, MatrixFactory.rowVector(predicted.size()));
    }

    interface SumLossFunction extends LossFunction {
        double loss(double predicted, double expected);
        double derivative(double predicted, double expected);

        @Override
        default double apply(RowVector predicted, RowVector expected) {
            double totalLoss = Matrix.applyOperationAndSum(predicted, expected, this::loss);
            return totalLoss / predicted.columns();
        }

        @Override
        default RowVector applyGradient(RowVector predicted, RowVector expected, RowVector out) {
            Matrix.applyOperation(predicted, expected, (predicted1, expected1) -> derivative(predicted1, expected1) / out.columns(), out);
            return out;
        }
    }

    LossFunction MeanSquaredLoss = new SumLossFunction() {
        public double loss(double predicted, double expected) {
            return (predicted - expected) * (predicted - expected) / 2;
        }

        public double derivative(double predicted, double expected) {
            return predicted - expected;
        }
    };

    LossFunction LogLoss = new SumLossFunction() {
        public double loss(double predicted, double expected) {
            return expected == 0 ? -Math.log(predicted) : -Math.log(1 - predicted);
        }

        public double derivative(double predicted, double expected) {
            return expected == 0 ? -1 / predicted : 1 / (1 - predicted);
        }
    };
}
