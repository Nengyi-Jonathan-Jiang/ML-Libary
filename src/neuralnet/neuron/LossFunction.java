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

    LossFunction MeanSquaredLoss = new LossFunction() {
        @Override
        public double apply(RowVector predicted, RowVector expected) {
            RowVector residual = (RowVector) predicted.minus(expected);
            double sumSquaredResiduals = residual.dot(residual);
            return sumSquaredResiduals / residual.columns();
        }

        @Override
        public RowVector applyGradient(RowVector predicted, RowVector expected, RowVector out) {
            predicted.subtract_to(expected, out);
            return (RowVector) out.multiply_to(2, out);
        }
    };

    LossFunction LogLoss = new LossFunction() {
        private double logLoss(double predicted, double expected) {
            return expected == 0 ? -Math.log(predicted) : -Math.log(1 - predicted);
        }

        private double logLossDerivative(double predicted, double expected) {
            return expected == 0 ? -1 / predicted : 1 / (1 - predicted);
        }

        @Override
        public double apply(RowVector predicted, RowVector expected) {
            double totalLoss = Matrix.applyOperationAndSum(predicted, expected, this::logLoss);
            return totalLoss / predicted.columns();
        }

        @Override
        public RowVector applyGradient(RowVector predicted, RowVector expected, RowVector out) {
            Matrix.applyOperation(predicted, expected, this::logLossDerivative, out);
            return (RowVector) out.times(1. / out.columns());
        }
    };
}
