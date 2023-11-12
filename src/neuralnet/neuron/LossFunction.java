package neuralnet.neuron;

import matrix.Matrix;

public interface LossFunction {
    /**
     * @param predicted A row vector
     * @param expected  A row vector
     */
    double apply(Matrix predicted, Matrix expected);

    /**
     * @param predicted A row vector
     * @param expected  A row vector
     * @return A row vector
     */
    Matrix applyGradient(Matrix predicted, Matrix expected, Matrix out);

    default Matrix applyGradient(Matrix predicted, Matrix expected) {
        return applyGradient(predicted, expected, Matrix.create(predicted.rows(), predicted.columns()));
    }


    LossFunction MeanSquaredLoss = new LossFunction() {
        @Override
        public double apply(Matrix predicted, Matrix expected) {
            Matrix residual = predicted.minus(expected);
            double sumSquaredResiduals = residual.times(residual.transpose()).at(0, 0);
            return sumSquaredResiduals / (2 * residual.columns());
        }

        @Override
        public Matrix applyGradient(Matrix predicted, Matrix expected, Matrix out) {
            Matrix residual = Matrix.subtract(predicted, expected, out);
            return Matrix.multiply(residual, 2, out);
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
        public double apply(Matrix predicted, Matrix expected) {
            double totalLoss = Matrix.applyOperationAndSum(predicted, expected, this::logLoss);
            return totalLoss / predicted.columns();
        }

        @Override
        public Matrix applyGradient(Matrix predicted, Matrix expected, Matrix out) {
            Matrix.applyOperation(predicted, expected, this::logLossDerivative, out);
            return out.times(1. / out.columns());
        }
    };
}
