package neuralnet.neuron;

import matrix.Matrix;

public interface LossFunction {
    /**
     * @param predicted A row vector
     * @param expected A row vector
     */
    double apply(Matrix predicted, Matrix expected);

    /**
     * @param predicted A row vector
     * @param expected A row vector
     */
    double applyDerivative(Matrix predicted, Matrix expected);

    LossFunction MeanSquaredLoss = new LossFunction() {
        @Override
        public double apply(Matrix predicted, Matrix expected) {
            Matrix residual = predicted.minus(expected);
            /*
             * <M, N> x <N, 1> -> <M, 1>
             * <1, M> x <M, 1> -> <1, 1>
             */

            return residual.times_elementwise(residual)
                    .times(Matrix.one(residual.columns, 1))
                    .transpose()
                    .times(Matrix.one(residual.rows, 1))
                    .at(0, 0);
        }

        public double applyDerivative(Matrix input) {

        }
    };
}
