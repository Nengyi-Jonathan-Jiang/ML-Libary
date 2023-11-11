package neuralnet.neuron;

public interface ActivationFunction {
    double apply(double input);
    double applyDerivative(double input);

    ActivationFunction ReLU = new ActivationFunction() {
        @Override
        public double apply(double input) {
            return Math.max(input, 0);
        }

        public double applyDerivative(double input) {
            return input > 0 ? 1 : 0;
        }
    };
    ActivationFunction Sigmoid = new ActivationFunction() {
        @Override
        public double apply(double input) {
            return 1 / (1 + Math.exp(-input));
        }

        @Override
        public double applyDerivative(double input) {
            double s = apply(input);
            return s * (1 - s);
        }
    };
    ActivationFunction Tanh = new ActivationFunction() {
        @Override
        public double apply(double x) {
            return Math.tanh(x);
        }

        @Override
        public double applyDerivative(double input) {
            double t = apply(input);
            return 1 - t * t;
        }
    };
    ActivationFunction Softplus = new ActivationFunction() {
        @Override
        public double apply(double input) {
            return Math.log(1 + Math.exp(input));
        }

        @Override
        public double applyDerivative(double input) {
            return 1 / (1 + Math.exp(input));
        }
    };
}
