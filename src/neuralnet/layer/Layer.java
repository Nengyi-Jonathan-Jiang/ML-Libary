package neuralnet.layer;

import matrix.Matrix;

public abstract class Layer {
    protected Layer next;

    public abstract void acceptInput(Matrix input);
    public abstract Matrix getOutput();
}
