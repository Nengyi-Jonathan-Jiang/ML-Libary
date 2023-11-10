package neuralnet.util;

import java.util.Random;

public final class MathUtil {
    private static final Random random = new Random();

    private MathUtil() {}

    public static double randUniform() {
        return random.nextDouble();
    }

    public static double randGaussian() {
        return random.nextGaussian();
    }
}
