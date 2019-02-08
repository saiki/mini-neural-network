package jp.saiki.nn;

@FunctionalInterface
public interface Layer {

    public default double[][] getWeight() {
        return null;
    }

    public default double[] getBias() {
        return null;
    }

    public default void update(double[][] newWeight, double[] newBias) {
        // nop
    }

    public double[] forword(double[] input);
    
}