package jp.saiki.nn;

import java.io.Serializable;

@FunctionalInterface
public interface Layer extends Serializable, Cloneable {

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