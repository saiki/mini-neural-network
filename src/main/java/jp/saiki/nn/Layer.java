package jp.saiki.nn;

@FunctionalInterface
public interface Layer {

    public double[] forword(double[] input);
    
}