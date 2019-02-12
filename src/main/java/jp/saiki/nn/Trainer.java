package jp.saiki.nn;

public interface Trainer {

    public void train(double[][] data, double[][] teacher);

}