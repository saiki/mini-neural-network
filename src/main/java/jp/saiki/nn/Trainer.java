package jp.saiki.nn;

public interface Trainer {

    public Model train(double[][] data, double[][] teacher);

}