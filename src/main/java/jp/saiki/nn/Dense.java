package jp.saiki.nn;

public class Dense implements Layer {

    private double[] weight;

    private double[] bias;

    public Dense(double[] initWeight, double[] bias) {
        this.weight = initWeight;
        this.bias = bias;
    }

    public double[] forword(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] * weight[i] + bias[i];
        }
        return output;
    }

    public void update(double[] newWeight) {
        this.weight = newWeight;
    }
}