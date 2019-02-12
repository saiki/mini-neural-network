package jp.saiki.nn;

import java.util.Arrays;

public class Dense implements Layer {

    private double[][] weight;

    private double[] bias;

    public Dense(int input, int output) {
        this.weight = new double[output][input];
        for (int i = 0; i < output; i++) {
            this.weight[i] = Utils.random(input);
        }
        this.bias = Utils.random(output);
    }

    public Dense(double[][] initWeight, double[] bias) {
        this.weight = initWeight;
        this.bias = bias;
    }

    @Override
    public double[] forword(double[] input) {
        double[] output = new double[this.weight[0].length];
        Arrays.fill(output, 0d);
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < input.length; j++) {
                output[i] += input[j] * this.weight[i][j];
            }
            output[i] += this.bias[i];
        }
        return output;
    }

    @Override
    public double[][] getWeight() {
        return this.weight;
    }

    @Override
    public double[] getBias() {
        return this.bias;
    }

    @Override
    public void update(double[][] newWeight, double[] newBias) {
        this.weight = newWeight;
        this.bias = newBias;
    }
}