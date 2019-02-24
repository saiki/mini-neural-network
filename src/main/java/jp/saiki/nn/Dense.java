package jp.saiki.nn;

import java.util.Arrays;

public class Dense implements Layer {

	private static final long serialVersionUID = -3789067412634838790L;

    private final int inputSize;

	private final int outputSize;

    private double[][] weight;

    private double[] bias;

    public Dense(int input, int output) {
		this.inputSize = input;
		this.outputSize = output;
        this.weight = new double[input][output];
        for (int i = 0; i < input; i++) {
            this.weight[i] = Utils.random(output);
        }
        this.bias = Utils.random(output);
    }

    public Dense(double[][] initWeight, double[] bias) {
        this.weight = initWeight;
		this.inputSize = initWeight.length;
		this.outputSize = initWeight[0].length;
        this.bias = bias;
    }

    @Override
    public double[] forword(double[] input) {
        double[] output = new double[this.outputSize];
        Arrays.fill(output, 0d);
        for (int i = 0; i < this.outputSize; i++) {
            for (int j = 0; j < this.inputSize; j++) {
                output[i] += input[j] * this.weight[j][i];
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

    @Override
    public Dense clone() {
        double[][] weight = new double[this.weight.length][this.weight[0].length];
        double[] bias = new double[this.bias.length];
        System.arraycopy(this.weight, 0, weight, 0, this.weight.length);
        System.arraycopy(this.bias, 0, bias, 0, this.bias.length);
        return new Dense(weight, bias);
    }
}
