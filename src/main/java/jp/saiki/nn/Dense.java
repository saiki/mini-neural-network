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
        this.weight = new double[output][input];
        for (int i = 0; i < output; i++) {
            this.weight[i] = Utils.random(input);
        }
        this.bias = Utils.random(output);
    }

    public Dense(double[][] initWeight, double[] bias) {
        this.weight = initWeight;
		this.inputSize = initWeight.length;
        this.bias = bias;
		this.outputSize = bias.length;
    }

    @Override
    public double[] forword(double[] input) {
        double[] output = new double[this.outputSize];
        Arrays.fill(output, 0d);
        for (int i = 0; i < this.weight.length; i++) {
            for (int j = 0; j < this.weight[i].length; j++) {
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

    @Override
    public Dense clone() {
        double[][] weight = new double[this.weight.length][this.weight[0].length];
        for (int i = 0; i < weight.length; i++) {
            System.arraycopy(this.weight[i], 0, weight[i], 0, weight[i].length);
        }
        double[] bias = new double[this.bias.length];
        System.arraycopy(this.bias, 0, bias, 0, this.bias.length);
        return new Dense(weight, bias);
    }
}
