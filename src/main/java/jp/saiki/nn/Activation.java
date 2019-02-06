package jp.saiki.nn;

public class Activation {

    public static double[] sigmoid(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = 1 / ( 1 + Math.pow(Math.E, -1 * input[i]));
        }
        return output;
    }

    public static double[] relu(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0d, input[i]);
        }
        return output;
    }

    public static double[] softmax(double[] input) {
        double[] output = new double[input.length];
        double sum = 0d;
        for (int i = 0; i < input.length; i++) {
            sum += Math.pow(Math.E, input[i]);
        }
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] / sum;
        }
        return output;
    }
 }