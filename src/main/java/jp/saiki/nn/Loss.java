package jp.saiki.nn;

public class Loss {

    public static double meanSquaredError(double[] input, double[] teacher) {
        double sum = 0d;
        for (int i = 0; i < input.length; i++) {
            sum += Math.pow(input[i] - teacher[i], 2);
        }
        return sum * 0.5d;
    }
}