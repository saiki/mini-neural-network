package jp.saiki.nn;

public class Utils {

    public  static double[] random(int size) {
        double[] output = new double[size];
        for (int i = 0; i < size; i++) {
            output[i] = Math.random();
        }
        return output;
    }

    public static double[] vector(int size, double fill) {
        double[] output = new double[size];
        for (int i = 0; i < size; i++) {
            output[i] = fill;
        }
        return output;
    }

}