package jp.saiki.nn;

import java.util.Random;

public class Utils {

    public  static double[] random(int size) {
        Random random = new Random();
        double[] output = new double[size];
        for (int i = 0; i < size; i++) {
            output[i] = random.nextDouble();
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