package jp.saiki.nn;

import org.junit.Test;
import static org.junit.Assert.*;

import java.util.Arrays;

public class XorTest {

    double[][] input = new double[][] {
        new double[]{1d, 1d},
        new double[]{1d, 0d},
        new double[]{0d, 1d},
        new double[]{0d, 0d},
    };

    double[][] teacher = new double[][] {
        new double[]{0d},
        new double[]{1d},
        new double[]{1d},
        new double[]{0d},
    };

    @Test
    public void testTrainer() {
        Model model = new Sequential();
        model.addLayer(new Dense(2, 2));
        model.addLayer((double[] d) -> { return Activation.relu(d); });
        model.addLayer(new Dense(2, 1));
        Trainer trainer = new GeneticAlgorithmTrainer(model);
        model = trainer.train(input, teacher);
        assertNotNull("not null", model.predict(new double[]{1d, 1d}));
        for (double[] eachInput : input) {
            System.out.println(Arrays.toString(eachInput)+":"+Arrays.toString(model.predict(eachInput)));
        }
    }

    @Test
    public void testXor() {
        Model model = new Sequential();
        double[][] weightInput = new double[][] {
            new double[]{1d, 1d},
            new double[]{1d, 1d}
        };
        double[] biasInput = new double[]{0d, -1d};
        model.addLayer(new Dense(weightInput, biasInput));
        model.addLayer((double[] d) -> { return Activation.relu(d); });
        double[][] weightHidden = new double[][] {
            new double[] {1d, -2d}
        };
        double[] biasHidden = new double[]{0d};
        model.addLayer(new Dense(weightHidden, biasHidden));
        for (int i = 0; i < input.length; i++) {
            assertArrayEquals("計算結果が一致しない", teacher[i], model.predict(input[i]), 0d);
        }
    }
}
