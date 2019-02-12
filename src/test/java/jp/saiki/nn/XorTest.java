package jp.saiki.nn;

import org.junit.Test;
import static org.junit.Assert.*;

public class XorTest {

    @Test
    public void testXOr() {
        double[][] input = new double[][] {
            new double[]{1d, 1d},
            new double[]{1d, 0d},
            new double[]{0d, 1d},
            new double[]{0d, 0d},
        };
        double[][] teacher = new double[][] {
            new double[]{0d, 1d},
            new double[]{1d, 0d},
            new double[]{1d, 0d},
            new double[]{0d, 1d},
        };
        Model model = new Sequential();
        model.addLayer(new Dense(2, 2));
        model.addLayer((double[] d) -> { return Activation.relu(d); });
        model.addLayer(new Dense(2, 2));
        model.addLayer((double[] d) -> { return Activation.softmax(d); });
        Trainer trainer = new GeneticAlgorithmTrainer(model);
        model = trainer.train(input, teacher);
        assertNotNull("not null", model.predict(new double[]{1d, 1d}));
    }
}
