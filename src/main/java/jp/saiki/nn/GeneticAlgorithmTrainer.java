package jp.saiki.nn;

import java.util.Arrays;
import java.util.Random;

public class GeneticAlgorithmTrainer implements Trainer {

    public static final int DEFAULT_EPOCH = 10;

    private Model model = null;

    private int epoch = DEFAULT_EPOCH;

    private Random random = new Random();

    public GeneticAlgorithmTrainer(Model model){
        this.model = model;
    }

    public GeneticAlgorithmTrainer(Model model, int epoch){
        this.model = model;
        this.epoch = epoch;
    }

    public void train(double[][] data, double[][] teacher) {
        Model[] models = new Model[3];
        for (int e = 0; e < epoch; e++) {
            double[] loss = new double[]{0, 0, 0};
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < models.length; j++) {
                    double[] result = models[j].predict(data[i]);
                    loss[j] += Loss.meanSquaredError(result, teacher[i]);
                }
            }
            for (int i = 0; i < loss.length; i++) {
                loss[i] = loss[i] / data.length;
            }
        }
    }

    private Model select(Model[] models, double[] loss) {
        return models[minLossIndex(loss)];
    }

    private int minLossIndex(double[] loss) {
        int minIndex = 0;
        for (int i = 0; i < loss.length; i++) {
            if (loss[i] < loss[minIndex]) {
                minIndex = i;
            }
        }
        return minIndex;
    }

    private Model crossover(Model[] models, double[] loss) {
        Model result = models[random.nextInt(models.length)];
        Model second = models[random.nextInt(models.length)];
        for (int i = 0; i < result.getLayers().size(); i++) {
            if (result.getLayers().get(i).getWeight() == null) {
                continue;
            }
            double[][] mergedWeight = result.getLayers().get(i).getWeight();
            for (int j = 0; j < mergedWeight.length; j++) {
                int start = random.nextInt(mergedWeight.length);
                double[] weight = second.getLayers().get(i).getWeight()[j];
                for (int x = start; x < weight.length; x++) {
                    mergedWeight[j][x] = weight[x];
                }
            }
            double[] mergedBias = result.getLayers().get(i).getBias();
            int start = random.nextInt(mergedWeight.length);
            double[] bias = second.getLayers().get(i).getWeight()[j];
            for (int x = start; x < bias.length; x++) {
                mergedBias[x] = bias[x];
            }
            result.getLayers().get(i).update(mergedWeight, mergedBias);
        }
        return result;
    }

    private Model mutation(Model[] models) {
        Model model = models[random.nextInt(models.length)];
        for (Layer layer : model.getLayers()) {
            if (layer.getWeight() == null) {
                continue;
            }
            double[][] weights = layer.getWeight();
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][i] = random.nextDouble();
                }
            }
            double[] bias = layer.getBias();
            for (int i = 0; i < bias.length; i++) {
                bias[i] = random.nextDouble();
            }
            layer.update(weights, bias);
        }
        return model;
    }

    public Model getModel() {
        return this.model;
    }
 }