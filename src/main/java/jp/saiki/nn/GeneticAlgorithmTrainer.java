package jp.saiki.nn;

import java.util.Arrays;
import java.util.Random;

public class GeneticAlgorithmTrainer implements Trainer {

    public static final int DEFAULT_EPOCH = 1000;

    private static final int EACH_DEFAULT_SIZE = 10;

    private final Model original;

    private final int epoch;

    private final int selectSize;

    private final int crossoverSize;

    private final int mutationSize;

    private Random random = new Random();

    public GeneticAlgorithmTrainer(Model model){
        this.original = model;
        this.epoch = DEFAULT_EPOCH;
        this.selectSize = EACH_DEFAULT_SIZE;
        this.crossoverSize = EACH_DEFAULT_SIZE;
        this.mutationSize = EACH_DEFAULT_SIZE;
    }

    public GeneticAlgorithmTrainer(Model model, int epoch){
        this.original = model;
        this.epoch = epoch;
        this.selectSize = EACH_DEFAULT_SIZE;
        this.crossoverSize = EACH_DEFAULT_SIZE;
        this.mutationSize = EACH_DEFAULT_SIZE;
    }

    public GeneticAlgorithmTrainer(Model model, int epoch, int selectSize, int crossoverSize, int mutationSize){
        this.original = model;
        this.epoch = epoch;
        this.selectSize = selectSize;
        this.crossoverSize = crossoverSize;
        this.mutationSize = mutationSize;
    }

    public Model train(double[][] data, double[][] teacher) {
        Model[] models = new Model[this.selectSize+this.crossoverSize+this.mutationSize];
        for (int i = 0; i < models.length; i++) {
            models[i] = mutation(this.original).clone();
        }
        for (int e = 0; e < epoch; e++) {
            double[] loss = new double[models.length];
            Arrays.fill(loss, 0d);
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < models.length; j++) {
                    double[] result = models[j].predict(data[i]);
                    loss[j] += Loss.meanSquaredError(result, teacher[i]);
                }
            }
            for (int i = 0; i < loss.length; i++) {
                loss[i] = loss[i] / data.length;
            }
            System.out.println(minLossIndex(loss)+":"+loss[minLossIndex(loss)]);
            Model[] newModels = new Model[this.selectSize+this.crossoverSize+this.mutationSize];
            Model[] select = select(models, loss);
            System.arraycopy(select, 0, newModels, 0, select.length);
            Model[] crossover = crossover(select);
            System.arraycopy(crossover, 0, newModels, select.length, select.length);
            Model[] mutation = new Model[this.mutationSize];
            for (int i = 0; i < mutation.length; i++) {
                mutation[i] = mutation(this.original).clone();
            }
            System.arraycopy(mutation, 0, newModels, select.length+crossover.length, mutation.length);
            models = newModels;
        }
        return models[0];
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

    private Model[] select(Model[] models, double[] loss) {
        Model[] result = new Model[selectSize];
        for (int i = 0; i < result.length; i++) {
            int minLossIndex = minLossIndex(loss);
            result[i] = models[minLossIndex];
            Model[] nextModels = new Model[models.length-1];
            if (minLossIndex > 0) {
                System.arraycopy(models, 0, nextModels, 0, minLossIndex);
            }
            if (minLossIndex < models.length-1) {
                System.arraycopy(models, minLossIndex+1, nextModels, minLossIndex, loss.length-minLossIndex-1);
            }
            double[] nextLoss = new double[loss.length-1];
            if (minLossIndex > 0) {
                System.arraycopy(loss, 0, nextLoss, 0, minLossIndex);
            }
            if (minLossIndex < models.length-1) {
                System.arraycopy(loss, minLossIndex+1, nextLoss, minLossIndex, loss.length-minLossIndex-1);
            }
            models = nextModels;
            loss = nextLoss;
        }
        return result;
    }

    private Model[] crossover(Model[] models) {
        Model[] result = new Model[this.crossoverSize];
        for (int i = 0; i < result.length; i++) {
            result[i] = crossover(models[this.random.nextInt(models.length)], models[this.random.nextInt(models.length)]);
        }
        return result;
    }

    private Model crossover(Model to, Model from) {
        for (int i = 0; i < to.getLayers().size(); i++) {
            if (to.getLayers().get(i).getWeight() == null) {
                continue;
            }
            double[][] mergedWeight = to.getLayers().get(i).getWeight();
            for (int j = 0; j < mergedWeight.length; j++) {
                int start = random.nextInt(mergedWeight[j].length);
                double[] weight = from.getLayers().get(i).getWeight()[j];
                System.arraycopy(weight, start, mergedWeight[j], start, weight.length-start);
            }
            double[] mergedBias = to.getLayers().get(i).getBias();
            int start = random.nextInt(mergedBias.length);
            double[] bias = from.getLayers().get(i).getBias();
            System.arraycopy(bias, start, mergedBias, start, bias.length-start);
            to.getLayers().get(i).update(mergedWeight, mergedBias);
        }
        return to.clone();
    }

    private Model mutation(final Model model) {
        for (Layer layer : model.getLayers()) {
            if (layer.getWeight() == null) {
                continue;
            }
            double[][] weights = layer.getWeight();
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] = random.nextDouble();
                }
            }
            double[] bias = layer.getBias();
            for (int i = 0; i < bias.length; i++) {
                bias[i] = random.nextDouble();
            }
            layer.update(weights, bias);
        }
        return model.clone();
    }
 }
