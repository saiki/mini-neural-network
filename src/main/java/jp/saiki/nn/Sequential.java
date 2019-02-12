package jp.saiki.nn;

import java.util.ArrayList;
import java.util.List;

public class Sequential implements Model {
    
    private final List<Layer> layers = new ArrayList<>();

    public Sequential() {
    }

    @Override
    public List<Layer> getLayers() {
        return layers;
    }

    @Override
    public void addLayer(Layer layer) {
        this.layers.add(layer);
    }

    @Override
    public double[] predict(double[] value) {
        double[] out = value;
        for (Layer layer : layers) {
            out = layer.forword(out);
        }
        return out;
    }
}