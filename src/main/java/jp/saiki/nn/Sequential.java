package jp.saiki.nn;

import java.util.ArrayList;
import java.util.List;

public class Sequential implements Model {
    
    private static final long serialVersionUID = 2767731419815147667L;

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

    @Override
    public Sequential clone() {
        try {
            Sequential m = (Sequential)super.clone();
            for (int i = 0; i < m.layers.size(); i++) {
                m.getLayers().get(i).update(this.layers.get(i).getWeight(), this.layers.get(i).getBias());
            }
            return m;
        } catch (CloneNotSupportedException ex) {
            throw new InternalError(ex);
        }
    }
}