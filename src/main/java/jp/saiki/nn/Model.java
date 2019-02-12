package jp.saiki.nn;

import java.util.List;

public interface Model {
    
    public List<Layer> getLayers();

    public void addLayer(Layer layer);

    public double[] predict(double[] value);

}