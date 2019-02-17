package jp.saiki.nn;

import java.io.Serializable;
import java.util.List;

public interface Model extends Serializable, Cloneable {
    
    public List<Layer> getLayers();

    public void addLayer(Layer layer);

    public double[] predict(double[] value);

    public Model clone();

}