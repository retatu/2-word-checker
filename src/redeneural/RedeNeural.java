package redeneural;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationElliott;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

/**
 *
 * @author manuel
 */
public class RedeNeural {

    /**
     * @param args the command line arguments
     */
    //conjunto de entradas para o treinamento	 
    public static double input1[][] = {{132456789/100000000, 123456789/100000000}, {134256789/100000000, 123456789/100000000}, {432165789/100000000, 123456789/100000000}, {432516798/100000000, 123456789/100000000}, {129876543/100000000, 123456789/100000000}, {132547689/100000000, 123456789/100000000}, {927456381/100000000, 123456789/100000000}, {123456789/100000000, 123456789/100000000}};
    //conjunto de saídas para o trinamento
    public static double ideal1[][] = {{2/100}, {4/100}, {6/100}, {10/100}, {24/100}, {6/100}, {24/100}, {0}};

    public static void main(String[] args) {

        double input[][] = new double[100][2];
        double ideal[][] = new double[100][1];

//        for (int i = 1; i < 100; i++) {
//            input[i][0] = i;
//            input[i][1] = i * 2;
//            ideal[i][0] = ((input[i][0] + input[i][1]) / 2);
//            System.out.println("0: " + i + " 1: " + (i * 2) + " final" + ideal[i][0]);
//        }
        // cria a rede neural
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, 2));
        network.addLayer(new BasicLayer(new ActivationLinear(), true, 3));
        network.addLayer(new BasicLayer(new ActivationLinear(), false, 1));
        network.getStructure().finalizeStructure();
        network.reset();

        // cria os dados de treinamento
        //MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
        MLDataSet trainingSet = new BasicMLDataSet(input1, ideal1);
        // cria o treinamento
        final ResilientPropagation train = new ResilientPropagation(network, trainingSet);
        //final Backpropagation train = new Backpropagation(network, trainingSet);

        int epoca = 1;
        //faz o treinamento
        do {
            train.iteration();
            System.out.println("Época #" + epoca + " Error:" + train.getError());
            epoca++;
        } while (train.getError() > 0.001);
        train.finishTraining();

        // testa a rede neural
        System.out.println("Resultados:");
        for (MLDataPair pair : trainingSet) {
            final MLData output = network.compute(pair.getInput());
            System.out.println(pair.getInput().getData(0) + "," + pair.getInput().getData(1)
                    + ", real=" + output.getData(0) + ",ideal=" + pair.getIdeal().getData(0));
        }
        double[] entrada = {0.0, 0.0};
        double[] saida = {0.0};

        network.compute(entrada, saida);
        System.out.println("xor =" + String.valueOf(saida[0]));

        Encog.getInstance().shutdown();

    }

}
