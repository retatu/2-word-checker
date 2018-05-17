package redeneural;

import java.io.UnsupportedEncodingException;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

/**
 *
 * @author micro
 */
public class RedeNeural11 {

    public static String input[][] = {{"123 casa da dona", "casa da dona"},
    {"gato", "gatoo"}, {"gato", "gatu"},  {"coelho 123", "animal"},
    {"alooo", "alooo1"}, {"mesa", "objeto quadrado"}, {"pai1", "1pai"},
    {"canetao 4", "objeto"}, {"comida", "acomida"},
    {"loucura", "eloucuraa"}, {"odio", "odio"},
    {"fome", "sentimento"},{"fome", "sentimento"}};
    public static String output[][] = {{"true"},
    {"true"},{"true"}, {"false"},
    {"true"}, {"false"}, {"true"},
    {"false"}, {"true"},
    {"true"}, {"true"},
    {"false"}, {"false"}};

    public static double input_ascii[][];
    public static double output_ascii[][];
    public static int maior_palavra;

    public static void main(final String args[]) {

        maior_palavra = 0;
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[i].length; j++) {
                if (input[i][j].length() > maior_palavra) {
                    maior_palavra = input[i][j].length();
                }
            }
        }
 
        input_ascii = new double[input.length][maior_palavra];
        output_ascii = new double[input.length][1];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < 2 ; j++) {
                input_ascii[i][j] = toAscii(input[i][j]);
                System.out.println("input ascII i: "+i+" j:"+j+" is:"+input_ascii[i][j]
                        +" The real worl is: "+input[i][j]);
            }
            System.out.println("The previous worlds are equals? No. But the means are equals? "+output[i][0]);
        }
        System.out.println("Output: ");
        for (int i = 0; i < output.length; i++) {
            //System.out.println(output[i][0]);
            if(output[i][0] == "false"){
                output_ascii[i][0] = 0.0;
            }else{
                output_ascii[i][0] = 1.0;
            }
            //output_ascii[i][0] = average(toAscii(output[i][0], 5));
            //System.out.println(output_ascii[i][0]);
        }

        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, maior_palavra));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 5));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        network.getStructure().finalizeStructure();
        network.reset();

        MLDataSet trainingSet = new BasicMLDataSet(input_ascii, output_ascii);

        final ResilientPropagation train = new ResilientPropagation(network, trainingSet);

        int epoca = 1;

        do {
            train.iteration();
            System.out.println("Época " + epoca + " Error:" + train.getError());
            epoca++;
        } while (train.getError() > 0.05);//normalmente encontra com esse erro, mas as vezes não por causo das conexões entre os neurônios
        train.finishTraining();

        System.out.println("Neural Network Results:");
        for (MLDataPair pair : trainingSet) {
            final MLData output = network.compute(pair.getInput());

            System.out.println(pair.getInput().getData(0)
                    + ", " + pair.getInput().getData(1)
                    + ", actual="
                    + output.getData(0) + ", "
                    + pair.getIdeal().getData(0) + " , ");

        }
        Encog.getInstance().shutdown();

    }

    //Converte para ascII
    public static double toAscii(String world) {
        double ascii = 0;
        try {
            byte[] bytes = world.getBytes("US-ASCII");
            for (int i = 0; i < bytes.length; i++) {
                ascii += 100.0 / bytes[i];
                //System.out.print(ascii[i]);
            }
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }

        return ascii;
    }
}
