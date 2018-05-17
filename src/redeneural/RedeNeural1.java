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
 * @author manuel
 */
public class RedeNeural1 {

    public static String input[][] = {{"cachorro", "animal"},
    {"gato", "animal"}, {"rodolfo", "animal"}, {"coelho", "animal"},
    {"lobo", "animal"}, {"mesa", "objeto"}, {"cadeira", "objeto"},
    {"canetao", "objeto"}, {"amor", "sentimento"},
    {"dor de cabeca", "sentimento"}, {"odio", "sentimento"},
    {"fome", "sentimento"}};

    public static double input_ascii[][];
    public static double output_ascii[][];
    public static int maior_palavra;

    public static void main(final String args[]) {

        maior_palavra = 0;
        for (int i = 0; i < input.length; i++) {
            if (input[i][0].length() > maior_palavra) {
                maior_palavra = input[i][0].length();
            }
        }

        input_ascii = new double[input.length][maior_palavra];
        output_ascii = new double[input.length][1];

        for (int i = 0; i < input.length; i++) {
            input_ascii[i] = toAscii(input[i][0], maior_palavra);
            output_ascii[i][0] = average(toAscii(input[i][1], maior_palavra));
        }
        

        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, maior_palavra));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
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
        } while (train.getError() > 0.0001);
        train.finishTraining();

        double tocni = 0;

        System.out.println("Neural Network Results:");
        for (MLDataPair pair : trainingSet) {
            final MLData output = network.compute(pair.getInput());

            System.out.println(pair.getInput().getData(0) + ", actual="
                    + output.getData(0) + " , "
                    + denormaliziraj(output.getData(0)) + " ,ideal="
                    + pair.getIdeal().getData(0) + " , "
                    + denormaliziraj(pair.getIdeal().getData(0)));

            if (denormaliziraj(output.getData(0)) == denormaliziraj(pair.getIdeal().getData(0))) {
                tocni++;
            }
        }

        System.out.println("Uspješnost: " + (tocni / output_ascii.length) * 100 + "%");

        Encog.getInstance().shutdown();

    }

    public static double average(double[] ascii) {
        double sum = 0;
        for (double val : ascii) {
            sum += val;
        }
        return sum / ascii.length;
    }

    public static double[] toAscii(String s, int najveci) {
        double[] ascii = new double[najveci];
        try {
            byte[] bytes = s.getBytes("US-ASCII");
            for (int i = 0; i < bytes.length; i++) {
                ascii[i] = 90.0 / bytes[i];
            }

        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        return ascii;
    }

    public static String denormaliziraj(double trenutna) {
        double najmanjaRazlika = 1;
        int indeks = 0;
        for (int i = 0; i < output_ascii.length; i++) {
            if (Math.abs(output_ascii[i][0] - trenutna) < najmanjaRazlika) {
                najmanjaRazlika = Math.abs(output_ascii[i][0] - trenutna);
                indeks = i;
            }
        }
        return input[indeks][1];
    }

}
