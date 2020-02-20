using System;

namespace NeuralNetwork
{
    public class Neuron
    {
        public double[] Dendrites { get; }
        public double Bias { get; private set; }
        public double Value { get; set; }

        public Neuron(Random random, int dendrites)
        {
            Bias = random.NextDouble();
            Dendrites = new double[dendrites];
            for (int i = 0; i < dendrites; i++)
            {
                Dendrites[i] = random.NextDouble();
            }
        }

        public Neuron(double[] dendrites)
        {
            Dendrites = dendrites;
        }

        public void Mutate(Random random, Func<Random,double,double> mutationFunc)
        {
            Bias = mutationFunc(random, Bias);
            for (int i = 0; i < Dendrites.Length; i++)
            {
                Dendrites[i] = mutationFunc(random, Dendrites[i]);
            }
        }

        public Neuron Clone()
        {
            double[] dendrites = new double[Dendrites.Length];
            for (int i = 0; i < dendrites.Length; i++)
            {
                dendrites[i] = Dendrites[i];
            }
            return new Neuron(dendrites);
        }
    }
}