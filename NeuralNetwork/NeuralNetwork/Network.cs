using System;

namespace NeuralNetwork
{
    public class Network : IComparable<Network>
    {
        public Layer[] Layers { get; }
        public double Fitness { get; set; }

        public Network(Random random, int inputs, int outputs, int[] layers, Func<double, double> activationFunction)
        {
            Layers = new Layer[layers.Length + 2];
            Layers[0] = new Layer(random, inputs, layers[0], activationFunction);
            for (int i = 1; i <= layers.Length; i++)
            {
                int currentLayerIndex = i - 1;
                int dendrites;
                int nextLayerIndex = i;
                if (nextLayerIndex < layers.Length)
                {
                    dendrites = layers[nextLayerIndex];
                }
                else
                {
                    dendrites = outputs;
                }
                Layers[i] = new Layer(random, layers[currentLayerIndex], dendrites, activationFunction);
            }
            Layers[^1] = new Layer(random, outputs, 0, activationFunction);
        }

        public Network(Layer[] layers)
        {
            Layers = layers;
        }

        public void Forward(double[] input, double[] output)
        {
            Layers[0].SetValues(input);
            
            for (int index = 0; index < Layers.Length - 1; index++)
            {
                Layer layer = Layers[index];
                Layer nextLayer = Layers[index + 1];
                layer.Forward(nextLayer);
            }

            Layers[^1].GetValues(output);
        }

        public void Mutate(Random random, Func<Random, double, double> mutationFunc)
        {
            foreach (Layer layer in Layers)
            {
                layer.Mutate(random, mutationFunc);
            }
        }

        public Network CloneNetwork()
        {
            Layer[] layers = new Layer[Layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = Layers[i].Clone();
            }
            return new Network(layers);
        }

        public int CompareTo(Network other)
        {
            if (ReferenceEquals(this, other)) return 0;
            if (ReferenceEquals(null, other)) return 1;
            return -Fitness.CompareTo(other.Fitness);
        }
    }
}