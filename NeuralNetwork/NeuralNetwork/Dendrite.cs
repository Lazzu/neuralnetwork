using System;

namespace NeuralNetwork
{
    public class Dendrite
    {
        public double Weight { get; }

        public Dendrite(Random random)
        {
            Weight = random.NextDouble();
        }

        public Dendrite(double weight)
        {
            Weight = weight;
        }
    }
}