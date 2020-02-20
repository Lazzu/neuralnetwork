using System;
using System.Linq;

namespace NeuralNetwork
{
    public class Layer
    {
        public Neuron[] Neurons { get; }
        
        private Func<double, double> _activationFunction;

        public Layer(Random random, int neurons, int dendrites, Func<double, double> activationFunction)
        {
            _activationFunction = activationFunction;
            Neurons = new Neuron[neurons];
            for (int i = 0; i < neurons; i++)
            {
                Neurons[i] = new Neuron(random, dendrites);
            }
        }

        public Layer(Neuron[] neurons, Func<double, double> activationFunction)
        {
            _activationFunction = activationFunction;
            Neurons = neurons;
        }

        public void SetValues(double[] input)
        {
            for (int index = 0; index < Neurons.Length; index++)
            {
                Neurons[index].Value = input[index];
            }
        }

        public void Forward(Layer layer)
        {
            for (int index = 0; index < layer.Neurons.Length; index++)
            {
                Neuron nextNeuron = layer.Neurons[index];
                double weightedBiasedSums = Neurons.Sum(currentNeuron =>
                    currentNeuron.Value * currentNeuron.Dendrites[index]);
                nextNeuron.Value = _activationFunction(weightedBiasedSums + nextNeuron.Bias);
            }
        }

        public void GetValues(double[] output)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                output[i] = Neurons[i].Value;
            }
        }

        public void Mutate(Random random, Func<Random,double,double> mutationFunc)
        {
            foreach (Neuron neuron in Neurons)
            {
                neuron.Mutate(random, mutationFunc);
            }
        }

        public Layer Clone()
        {
            Neuron[] neurons = new Neuron[Neurons.Length];
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i] = Neurons[i].Clone();
            }
            return new Layer(neurons, _activationFunction);
        }
    }
}