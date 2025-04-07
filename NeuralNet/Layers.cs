using System.Collections.Generic;
using System.Security.Cryptography;

namespace NeuralNet
{
    /// <summary>
    /// Linear layer
    /// </summary>
    class Linear
    {
        private readonly double[,] weights = new double[0,0];
        private readonly Tensor biases;
        private Tensor inputs;
        public Linear(int inputSize, int outputSize)
        {
            Random rnd = new();
            double[,] weights = new double[inputSize, outputSize];
            double[] biases = new double[outputSize];
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    weights[i,j] = rnd.NextDouble();
                    biases[j] = rnd.NextDouble();
                }
            }
            this.weights = weights;
            this.biases = new Tensor(biases);
            inputs = new Tensor([0]);
        }

        public double[,] GetWeights()
        {
            return weights;
        }
   
        /*
        public Tensor forward(Tensor inputs)
        {
            this.inputs = inputs;
            return inputs
        }
        */

        public Tensor backward(Tensor gradient)
        {
            return new Tensor([0]);
        }
    }


}