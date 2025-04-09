namespace NeuralNet
{
    /// <summary>
    /// General class for optimizers
    /// </summary>
    abstract class Optimizer
    {
        protected NN network;

        public Optimizer(NN network)
        {
            this.network = network;
        }

        public abstract void Step(NN network);
    }

    /// <summary>
    /// Stochastic gradient descent optimizer
    /// </summary>
    class SGD : Optimizer
    {
        private double lr;

        public SGD(NN network, double lr = 0.01)
            : base(network)
        {
            this.lr = lr;
        }

        /// <summary>
        /// Performs a single optimization step for network
        /// </summary>
        /// <param name="network">Neural network</param>
        public override void Step(NN network)
        {
            foreach (Layer layer in network.GetLayers())
            {   
                Tensor newWeights = 
                Tensor.SubtractTensors(layer.GetWeights(), Tensor.ScaleTensor(layer.GetGradWeights(), lr));
                layer.SetWeights(newWeights);
                Tensor newBiases =
                Tensor.SubtractTensors(layer.GetBiases(), Tensor.ScaleTensor(layer.GetGradBiases(), lr));
                layer.SetBiases(newBiases);           
            }
        }
    }
}
