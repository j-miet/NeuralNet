namespace NeuralNet
{
    /// <summary>
    /// Neural network class
    /// </summary>
    class NN
    {
        private Layer[] layers;

        public NN(Layer[] layers)
        {
            this.layers = layers;
        }

        /// <summary>
        /// Forwards entire network
        /// </summary>
        /// <param name="inputs">Input layer data</param>
        /// <returns></returns>
        public Tensor Forward(Tensor inputs)
        {
            Tensor outputValues = inputs;
            foreach (Layer layer in layers)
            {
                outputValues = layer.Forward(outputValues);
            }
            return outputValues;
        }

        /// <summary>
        /// Backpropagates through all layers
        /// </summary>
        /// <param name="gradient">Loss function gradient</param>
        /// <returns>Gradient of first layer (if needed for some reason)</returns>
        public Tensor Backward(Tensor gradient)
        {
            Tensor gradValues = gradient;
            foreach(Layer layer in layers.Reverse())
            {
                gradValues = layer.Backward(gradValues);
            }
            return gradValues;
        }

        /// <summary>
        /// Returns an array of all layers
        /// </summary>
        /// <returns>Array of Layer objects</returns>
        public Layer[] GetLayers()
        {
            return layers;
        }
    }
    
}