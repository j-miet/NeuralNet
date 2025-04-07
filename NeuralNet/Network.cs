namespace NeuralNet
{
    
    class NN
    {
        private readonly int inCount;
        private readonly int outCount;
        private readonly int layers;

        public NN(int inCount, int outCount, int layers)
        {
            this.inCount = inCount;
            this.outCount = outCount;
            this.layers = layers;
        }
    }
}

