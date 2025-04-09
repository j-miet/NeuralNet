namespace NeuralNet
{
    /// <summary>
    /// General class for network layers
    /// </summary>
    abstract class Layer
    {
        protected Tensor paramWeights;
        protected Tensor paramBiases;
        protected Tensor gradWeights;
        protected Tensor gradBiases;
        protected Tensor inputs;

        public Layer()
        {
            paramWeights = new Tensor(new double[,]{{0}, {0}});
            paramBiases = new Tensor(new double[,]{{0}, {0}});
            gradWeights = new Tensor(new double[,]{{0}, {0}});
            gradBiases = new Tensor(new double[,]{{0}, {0}});
            inputs = new Tensor(new double[,]{{0}, {0}});
        }

        public void SetWeights(Tensor weights)
        {
            paramWeights = weights;
        }

        public Tensor GetWeights()
        {
            return paramWeights;
        }

        public void SetBiases(Tensor biases)
        {
            paramBiases = biases;
        }

        public Tensor GetBiases()
        {
            return paramBiases;
        }

        public void SetGradWeights(Tensor weights)
        {
            gradWeights = weights;
        }

        public Tensor GetGradWeights()
        {
            return gradWeights;
        }

        public void SetGradBiases(Tensor biases)
        {
            gradBiases = biases;
        }

        public Tensor GetGradBiases()
        {
            return gradBiases;
        }

        public abstract Tensor Forward(Tensor inputs);
 
        public abstract Tensor Backward(Tensor gradient);
    }

    /// <summary>
    /// Linear layer.
    /// All vectors are treated as column vectors
    /// </summary>
    class Linear : Layer
    {

        public Linear(int inputSize, int outputSize)
        {

            Random rnd = new();
            double[,] weightsArray = new double[outputSize, inputSize];
            double[,] biasesArray = new double[outputSize, 1];
            for (int i = 0; i < outputSize; i++)
            {
                biasesArray[i,0] = rnd.NextDouble();
                for (int j = 0; j < inputSize; j++)
                {
                    weightsArray[i,j] = rnd.NextDouble();
                }
            }
            paramWeights = new Tensor(weightsArray);
            paramBiases = new Tensor(biasesArray);
            gradWeights = new Tensor(new double[,] {{0},{0}});
            gradBiases = new Tensor(new double[,] {{0},{0}});
            inputs = new Tensor(new double[,] {{0},{0}});
        }

        /// <summary>
        /// Updates and forwards inputs of this layer 
        /// </summary>
        /// <param name="inputs">Input parameters</param>
        /// <returns>Outputs after forward pass</returns>
        public override Tensor Forward(Tensor inputs)
        {
            this.inputs = inputs;
            return Tensor.SumTensors(Tensor.TensorProduct(paramWeights, inputs), paramBiases);
        }
        
        /// <summary>
        /// Backpropagates gradient values to current layer.
        /// <para>
        /// As layer is linear i.e. X*w + b, following value are obtained
        /// <para>
        /// -gradient weights: gradient with respect to w => D(grad) = grad * X^T
        /// </para>
        /// <para>
        /// -gradient biases: gradient with respect to b => D(grad) = grad * u, where u in unit vector. This yields 
        /// a column vector where each element is sum of grad rows i.e. grad = [Sum(grad[0]), ... , Sum(grad[1])]^T
        /// </para>
        /// <para>
        /// -input gradient => D(grad) = w^T * grad
        /// </para>
        /// </para>
        /// </summary>
        /// <param name="gradient">Total gradient</param>
        /// <returns>Gradient of current layer</returns>
        public override Tensor Backward(Tensor gradient)
        {
            double[,] gradVals = new double[gradient.GetSize()[0],1];
            double[,] gradData = gradient.GetData();

            for (int row = 0; row < gradient.GetSize()[0]; row++)
            {
                for (int col = 0; col < gradient.GetSize()[1]; col++)
                {
                    gradVals[row,0] +=  gradData[row,col];
                }
            }
            gradWeights = Tensor.TensorProduct(gradient, Tensor.Transpose(inputs));
            gradBiases = new Tensor(gradVals);
            return Tensor.TensorProduct(Tensor.Transpose(paramWeights), gradient);
        }
    }

    /// <summary>
    /// ReLU (rectified linear unit) activation layer
    /// </summary>
    class Relu : Layer
    {
        
        public Relu()
        {

        }

        /// <summary>
        /// ReLU activation function
        /// </summary>
        /// <param name="inputs">Input tensor</param>
        /// <returns>A tensor with each element run through ReLU function</returns>
        private Tensor Activation(Tensor inputs)
        {
            double[,] tensorData = (double[,])inputs.GetData().Clone();
            for (int row = 0; row < inputs.GetSize()[0]; row++)
            {
                for (int col = 0; col < inputs.GetSize()[1]; col++)
                {
                    tensorData[row,col] = Math.Max(tensorData[row,col], 0);
                }
            }
            Tensor tensorObject = new(tensorData);
            return tensorObject;
        }

        /// <summary>
        /// Elementwise ReLU gradient i.e. 1 if x > 0, else 0
        /// </summary>
        /// <param name="tensor">Tensor</param>
        /// <returns>Tensor with each element 1 or 0</returns>
        private Tensor Gradient(Tensor tensor)
        {
            double[,] tensorData = (double[,])tensor.GetData().Clone();
            for (int row = 0; row < tensor.GetSize()[0]; row++)
            {
                tensorData = (double[,])tensor.GetData().Clone();
                for (int col = 0; col < tensor.GetSize()[1]; col++)
                {
                    if (tensorData[row,col] > 0)
                    {
                        tensorData[row,col] = 1;
                    }
                    else
                    {
                        tensorData[row,col] = 0;
                    }
                }
            }
            Tensor gradTensor = new(tensorData);
            return gradTensor;
        }

        /// <summary>
        /// Forward pass
        /// </summary>
        /// <param name="inputs">Input tensor</param>
        /// <returns>Output tensor with ReLU applied</returns>
        public override Tensor Forward(Tensor inputs)
        {
            this.inputs = inputs;
            return Activation(inputs);
        }

        /// <summary>
        /// Backward pass
        /// </summary>
        /// <param name="gradient">Gradient of previous layer</param>
        /// <returns>Output gradient</returns>
        public override Tensor Backward(Tensor gradient)
        {
            return Tensor.MultiplyTensor(gradient, Gradient(inputs));
        }

    }


}