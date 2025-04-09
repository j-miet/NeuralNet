namespace NeuralNet
{

    /// <summary>
    /// Trains a neural network for given inputs and target outputs
    /// </summary>
    static class TrainNN
    {
        /// <summary>
        /// Neural network
        /// </summary>
        /// <param name="network">NN object</param>
        /// <param name="inputs">Tensor</param>
        /// <param name="targets">Tensor</param>
        /// <param name="epochs">Number of epochs</param>
        /// <param name="loss">Loss function</param>
        /// <param name="optimizer">Optimizer</param>
        public static void Train(NN network, 
                            Tensor inputs, 
                            Tensor targets, 
                            int epochs, 
                            LossFunction loss, 
                            Optimizer optimizer)
        {
            Tensor[] inputTensors = Tensor.TensorList(inputs);
            Tensor[] targetTensors = Tensor.TensorList(targets);
            for (int epoch = 1; epoch < epochs+1; epoch++)
            {
                double epochLoss = 0;
                for (int i = 0; i < inputTensors.Length; i++)
                {
                    Tensor predicted = network.Forward(inputTensors[i]);
                    epochLoss += loss.Loss(predicted, targetTensors[i]);
                    Tensor grad = loss.Grad(predicted, targetTensors[i]);
                    network.Backward(grad);
                    optimizer.Step(network);
                    Console.WriteLine("Epoch "+ epoch +" | Input: " + (i+1) + " | Loss: " + epochLoss);
                }
            }
        }

        /// <summary>
        /// Prints input, prediction and target neatly
        /// <para>
        /// Input, prediction and target are all nx1 tensors
        /// </para>
        /// </summary>
        /// <param name="input">Input values</param>
        /// <param name="predicted">Predicted values</param>
        /// <param name="target">Target values</param>
        public static void PrettyPrint(Tensor input, Tensor predicted, Tensor target)
        {
            Tensor[] transposes = [
                Tensor.Transpose(input), 
                Tensor.Transpose(predicted),
                Tensor.Transpose(target),
            ];

            foreach (Tensor tensor in transposes)
            {
                double[] values = (double[])new double[tensor.GetSize()[1]].Clone();
                for (int col = 0; col < tensor.GetSize()[1]; col++)
                {
                    values[col] = tensor.GetData()[0, col];
                }
                Console.Write("["+string.Join<double>(" ", values)+"]  ");
            }
            Console.WriteLine();
        }
    }

}