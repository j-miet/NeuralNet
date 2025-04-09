namespace NeuralNet
{
    /// <summary>
    /// General class for loss functions
    /// </summary>
    abstract class LossFunction
    {
        public abstract double Loss(Tensor predicted, Tensor actual);

        public abstract Tensor Grad(Tensor predicted, Tensor actual);
    }

    /// <summary>
    /// Mean squared error, but is actually just total squared error as no normalization is done
    /// </summary>
    class MSE : LossFunction
    {
        /// <summary>
        /// Total loss of network
        /// <para>
        /// Loss is calculated as Sum(predicted - actual)**2
        /// </para>
        /// </summary>
        /// <param name="predicted">Predicted outputs</param>
        /// <param name="actual">Target outputs</param>
        /// <returns>Loss value >= 0</returns>
        public override double Loss(Tensor predicted, Tensor actual)
        {
            Tensor subtract = Tensor.SubtractTensors(predicted, actual);
            Tensor squared = Tensor.MultiplyTensor(subtract, subtract);
            return Tensor.SumValues(squared);
        }

        /// <summary>
        /// Gradient tensor 2*(predicted - actual)
        /// </summary>
        /// <param name="predicted">Predicted ouputs</param>
        /// <param name="actual">Target outputs</param>
        /// <returns>Tensor vector of loss gradients</returns>
        public override Tensor Grad(Tensor predicted, Tensor actual)
        {
            Tensor subtract = Tensor.SubtractTensors(predicted, actual);
            return Tensor.ScaleTensor(subtract, 2);
        }
    }
}