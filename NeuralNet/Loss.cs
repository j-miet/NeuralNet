using NeuralNet;

namespace Loss
{
    static class MSE
    {
   
        public static double Loss(Tensor predicted, Tensor actual)
        {
            Tensor subtract = Tensor.SubtractTensors(predicted, actual);
            Tensor squared = Tensor.MultiplyTensor(subtract, subtract);
            return Tensor.SumValues(squared);
        }

        public static Tensor Grad(Tensor predicted, Tensor actual)
        {
            Tensor subtract = Tensor.SubtractTensors(predicted, actual);
            return Tensor.ScaleTensor(subtract, 2);
        }
    }
}