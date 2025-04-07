using NeuralNet;

static class TrainNN
{
    static void Main(string[] args)
    {
        double[] inputs = Config.inputs;
        double[] outputs = Config.outputs;
        int layers = Config.LAYERS;

        NN network = new(inputs.Length, outputs.Length, layers);
        Tensor tensor1 = new([1.0, 2.2, 3.0]);
        Tensor tensor2 = new([-1, 4.3, 3.0]);
        double[,] matrix = new double[3,0];
        Console.WriteLine(matrix.GetLength(1));
        
    }
}
