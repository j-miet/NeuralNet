using NeuralNet;

static class TrainAndTest
{
    static void Main(string[] args)
    {
        Tensor inputs = new(new double[,] {
            { 1, 1 }, 
            { 1, 0 }, 
            { 0, 1 }, 
            { 0, 0 }, 
            });
        Tensor targets = new(new double[,] {
            { 1, 1, 1},
            { 0, 0, 0},
            { 0, 0, 1},
            { 0, 2, 0},
        });

        NN network = new([new Linear(2, 4), new Relu(), new Linear(4, 3)]);
        MSE loss = new(); 
        SGD optimizer = new(network);
        TrainNN.Train(network, inputs, targets, 5000, loss, optimizer);

        Tensor[] inputTensors = Tensor.TensorList(inputs);
        Tensor[] targetTensors = Tensor.TensorList(targets);
        var zip = Enumerable.Zip(inputTensors, targetTensors);
        Console.WriteLine("\nInputs, Predictions and Targets:\n");
        foreach (var (First, Second) in zip)
        {
            Tensor predicted = network.Forward(First);
            TrainNN.PrettyPrint(First, predicted, Second);
        }
    }
}
