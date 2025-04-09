# NeuralNet

A simple neural network in C#, for learning purposes

Mostly adapted from [this video](https://www.youtube.com/watch?v=o64FV-ez6Gw)  
Lacks the batching, but implements custom tensor objects in place of Numpy library.

### Features

- No external dependencies required
- Basic 2D Tensor class build on top of arrays
- Implements linear and ReLU activation layers
- Uses MSE (actually total squared error because no normalization) as loss function, SGD as optimizer

### How to use
In the example below:

- 4 inputs are passed, each with 2 values + 4 targets, each with 3 values
- Create a new neural network: 2x4 linear layer -> apply ReLU -> 4x3 linear layer
- MSE loss function, SGD optimizer
- Trains network with given inputs & targets over 5000 epochs
- Separate each input and corresponding target to a Tensor pair
- Finally, displays each input (left), prediction (middle) and target (right)

```
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
```

This will give something like this as output:

>...  
Epoch 4999 | Input: 3 | Loss: 1,919570903492445  
Epoch 4999 | Input: 4 | Loss: 2,5897314273092396  
Epoch 5000 | Input: 1 | Loss: 0,6240217192524604  
Epoch 5000 | Input: 2 | Loss: 1,2753347837517826  
Epoch 5000 | Input: 3 | Loss: 1,9195705941435515  
Epoch 5000 | Input: 4 | Loss: 2,5897315662620315  
&nbsp;  
Inputs, Predictions and Targets:  
&nbsp;  
[1 1]  [0,7544540455195896 0,24918512584707192 1,0024116324921373]  [1 1 1]  
[1 0]  [0,2537740227168682 0,744413143131253 0,004910539329913366]  [0 0 0]  
[0 1]  [0,25198247576299976 0,7595856814134109 0,9975518376352713]  [0 0 1]  
[0 0]  [-0,24869754703972174 1,2548136986975917 5,074447304742602E-05]  [0 2 0]  

Adding more linear layers will not help with the predictions.  
But adding a non-linear ReLU layer between linear layers

```
NN network = new([new Linear(2, 4), new Relu(), new Linear(4, 3)]);
```

and training the network again, yields fantastic results:

>...  
Epoch 4999 | Input: 3 | Loss: 1,2314588657406425E-29  
Epoch 4999 | Input: 4 | Loss: 2,42653691483181E-29  
Epoch 5000 | Input: 1 | Loss: 5,2385294487332815E-30  
Epoch 5000 | Input: 2 | Loss: 8,533410397590965E-30  
Epoch 5000 | Input: 3 | Loss: 1,2314588657406425E-29  
Epoch 5000 | Input: 4 | Loss: 2,42653691483181E-29  
&nbsp;  
Inputs, Predictions and Targets:  
&nbsp;  
[1 1]  [0,9999999999999998 1,0000000000000022 1,0000000000000004]  [1 1 1]  
[1 0]  [2,3592239273284576E-16 -1,7763568394002505E-15 2,7755575615628914E-16]  [0 0 0]  
[0 1]  [7,91033905045424E-16 -1,7763568394002505E-15 1]  [0 0 1]  
[0 0]  [-9,992007221626409E-16 2,000000000000003 1,1102230246251565E-15]  [0 2 0]  
