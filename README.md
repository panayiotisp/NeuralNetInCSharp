# NeuralNetInCSharp

This is a small neural network implementation written in C#. The example in
`Program.cs` trains the network to learn the simple function `y = 2x`.

When experimenting with different activation functions it is important to keep
the output layer linear for regression tasks. Using a sigmoid activation for the
output would squash all predictions to the `(0,1)` range which is why the example
uses a sigmoid hidden layer paired with a linear output layer.
