# NeuralNetInCSharp

This is a small neural network implementation written in C#. The example in
`Program.cs` trains the network to learn the simple function `y = 2x`.

When experimenting with different activation functions it is important to keep
the output layer linear for regression tasks. Using a sigmoid activation will
squash values to the `(0,1)` range and hidden sigmoids can also limit the
network's ability to extrapolate far beyond the training data. The example
therefore keeps both the hidden and output layers linear so the network can
learn the unbounded function `y = 2x`.
