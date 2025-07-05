using NeuralNetInCSharp.ActivationFuncs;
using NeuralNetInCSharp.Models;


namespace NeuralNetInCSharp {

    /// <summary>
    /// Starts up a console app that builds, trains, and tests a neural network learning
    /// </summary>
    internal class Program {

        /// <summary>
        /// Sets up the network topology and parameters, runs training, then prints out a few test predictions to the console.
        /// </summary>
        private static void Main() {
            // set the network parameters
            int inputCount = 1;
            int[] hiddenLayers = [2];   // one hidden layer with 2 neurons
            int outputCount = 1;
            int epochs = 10000;
            double learningRate = 0.01;

            // Use a sigmoid in the hidden layer but keep the output linear so
            // predictions aren't squashed to the range (0,1)
            IActivationFunction hiddenActivation = new Sigmoid();
            IActivationFunction outputActivation = new Linear();

            // Build it
            var net = new NeuralNetwork(inputCount: inputCount,
                                        hiddenLayers: hiddenLayers,
                                        outputCount: outputCount,
                                        hiddenActivation: hiddenActivation,
                                        outputActivation: outputActivation,
                                        learningRate: learningRate);

            // Make some training data for y = 2x
            double[][] xTrain = [[1.0], [2.0], [3.0], [4.0]];
            double[][] yTrain = [[2.0], [4.0], [6.0], [8.0]];

            // train the network
            Console.WriteLine("Training...");
            net.Train(xTrain: xTrain, yTrain: yTrain, epochs: epochs);
            Console.WriteLine("Done.");

            // Try out some predictions
            foreach (double x in new[] { 5.0, 10.0, 525.0 }) {
                double pred = net.Predict([x])[0];
                Console.WriteLine($"for x={x} predict y={pred}");
            }
        }

    } // class: Program

} // namespace: NeuralNetInCSharp
