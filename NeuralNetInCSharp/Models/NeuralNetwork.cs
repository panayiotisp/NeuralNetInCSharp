using NeuralNetInCSharp.ActivationFuncs;


namespace NeuralNetInCSharp.Models {

    /// <summary>
    /// A simple feed-forward neural network made up of layers you define.
    /// Uses the same activation function everywhere and updates weights via back-prop.
    /// </summary>
    public class NeuralNetwork {

        /// <summary>
        /// The layers of the network, each with its own set of neurons(perceptrons).
        /// </summary>
        private readonly Layer[] NetworkLayers;

        /// <summary>
        /// how big each weight update step is
        /// </summary>
        private readonly double LearningRate;

        #region Constructor (public)
        /// <summary>
        /// Set up a new network.
        /// </summary>
        /// <param name="inputCount">How many input features each example has.</param>
        /// <param name="hiddenLayers">An array of hidden-layer sizes (e.g. new[] { 2, 3 } makes a 2-neuron layer then a 3-neuron layer).</param>
        /// <param name="outputCount">How many outputs the network should produce.</param>
        /// <param name="hiddenActivation">The activation function for hidden layers.</param>
        /// <param name="outputActivation">The activation function for the output layer. If null, <paramref name="hiddenActivation"/> is used.</param>
        /// <param name="learningRate">How big each weight update step is (default is 0.1).</param>
        public NeuralNetwork(int inputCount, int[] hiddenLayers, int outputCount,
                             IActivationFunction hiddenActivation,
                             IActivationFunction? outputActivation = null,
                             double learningRate = 0.1) {
            LearningRate = learningRate;

            // Build a list [ inputCount, ...hidden..., outputCount ]
            List<int> sizes = [inputCount, .. hiddenLayers, outputCount];

            // Create each layer
            NetworkLayers = new Layer[sizes.Count - 1];
            for (int i = 0; i < NetworkLayers.Length; i++) {
                IActivationFunction act = i == NetworkLayers.Length - 1
                    ? outputActivation ?? hiddenActivation
                    : hiddenActivation;
                NetworkLayers[i] = new Layer(sizes[i], sizes[i + 1], act);
            }
        }
        #endregion

        #region Method: FeedForward (public)
        /// <summary>
        /// Runs a forward pass: feed inputs in, get outputs back.
        /// </summary>
        /// <param name="inputs">The input values to push through the network.</param>
        /// <returns>The final activations from the output layer. </returns>
        public double[] FeedForward(double[] inputs) {
            double[] activations = inputs;
            foreach (Layer layer in NetworkLayers) {
                activations = layer.Compute(activations);
            }

            return activations;
        }
        #endregion

        #region Method: GetWeights (public)
        /// <summary>
        /// Does back-propagation on one example, adjusting all weights and biases.
        /// </summary>
        /// <param name="inputs">The inputs for this training example.</param>
        /// <param name="targets">The desired outputs for this training example.</param>
        public void BackPropagate(double[] inputs, double[] targets) {
            // 1. Forward pass but remember each layer’s outputs
            List<double[]> layerInputs = [inputs];
            List<double[]> layerOutputs = [];
            double[] act = inputs;
            foreach (Layer layer in NetworkLayers) {
                act = layer.Compute(act);
                layerOutputs.Add(act);
                layerInputs.Add(act);
            }

            // 2. Work backwards: compute deltas
            for (int l = NetworkLayers.Length - 1; l >= 0; l--) {
                Layer layer = NetworkLayers[l];
                double[] errors = new double[layer.Neurons.Length];

                if (l == NetworkLayers.Length - 1) {
                    // output layer: error = target − actual
                    for (int j = 0; j < layer.Neurons.Length; j++) {
                        errors[j] = targets[j] - layerOutputs[l][j];
                    }
                } else {
                    // hidden layer: error = sum of next-layer (weight × delta)
                    Layer next = NetworkLayers[l + 1];
                    for (int j = 0; j < layer.Neurons.Length; j++) {
                        double e = 0;
                        for (int k = 0; k < next.Neurons.Length; k++) {
                            e += next.Neurons[k].Weights[j] * next.Neurons[k].Delta;
                        }

                        errors[j] = e;
                    }
                }

                // Update deltas, weights, and biases
                for (int j = 0; j < layer.Neurons.Length; j++) {
                    PerceptronNeuron neuron = layer.Neurons[j];
                    double outputVal = layerOutputs[l][j];
                    neuron.Delta = errors[j] * neuron.ActivationInstance.Derivative(outputVal);

                    // grab the inputs that went into this layer
                    double[] prevActivations = l == 0
                        ? inputs
                        : layerOutputs[l - 1];

                    // tweak each weight
                    for (int w = 0; w < neuron.Weights.Length; w++) {
                        neuron.Weights[w] += LearningRate * neuron.Delta * prevActivations[w];
                    }

                    // tweak the bias
                    neuron.Bias += LearningRate * neuron.Delta;
                }
            }
        }
        #endregion

        #region Method: Train (public)
        /// <summary>
        /// Train the network on a batch of examples for a set number of passes.
        /// </summary>
        /// <param name="xTrain">Array of input arrays—one per training example.</param>
        /// <param name="yTrain">Array of target arrays matching xTrain.</param>
        /// <param name="epochs">How many times to loop over the entire dataset.</param>
        public void Train(double[][] xTrain, double[][] yTrain, int epochs) {
            for (int e = 0; e < epochs; e++) {
                for (int i = 0; i < xTrain.Length; i++) {
                    BackPropagate(xTrain[i], yTrain[i]);
                }
            }
        }
        #endregion

        #region Method: Predict (public)
        /// <summary>
        /// Run one sample through the network and get its prediction.
        /// </summary>
        /// <param name="input">The input values for which you want a prediction.</param>
        /// <returns>The network’s output for the given input.</returns>
        public double[] Predict(double[] input) {
            return FeedForward(input);
        }
        #endregion

    } // class: NeuralNetwork

} // namespace: NeuralNetInCSharp.Models