using NeuralNetInCSharp.ActivationFuncs;


namespace NeuralNetInCSharp.Models {

    /// <summary>
    /// Represents one layer in the network: a simple collection of neurons that all take the same inputs.
    /// </summary>
    public class Layer {

        /// <summary>
        /// The neurons contained in this layer.
        /// </summary>
        public PerceptronNeuron[] Neurons { get; private set; }

        #region Constructor (public)
        /// <summary>
        /// Creates a new layer with the given number of neurons.
        /// Each neuron gets its own set of random weights and the same activation function.
        /// </summary>
        /// <param name="inputCount">How many inputs each neuron should accept.</param>
        /// <param name="neuronCount">How many neurons to include in this layer.</param>
        /// <param name="activation">The activation function that every neuron in this layer will use.</param>
        public Layer(int inputCount, int neuronCount, IActivationFunction activation) {
            Neurons = new PerceptronNeuron[neuronCount];
            for (int i = 0; i < neuronCount; i++) {
                Neurons[i] = new PerceptronNeuron(inputCount, activation);
            }
        }
        #endregion

        #region Method: Compute (public)
        /// <summary>
        /// Feeds the same input array into each neuron in the layer and returns an array of their outputs.
        /// </summary>
        /// <param name="inputs">An array of input values; its length must match each neuron's expected input count.</param>
        /// <returns>An array of output values—one per neuron in this layer.</returns>
        public double[] Compute(double[] inputs) {
            double[] outputs = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++) {
                outputs[i] = Neurons[i].Compute(inputs);
            }

            return outputs;
        }
        #endregion

    } // class: Layer

} // namespace: NeuralNetInCSharp.Models