using NeuralNetInCSharp.ActivationFuncs;


namespace NeuralNetInCSharp.Models {

    /// <summary>
    /// A simple perceptron-style neuron. It holds a weight for each input plus a bias,
    /// then runs the total through the given activation function.
    /// A single neuron holds:
    ///   - an array of weights(one per input),
    ///   - a bias,
    ///   - an activation function,
    ///   - and stores its last output and delta
    /// </summary>
    public class PerceptronNeuron {

        /// <summary>
        /// One weight per input, used to scale each input before summing.
        /// Weights are used to determine the importance of each input to the neuron's output.
        /// </summary>
        public double[] Weights { get; private set; }

        /// <summary>
        /// Constant term added to the weighted sum (shifting the activation function)
        /// </summary>
        public double Bias { get; set; }

        /// <summary>
        /// The most recent output after applying the activation function.
        /// </summary>
        public double Output { get; private set; }

        /// <summary>
        /// Error term used when adjusting weights during back-propagation.
        /// </summary>
        public double Delta { get; set; }

        /// <summary>
        /// The activation function to apply to the weighted sum of inputs.
        /// </summary>
        public IActivationFunction ActivationInstance { get; private set; }

        /// <summary>
        /// Random number generator used to initialize weights and bias.
        /// </summary>
        private static readonly Random RndGenerator = new();

        #region Constructor (public)
        /// <summary>
        /// Creates a new neuron with random weights and bias.
        /// </summary>
        /// <param name="inputCount">Number of inputs (and thus weights) this neuron will have.</param>
        /// <param name="activation">The activation function to apply to the weighted sum.</param>
        public PerceptronNeuron(int inputCount, IActivationFunction activation) {
            ActivationInstance = activation;
            Bias = RndGenerator.NextDouble() - 0.5;
            Weights = new double[inputCount];
            for (int i = 0; i < Weights.Length; i++) {
                Weights[i] = RndGenerator.NextDouble() - 0.5;
            }
        }
        #endregion

        #region Method: Compute (public)
        /// <summary>
        /// Computes the neuron's output from an array of inputs.
        /// </summary>
        /// <param name="inputs">Input values; the array length must match the number of weights.</param>
        /// <returns>The activated output (weight • input + bias, fed through the activation function).</returns>
        public double Compute(double[] inputs) {
            double sum = Bias;
            for (int i = 0; i < Weights.Length; i++) {
                sum += Weights[i] * inputs[i];
            }

            Output = ActivationInstance.Activate(sum);
            return Output;
        }
        #endregion

    } // class: PerceptronNeuron

} // namespace: NeuralNetInCSharp.Models