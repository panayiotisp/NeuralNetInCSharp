namespace NeuralNetInCSharp.ActivationFuncs {

    /// <summary>
    /// The Sigmoid activation function, which maps any real-valued input to the range (0,1).
    /// Implements <see cref="IActivationFunction"/> for use in neural networks.
    /// </summary>
    public class Sigmoid : IActivationFunction {

        #region Method: Activate (public - Interface IActivationFunction)
        /// <summary>
        /// Computes the Sigmoid activation: f(x) = 1 / (1 + e<sup>-x</sup>).
        /// </summary>
        /// <param name="x">The input value to the activation function.</param>
        /// <returns>The Sigmoid of <paramref name="x"/>, a value in the interval (0, 1).</returns>
        public double Activate(double x) {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        #endregion

        #region Method: Derivative (public - Interface IActivationFunction)
        /// <summary>
        /// Computes the derivative of the Sigmoid function in terms of its output.
        /// </summary>
        /// <param name="y">The output of the Sigmoid function (i.e., f(x)) for which the derivative is computed.</param>
        /// <returns>The derivative f′(x) = y * (1 - y), evaluated at the x that produced <paramref name="y"/>.</returns>
        public double Derivative(double y) {
            return y * (1 - y);
        }
        #endregion

    } // class: Sigmoid

} // namespace: NeuralNetInCSharp.ActivationFuncs
