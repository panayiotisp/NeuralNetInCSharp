namespace NeuralNetInCSharp.ActivationFuncs {

    /// <summary>
    /// interface for the activation functions to be used in our neural network implementation.
    /// Anything that can take a real x, compute f(x), and also provide f′(x) for back-prop.
    /// </summary>
    public interface IActivationFunction {

        /// <summary>
        /// Computes the activation function f(x).
        /// </summary>
        /// <param name="x">The input value to the activation function.</param>
        /// <returns>
        /// The result of the activation function, f(x).
        /// </returns>
        double Activate(double x);

        /// <summary>
        /// Computes the derivative of the activation function based on its output.
        /// </summary>
        /// <param name="y">The output of the activation function (i.e., f(x)) for which the derivative is to be computed </param>
        /// <returns>
        /// The derivative f′(x) evaluated at the x that produced <paramref name="y"/>.
        /// </returns>
        double Derivative(double y);

    } // interface: IActivationFunction

} // namespace: NeuralNetInCSharp.ActivationFuncs
