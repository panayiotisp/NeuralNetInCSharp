namespace NeuralNetInCSharp.ActivationFuncs {

    /// <summary>
    /// Anything that can take a real x, compute f(x), 
    /// and also provide f′(x) for back-prop.
    /// </summary>
    public interface IActivationFunction {

        /// <summary>
        /// Compute f(x).
        /// </summary>
        double Activate(double x);

        /// <summary>
        /// Given y=f(x), compute f′(x).
        /// </summary>
        double Derivative(double y);

    } // interface: IActivationFunction

} // namespace: NeuralNetInCSharp.ActivationFuncs