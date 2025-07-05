namespace NeuralNetInCSharp.ActivationFuncs {

    /// <summary>
    /// A simple identity activation: it just hands back whatever you give it.
    /// </summary>
    public class Linear : IActivationFunction {

        #region Activate (public - Interface IActivationFunction)
        /// <summary>
        /// Passes the input straight through.
        /// </summary>
        /// <param name="x">The value to activate.</param>
        /// <returns>Exactly the same <paramref name="x"/> you passed in as this is a linear function</returns>
        public double Activate(double x) {
            return x;
        }
        #endregion

        #region Derivative (public - Interface IActivationFunction)
        /// <summary>
        /// The slope of the identity function is always 1.
        /// </summary>
        /// <param name="y">The output from Activate (not used here).</param>
        /// <returns>1.0, since d/dx(x) = 1</returns>
        public double Derivative(double y) {
            return 1.0;
        }
        #endregion

    } // class: Linear

} // namespace: NeuralNetInCSharp.ActivationFuncs
