"""Linear Classifier Module"""
import numpy as np

class DSELinearClassifier:
    """Linear Classifier.

    Parameters
    ------------
    activation : string
      Type of linear classifier
    random_state : int
      Random number generatorseed for random weight
      initialization.
    initial_weight : array
      Sets the initial weight for the classifier
    learning rate : float
     Learning rate(between 0.0 and 1.0)

    Attributes
    ------------
    initial_weight : 1D-array
      Weights after fitting
    _fit_errors : list
      Number of misclassifications (total cost per epoch for Logistic Regression)
    """

    def __init__(self, activation, random_state = 42, initial_weight = None, learning_rate = 1):
        self.activation = activation
        self.random_state = random_state
        self.initial_weight = initial_weight
        self.learning_rate = learning_rate

    def fit(self, X, y, batch_size = None, max_epochs = 10):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like shape - [n_examples]
          Target values.
        batch_size : int
          Batch size for mini-batch gradient descent (not applicable for perceptron)
        max_epochs : int
          Passes over the training dataset.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        if self.initial_weight is None:
            self.initial_weight = rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self._fit_errors = []

        m = np.size(y)
        if batch_size == None:
            batch_size = m

        if self.activation == 'Perceptron':
            for i in range(max_epochs):
                errors = 0
                for xi, target in zip(X, y):
                    update = self.learning_rate * (target - self.predict(xi))
                    self.initial_weight[1:] += update * xi
                    self.initial_weight[0] += update
                    errors += int(update != 0.0)
                self._fit_errors.append(errors)
            return self

        if self.activation == 'Logistic':
            for i in range(max_epochs):
                cost_sum = 0
                for j in range(0,m,batch_size):
                    X_i = X[j:j+batch_size]
                    y_i = y[j:j+batch_size]
                    net_input = self.net_input(X_i)
                    output = self.log_activation(net_input)
                    errors = (y_i - output)
                    self.initial_weight[1:] += self.learning_rate * X_i.T.dot(errors)
                    self.initial_weight[0] += self.learning_rate * errors.sum()

                    cost = -y_i.dot(np.log(output)) - ((1 - y_i).dot(np.log(1 - output)))
                    cost_sum += cost
                self._fit_errors.append(cost_sum)
            return self

        if self.activation == 'Hypertan':

            for i in range(max_epochs):
                cost_sum = 0
                for j in range(0,m,batch_size):
                    X_i = X[j:j+batch_size]
                    y_i = y[j:j+batch_size]
                    net_input = self.net_input(X_i)
                    output = self.hypertan_activation(net_input)
                    errors = (y_i - output)
                    self.initial_weight[1:] += self.learning_rate * 2 * X_i.T.dot(errors)
                    self.initial_weight[0] += self.learning_rate * 2 * errors.sum()

                    cost = -(1/2)*((1+y_i).dot(np.log((1 + output)/2))
                           - ((1 - y_i).dot(np.log((1 - output)/2))))
                    cost_sum += cost
                self._fit_errors.append(cost_sum)
            return self

    def net_input(self, X):
        """Calculate net input (pre-threshold prediction)"""
        return np.dot(X, self.initial_weight[1:]) + self.initial_weight[0]

    def log_activation(self,z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -4, 4)))

    def hypertan_activation(self, z):
        """Compute hyperbolic tangent activation"""
        return np.tanh(np.clip(z,-4,4))

    def predict(self, X):
        """Return class label after unit step"""
        if self.activation == 'Perceptron':
            return np.where(self.net_input(X) >= 0.0, 1, -1)
        if self.activation == 'Hypertan':
            return np.where(self.hypertan_activation(self.net_input(X)) >= 0.0, 1, -1)
        if self.activation == 'Logistic':
            return np.where(self.log_activation(self.net_input(X)) >= 0.5, 1, 0)
