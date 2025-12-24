# Dense Layer / Fully Connected Layer 全连接层
# 实现神经网络中最基础的全连接层：
# 正向传播：计算输入与权重的乘积和偏置的加和，再经过激活函数。
# 反向传播：计算损失对输入、权重、偏置的梯度

# a example collected from 
# https://colab.research.google.com/github/Graylab/DL4Proteins-notebooks/blob/main/notebooks/WS01_NeuralNetworksWithNumpy.ipynb#scrollTo=2mKKhLLk4WgS

# Dense layer
class Layer_Dense:
  """
  Dense layer of a neural network
  Facilitates:
  - Forward propogation of data throught layer
  - Backward propogation of gradients during training
  """

  # Layer initialization
  def __init__(self, n_inputs, n_neurons):
    # Initialize weights and biases
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  # Forward pass
  def forward(self, inputs):
    # Remember input values
    self.inputs = inputs
    # Calculate output values from inputs, weights and biases
    self.output = np.dot(inputs, self.weights) + self.biases

  # Backward pass
  def backward(self, dvalues):
    # Gradients on parameters
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    # Gradient on values
    self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU:
  """
  Rectified linear unit activation function
  Applied to input of neural network layer
  Introduces non-linearity into the network
  """
  # Forward pass
  def forward(self, inputs):
    # Remember input values
    self.inputs = inputs
    # Calculate output values from inputs
    self.output = np.maximum(0, inputs)

  # Backward pass
  def backward(self, dvalues):
    # Since we need to modify original variable,
    # let’s make a copy of values first
    self.dinputs = dvalues.copy()
    # Zero gradient where input values were negative
    self.dinputs[self.inputs <= 0] = 0


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
  """
  Combination of softmax activation function and categorical cross entropy loss function
  Commonly used in classification tasks
  We minimize loss by adjustng model parameters to improve performance
  """

  # create activation and loss function objectives
  def __init__(self):
    self.activation = Activation_Softmax()
    self.loss = Loss_CategoricalCrossentropy()

  # forward pass
  def forward(self, inputs, y_true):
    # output layer's activation function
    self.activation.forward(inputs)
    # set the output
    self.output = self.activation.output
    # calculate and return loss value
    return self.loss.calculate(self.output, y_true)

  # backward pass
  def backward(self, dvalues, y_true):

    # number of samples
    samples = len(dvalues)

    # if labels one-hot encoded, turn into discrete values
    if len(y_true.shape) == 2:
      y_true = np.argmax(y_true, axis=1)

    # copy so we can safely modify
    self.dinputs = dvalues.copy()
    # Calculate gradient
    self.dinputs[range(samples), y_true] -= 1
    # Normalize gradient
    self.dinputs = self.dinputs / samples


# Adam optimizer
class Optimizer_Adam:

  """
  Adam optimization algorithm to optimize parameters of neural network
  Initalize with learning rate, decay, epsilon, momentum
  Pre-update params: Adjust learning rate based on decay
  Update params: Update params using momentum and cache corrections
  Post-update params: Track number of optimization steps performed
  """

  # Initialize optimizer - set settings
  def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.iterations = 0
    self.epsilon = epsilon
    self.beta_1 = beta_1
    self.beta_2 = beta_2

  # Call once before any parameter updates
  def pre_update_params(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

  # Update parameters
  def update_params(self, layer):

    # If layer does not contain cache arrays, create them filled with zeros
    if not hasattr(layer, 'weight_cache'):
      layer.weight_momentums = np.zeros_like(layer.weights)
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.bias_momentums = np.zeros_like(layer.biases)
      layer.bias_cache = np.zeros_like(layer.biases)

    # Update momentum with current gradients
    layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
    layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

    # Get corrected momentum
    # self.iteration is 0 at first pass
    # and we need to start with 1 here
    weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
    bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

    # update cache with squared current gradients
    layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
    layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

    # get corrected cache
    weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
    bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

    # Vanilla SGD parameter update + normalization with square root cache
    layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
    layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

  # call once after any parameter updates
  def post_update_params(self):
    self.iterations += 1


# Softmax activation
class Activation_Softmax:

  """
  Softmax activation function for multi-class classification
  Compute probabilities for each class
  """

  # Forward pass
  def forward(self, inputs):
    # Remember input values
    self.inputs = inputs
    # Get unnormalized probabilities
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

    # Normalize them for each sample
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    self.output = probabilities

  # Backward pass
  def backward(self, dvalues):
    # Create uninitialized array
    self.dinputs = np.empty_like(dvalues)

    # Enumerate outputs and gradients
    for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
      # Flatten output array
      single_output = single_output.reshape(-1, 1)

      # Calculate Jacobian matrix of the output
      jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

      # Calculate sample-wise gradient and add it to the array of sample gradients
      self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Common loss class
class Loss:

  # calculates data and regularization losses, given model output and ground truth values
  def calculate(self, output, y):

    # calculate sample losses
    sample_losses = self.forward(output, y)

    # calculate mean losses
    data_loss = np.mean(sample_losses)

    # return loss
    return data_loss


# cross entropy loss
class Loss_CategoricalCrossentropy(Loss):
  """
  Computes categorical cross entropy
  Quantifies discrepency between predicted and true class probabilities
  """

  # forward pass
  def forward(self, y_pred, y_true):

    # number samples in batch
    samples = len(y_pred)

    # clip data to prevent division by 0
    # clip both sides to not drag mean towards any value
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # probabilities for target values (only if categorical labels)
    if len(y_true.shape) == 1:
      correct_confidences = y_pred_clipped[ range(samples), y_true ]

    # mask values (only for one-hot encoded labels)
    elif len(y_true.shape) == 2:
      correct_confidences = np.sum( y_pred_clipped * y_true, axis=1 )

    # losses
    negative_log_likelihoods = -np.log(correct_confidences)
    return negative_log_likelihoods

  # backward pass
  def backward(self, dvalues, y_true):

    # number of samples
    samples = len(dvalues)
    # Number of labels in every sample
    # We’ll use the first sample to count them
    labels = len(dvalues[0])

    if len(y_true.shape) == 1:
      y_true = np.eye(labels)[y_true]

    # calculate gradient
    self.dinputs = -y_true / dvalues
    # Normalize gradient
    self.dinputs = self.dinputs / samples
