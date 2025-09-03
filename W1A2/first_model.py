import os
import warnings
import numpy as np
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
import json

try:
    from testCases import *
    from public_tests import *
    from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
    from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
    train_X, train_Y,test_X, test_Y = load_2D_dataset()

    print("Original assignment files imported successfully")
except ImportError as e:
    print(f"Failed to import original files: {e}")
    print("Required files:")
    print("   - testCases_v2.py")
    print("   - public_tests.py") 
    print("   - planar_utils.py")
    exit(1)



import sklearn
import sklearn.datasets
import scipy.io
import sklearn.linear_model


# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')


print("Neural Network Model Conversion using Original Dataset")
print("="*60)

def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """

    grads = {}
    costs = []                           
    m = X.shape[1]                      
    layers_dims = [X.shape[0], 20, 3, 1]


    parameters = initialize_parameters(layers_dims)



    for i in range(0, num_iterations):

        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        assert (lambd == 0 or keep_prob == 1)   
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)


        parameters = update_parameters(parameters, grads, learning_rate)


        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)


    return parameters



def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.

    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model

    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]ㅋ

    cross_entropy_cost = compute_cost(A3, Y) 
    L2_regularization_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) * lambd / (2 * m)


    cost = cross_entropy_cost + L2_regularization_cost

    return cost


def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m * (np.dot(dZ3, A2.T) + lambd * W3)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * (np.dot(dZ2, A1.T) + lambd * W2)

    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))

    dW1 = 1./m * (np.dot(dZ1, X.T) + lambd * W1)

    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """

    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    D1 = np.random.rand(*A1.shape)
    D1 = (D1 < keep_prob).astype(int)
    A1 *= D1
    A1 /= keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    D2 = np.random.rand(*A2.shape)
    D2 = (D2 < keep_prob).astype(int)
    A2 *= D2
    A2 /= keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache



def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 *= D2
    dA2 /= keep_prob

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)

    dA1 *= D1
    dA1 /= keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients



def create_tensorflow_model(input_size, layers_dims=[20, 3, 1], keep_prob=1.0, lambd=0.0):
    """
    Creates a TensorFlow model equivalent to the NumPy implementation
    
    Arguments:
    input_size -- size of input features
    layers_dims -- list of layer dimensions [hidden1, hidden2, output]
    keep_prob -- dropout keep probability
    lambd -- L2 regularization parameter
    
    Returns:
    model -- compiled TensorFlow model
    """
    
    model = keras.Sequential([
        layers.Input(shape=(input_size,)),
        
        layers.Dense(layers_dims[0], 
                    activation='relu',

                    name='dense_1'),
        

        layers.Dropout(1 - keep_prob, name='dropout_1') if keep_prob < 1 else layers.Lambda(lambda x: x),
        
        layers.Dense(layers_dims[1], 
                    activation='relu',

                    name='dense_2'),
        
        layers.Dropout(1 - keep_prob, name='dropout_2') if keep_prob < 1 else layers.Lambda(lambda x: x),
        
        layers.Dense(layers_dims[2], 
                    activation='sigmoid',

                    name='output')
    ])
    
    return model

def train_tensorflow_model(X, Y, learning_rate=0.3, num_iterations=30000, 
                          print_cost=True, lambd=0, keep_prob=1):
    """
    Train TensorFlow model equivalent to the original NumPy implementation
    Matches the exact training procedure with proper cost calculation
    """
    
    # Transpose data to match TensorFlow conventions (samples, features)
    X_tf = X.T  # Shape: (num_examples, input_size)
    Y_tf = Y.T  # Shape: (num_examples, output_size)
    
    # Create model
    input_size = X.shape[0]
    model = create_tensorflow_model(input_size, keep_prob=keep_prob, lambd=lambd)
    
    # Print model structure for verification
    print("Model Architecture:")
    model.summary()
    
    # Compile model with exact same optimizer settings
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Custom callback to match NumPy training exactly
    class CostCallback(keras.callbacks.Callback):
        def __init__(self, print_every=10000, record_every=1000):
            self.print_every = print_every
            self.record_every = record_every
            self.costs = []
            
        def on_epoch_end(self, epoch, logs=None):
            cost = logs.get('loss', 0)
            
            # Print cost every 10000 iterations (like NumPy version)
            if epoch % self.print_every == 0 and print_cost:
                print(f"Cost after iteration {epoch}: {cost}")
            
            # Record cost every 1000 iterations (like NumPy version)
            if epoch % self.record_every == 0:
                self.costs.append(cost)
    
    cost_callback = CostCallback()
    
    # Train model with full batch (matching NumPy SGD)
    epochs = num_iterations
    history = model.fit(X_tf, Y_tf, 
                       epochs=epochs, 
                       batch_size=X_tf.shape[0],  # Full batch like NumPy
                       verbose=0,
                       callbacks=[cost_callback])
    
    # Plot costs (matching NumPy version)
    if print_cost:
        plt.figure(figsize=(10, 6))
        plt.plot(cost_callback.costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title(f"Learning rate = {learning_rate}")
        plt.show()
    
    return model, history

def convert_to_tflite(model, quantize=False, optimize_for_size=True):
    """
    Convert TensorFlow model to TensorFlow Lite
    
    Arguments:
    model -- trained TensorFlow model
    quantize -- whether to apply quantization
    optimize_for_size -- whether to optimize for model size
    
    Returns:
    tflite_model -- converted TFLite model (bytes)
    """
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flags
    if optimize_for_size:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Apply quantization if requested
    if quantize:
        converter.target_spec.supported_types = [tf.float16]
        # For full integer quantization, you would need representative dataset:
        # converter.representative_dataset = representative_dataset_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
    
    # Convert model
    tflite_model = converter.convert()
    
    return tflite_model

def save_tflite_model(tflite_model, filename="model.tflite"):
    """Save TFLite model to file"""
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved as {filename}")

def test_tflite_model(tflite_model, X_test):
    """
    Test the converted TFLite model
    
    Arguments:
    tflite_model -- TFLite model bytes
    X_test -- test input data (shape: features, samples)
    
    Returns:
    predictions -- model predictions
    """
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare test data (transpose to match TensorFlow conventions)
    X_test_tf = X_test.T.astype(np.float32)
    
    predictions = []
    
    # Run inference for each sample
    for i in range(X_test_tf.shape[0]):
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], 
                             X_test_tf[i:i+1])  # Add batch dimension
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])
    
    return np.array(predictions)

# Complete example with load_2D_dataset() and enhanced functionality:

def main():
    """
    Complete pipeline: Load data -> Train -> Convert to TFLite -> Test
    """
    
    # Load the dataset
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    
    print(f"Training set: {train_X.shape}, {train_Y.shape}")
    print(f"Test set: {test_X.shape}, {test_Y.shape}")
    
    # Train different models for comparison (matching your config values)
    models_config = {
        'baseline': {'lambd': 0, 'keep_prob': 1},
        'l2_regularized': {'lambd': 0.7, 'keep_prob': 1},
        'dropout': {'lambd': 0, 'keep_prob': 0.86},
    }
    
    tflite_models = {}
    
    for model_name, config in models_config.items():
        print(f"\n=== Training {model_name} model ===")
        print(f"Configuration: lambd={config['lambd']}, keep_prob={config['keep_prob']}")
        
        # Train TensorFlow model
        model, history = train_tensorflow_model(
            train_X, train_Y,
            learning_rate=0.3,
            num_iterations=30000,
            print_cost=True,
            lambd=config['lambd'],
            keep_prob=config['keep_prob']
        )
        
        # Evaluate on test set
        test_X_tf = test_X.T
        test_Y_tf = test_Y.T
        test_loss, test_acc = model.evaluate(test_X_tf, test_Y_tf, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
        
        # Convert to TFLite
        print(f"Converting {model_name} to TFLite...")
        tflite_model = convert_to_tflite(model, quantize=True, optimize_for_size=True)
        
        # Save TFLite model
        filename = f"model_{model_name}.tflite"
        save_tflite_model(tflite_model, filename)
        
        # Test TFLite model
        tflite_predictions = test_tflite_model(tflite_model, test_X)
        
        # Compare TensorFlow vs TFLite predictions
        tf_predictions = model.predict(test_X_tf, verbose=0)
        
        # Calculate accuracy for TFLite model
        tflite_accuracy = np.mean((tflite_predictions > 0.5) == test_Y_tf.flatten())
        tf_accuracy = np.mean((tf_predictions > 0.5) == test_Y_tf)
        
        print(f"TensorFlow accuracy: {tf_accuracy:.4f}")
        print(f"TFLite accuracy: {tflite_accuracy:.4f}")
        print(f"Prediction difference (MSE): {np.mean((tf_predictions.flatten() - tflite_predictions)**2):.6f}")
        
        # Store for later use
        tflite_models[model_name] = {
            'tflite_model': tflite_model,
            'tf_model': model,
            'accuracy': tflite_accuracy,
            'config': config
        }
        
        # Visualize decision boundary (optional)
        try:
            plot_decision_boundary_tflite(tflite_model, train_X, train_Y, model_name)
        except Exception as e:
            print(f"Could not plot decision boundary for {model_name}: {e}")
    
    # Find best model
    best_model = max(tflite_models.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
    print(f"Best model config: {best_model[1]['config']}")
    
    return tflite_models

def plot_decision_boundary_tflite(tflite_model, X, Y, title="TFLite Model"):
    """
    Plot decision boundary for TFLite model
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create mesh
        h = 0.01
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Create grid points
        grid_points = np.c_[xx.ravel(), yy.ravel()].T
        
        # Predict using TFLite model
        Z = test_tflite_model(tflite_model, grid_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        colors = ['red', 'blue']
        for i in range(2):
            plt.scatter(X[0, Y[0] == i], X[1, Y[0] == i], 
                       c=colors[i], marker='o', s=50, edgecolors='black')
        
        plt.title(f'Decision Boundary - {title}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
        
    except Exception as e:
        print(f"Could not plot decision boundary: {e}")

# Example usage matching your original code structure:

if __name__ == "__main__":
    print("Neural Network Model Conversion using Original Dataset")
    print("="*60)
    
    # Load the dataset (your code)
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    
    print(f"Training set: {train_X.shape}, {train_Y.shape}")
    print(f"Test set: {test_X.shape}, {test_Y.shape}")
    
    # Method 1: Train from scratch with TensorFlow and convert to TFLite
    print("\n" + "="*40)
    print("METHOD 1: Train from scratch with TensorFlow")
    print("="*40)
    
    models = main()
    
    # Method 2: If you already have trained NumPy parameters
    print("\n" + "="*40)
    print("METHOD 2: Convert existing NumPy model")
    print("="*40)
    
    # Train original NumPy model first (example)
    print("Training original NumPy model...")
    numpy_parameters = model(train_X, train_Y, learning_rate=0.3, num_iterations=30000, 
                            print_cost=True, lambd=0, keep_prob=1)
    
    # Convert NumPy model to TFLite
    tflite_model, tf_model, results = train_and_convert_existing_model(
        numpy_parameters, train_X, train_Y, test_X, test_Y
    )
    
    # Save the converted model
    save_tflite_model(tflite_model, "converted_numpy_model.tflite")
    
    print("\nConversion completed!")
    print("Files created:")
    print("  - model_baseline.tflite")
    print("  - model_l2_regularized.tflite") 
    print("  - model_dropout.tflite")
    print("  - converted_numpy_model.tflite")
    
    # Quick test of the final TFLite model
    print("\n" + "="*40)
    print("FINAL TFLITE MODEL TEST")
    print("="*40)
    
    final_predictions = test_tflite_model(tflite_model, test_X[:, :5])  # Test first 5 samples
    print(f"Sample TFLite predictions: {final_predictions}")
    print(f"Actual labels: {test_Y[:, :5].flatten()}")
    
    # Model size information
    try:
        model_size = len(tflite_model)
        print(f"\nTFLite model size: {model_size / 1024:.2f} KB")
    except:
        pass

# For simple usage without the full pipeline:
def simple_usage_example():
    """Simple example for quick testing"""
    # Load dataset
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    
    # Basic model
    model, history = train_tensorflow_model(train_X, train_Y)
    tflite_model = convert_to_tflite(model, quantize=True)
    save_tflite_model(tflite_model, "my_model.tflite")
    
    # Test
    predictions = test_tflite_model(tflite_model, test_X)
    print(f"Sample predictions: {predictions[:5]}")
    
    return tflite_model

# Additional utility functions to match NumPy implementation exactly

def load_numpy_weights_to_tensorflow(model, numpy_parameters):
    """
    Load weights from NumPy parameters dictionary to TensorFlow model
    
    Arguments:
    model -- TensorFlow model
    numpy_parameters -- dictionary with keys like 'W1', 'b1', 'W2', 'b2', 'W3', 'b3'
    """
    
    try:
        # Get only Dense layers (skip Dropout layers)
        dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        
        print(f"Found {len(dense_layers)} dense layers in the model")
        
        # Set weights for each dense layer
        if 'W1' in numpy_parameters and 'b1' in numpy_parameters:
            # First layer: transpose weights for TensorFlow convention
            W1 = numpy_parameters['W1'].T  # TF expects (input_size, units)
            b1 = numpy_parameters['b1'].flatten()  # TF expects (units,)
            dense_layers[0].set_weights([W1, b1])
            print(f"Layer 1: W1 {W1.shape}, b1 {b1.shape}")
            
        if 'W2' in numpy_parameters and 'b2' in numpy_parameters:
            # Second layer
            W2 = numpy_parameters['W2'].T
            b2 = numpy_parameters['b2'].flatten()
            dense_layers[1].set_weights([W2, b2])
            print(f"Layer 2: W2 {W2.shape}, b2 {b2.shape}")
            
        if 'W3' in numpy_parameters and 'b3' in numpy_parameters:
            # Output layer
            W3 = numpy_parameters['W3'].T
            b3 = numpy_parameters['b3'].flatten()
            dense_layers[2].set_weights([W3, b3])
            print(f"Layer 3: W3 {W3.shape}, b3 {b3.shape}")
            
        print("Successfully loaded NumPy weights into TensorFlow model!")
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Please check the structure of your numpy_parameters dictionary")

def train_and_convert_existing_model(numpy_parameters, train_X, train_Y, test_X, test_Y):
    """
    Convert existing NumPy trained parameters to TFLite
    
    Arguments:
    numpy_parameters -- dictionary containing trained NumPy weights
    train_X, train_Y, test_X, test_Y -- dataset splits
    
    Returns:
    tflite_model, tf_model, comparison_results
    """
    
    print("Converting existing NumPy model to TensorFlow...")
    
    # Create TensorFlow model with same architecture
    input_size = train_X.shape[0]
    model = create_tensorflow_model(input_size, lambd=0, keep_prob=1)  # No regularization for pre-trained weights
    
    # Load NumPy weights
    load_numpy_weights_to_tensorflow(model, numpy_parameters)
    
    # Compile model (needed for evaluation)
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Test on training and test sets
    train_X_tf = train_X.T
    train_Y_tf = train_Y.T
    test_X_tf = test_X.T
    test_Y_tf = test_Y.T
    
    train_loss, train_acc = model.evaluate(train_X_tf, train_Y_tf, verbose=0)
    test_loss, test_acc = model.evaluate(test_X_tf, test_Y_tf, verbose=0)
    
    print(f"TensorFlow model - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
    
    # Convert to TFLite
    tflite_model = convert_to_tflite(model, quantize=True, optimize_for_size=True)
    
    # Test TFLite model
    tflite_train_pred = test_tflite_model(tflite_model, train_X)
    tflite_test_pred = test_tflite_model(tflite_model, test_X)
    
    # Calculate TFLite accuracies
    tflite_train_acc = np.mean((tflite_train_pred > 0.5) == train_Y_tf.flatten())
    tflite_test_acc = np.mean((tflite_test_pred > 0.5) == test_Y_tf.flatten())
    
    print(f"TFLite model - Train accuracy: {tflite_train_acc:.4f}, Test accuracy: {tflite_test_acc:.4f}")
    
    # Compare predictions
    tf_train_pred = model.predict(train_X_tf, verbose=0)
    tf_test_pred = model.predict(test_X_tf, verbose=0)
    
    train_mse = np.mean((tf_train_pred.flatten() - tflite_train_pred)**2)
    test_mse = np.mean((tf_test_pred.flatten() - tflite_test_pred)**2)
    
    print(f"Prediction differences - Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
    
    return tflite_model, model, {
        'tf_train_acc': train_acc,
        'tf_test_acc': test_acc,
        'tflite_train_acc': tflite_train_acc,
        'tflite_test_acc': tflite_test_acc,
        'train_mse': train_mse,
        'test_mse': test_mse
    }

# 필요한 함수들 import 후 (load_2D_dataset 등)
models = main()

# 또는 개별 모델만 훈련하려면:
train_X, train_Y, test_X, test_Y = load_2D_dataset()

# 기본 모델
model, history = train_tensorflow_model(train_X, train_Y)
tflite_model = convert_to_tflite(model, quantize=True)
save_tflite_model(tflite_model, "my_model.tflite")

# 테스트
predictions = test_tflite_model(tflite_model, test_X)