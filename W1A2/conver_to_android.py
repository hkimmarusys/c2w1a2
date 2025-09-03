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


    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

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
    W3 = parameters["W3"]

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



# ===================================================

class OriginalModelConverter:
    """Convert original assignment NumPy model to TensorFlow Lite"""
    
    def __init__(self, numpy_parameters):
        self.numpy_parameters = numpy_parameters
        self.tf_model = None
        print("✅ Original model converter initialized")
    
    def create_tensorflow_model(self):
        """Create TensorFlow model with same structure as original"""
        print("📌 Creating TensorFlow model (original structure)...")
        
        # 구조: (input=2 → hidden1=20 → hidden2=3 → output=1)
        # model = Sequential([
        #     Dense(20, input_shape=(2,), activation='relu', name='hidden_layer1'),  
        #     Dense(3, activation='relu', name='hidden_layer2'),                     
        #     Dense(1, activation='sigmoid', name='output_layer')                    
        # ])

        model = keras.Sequential([
            layers.Input(shape=(2,), name='input_layer'),
            layers.Dense(20, activation='relu', name = 'hidden_layer1'),
            layers.Dense(3, activation='relu', name='hidden_layer2'),
            layers.Dense(1, activation='sigmoid', name='output_layer')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.tf_model = model
        print("✅ TensorFlow model creation completed")
        return model
    
    def transfer_weights(self):
        """Transfer original NumPy weights to TensorFlow model"""
        print("📌 Transferring original weights...")
        
        if self.tf_model is None:
            raise ValueError("❌ Please create TensorFlow model first!")
        
        # Original NumPy parameters (assignment format)
        W1 = self.numpy_parameters['W1']  # (20, 2)
        b1 = self.numpy_parameters['b1']  # (20, 1)
        W2 = self.numpy_parameters['W2']  # (3, 20)
        b2 = self.numpy_parameters['b2']  # (3, 1)
        W3 = self.numpy_parameters['W3']  # (1, 3)
        b3 = self.numpy_parameters['b3']  # (1, 1)
        
        # Convert to TensorFlow format (Dense expects (in_dim, out_dim))
        tf_W1 = W1.T        # (2, 20)
        tf_b1 = b1.flatten()  # (20,)
        tf_W2 = W2.T        # (20, 3)
        tf_b2 = b2.flatten()  # (3,)
        tf_W3 = W3.T        # (3, 1)
        tf_b3 = b3.flatten()  # (1,)
        
        # Set weights
        self.tf_model.get_layer('hidden_layer1').set_weights([tf_W1, tf_b1])
        self.tf_model.get_layer('hidden_layer2').set_weights([tf_W2, tf_b2])
        self.tf_model.get_layer('output_layer').set_weights([tf_W3, tf_b3])
        
        print("✅ Original weights transfer completed")
    
    def convert_to_tflite(self, filename="converted_model.tflite"):
        """Convert the TensorFlow model to TensorFlow Lite format"""
        if self.tf_model is None:
            raise ValueError("❌ Please create TensorFlow model first!")
        
        print("📌 Converting TensorFlow model to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.tf_model)
        tflite_model = converter.convert()
        
        with open(filename, "wb") as f:
            f.write(tflite_model)
        
        print(f"✅ Conversion completed: {filename}")
        return tflite_model, filename   # ✅ 추가

        

if __name__ == "__main__":
    
    
    
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    input_dim = train_X.shape[0]
    

    numpy_parameters = {
        "W1": np.random.randn(20, input_dim),
        "b1": np.random.randn(20, 1),
        "W2": np.random.randn(3, 20),
        "b2": np.random.randn(3, 1),
        "W3": np.random.randn(1, 3),
        "b3": np.random.randn(1, 1)
    }

    # 모델 변환기 생성
    converter = OriginalModelConverter(numpy_parameters)

    # TensorFlow 모델 생성
    converter.create_tensorflow_model()

    # NumPy 가중치 → TensorFlow 모델로 전송
    converter.transfer_weights()

    # TFLite 변환 및 파일 저장
    tflite_model, tflite_file = converter.convert_to_tflite('original_planar_classifier.tflite')

    if tflite_model is not None:
        print(f"TFLite 모델이 성공적으로 생성되어 {tflite_file}로 저장되었습니다.")

        
