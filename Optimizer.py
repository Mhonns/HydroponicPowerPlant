"""Python code for optimizing the lettuce growth rate using Neural network
the neural structure will look like:

[]  - each node
0   - output layer
w   - weight
hn  - hidden layer at n

h1         h2                       hn      o layer
[[w, w, w] [[[w, w, w, w, w, w, w]   ...    [[w, w, w, w, w, w, w]
 [w, w, w]   [w, w, w, w, w, w, w]   ...     [w, w, w, w, w, w, w]
 [w, w, w]   [w, w, w, w, w, w, w]   ...     [w, w, w, w, w, w, w]
 [w, w, w]   [w, w, w, w, w, w, w]   ...     [w, w, w, w, w, w, w]
 [w, w, w]   [w, w, w, w, w, w, w]   ...     [w, w, w, w, w, w, w]
 [w, w, w]   [w, w, w, w, w, w, w]   ...     [w, w, w, w, w, w, w]
 [w, w, w]]  [w, w, w, w, w, w, w]]  ...]    [w, w, w, w, w, w, w]
                                             [w, w, w, w, w, w, w]
                                             [w, w, w, w, w, w, w]
                                             [w, w, w, w, w, w, w]]

                   Created by Nathadon Samairat and PMKFB team 12 Mar 2023
"""

import numpy as np

intputSize = 3          # the number of input nodes  (light, ph, ec)
outputSize = 1         # the number of output nodes (px per day)
hiddenSize = 4          # the number of nodes in hiddenlayers
hiddenLayerSize = 3     # the number of the hidden layers
haveSaved = False       # true to use the weight and bias from weightData.txt
save = True            # admit to save the weight and bias or not

# assign all weights and bias
def initializeNetwork():
    weightTensor = np.array(
        [np.random.rand(hiddenSize, intputSize),
        np.array([np.random.rand(hiddenSize, hiddenSize) for _ in range(hiddenLayerSize - 1)]),
        np.random.rand(hiddenSize, outputSize)],
        dtype=object
    )
    biasTensor = np.array(
        [np.zeros(hiddenSize),
        np.array([np.zeros(hiddenSize) for _ in range(hiddenLayerSize - 1)]),
        np.zeros(outputSize)],
        dtype=object
    )
    return weightTensor, biasTensor

# activation function sigmoid and Relu in this case
# Args
#    x = numpy array of inactive nodes
def sigmoidIng(x):
    return 1 / (1 + np.exp(-x))
def ReLUIng(x):
    return np.maximum(0, x)
def softmax(x):
    return (np.exp(x)/np.exp(x).sum())

# Forward propagation
# Calculate node activation using 
# activation = sum(weight * input) + bias
# Args
#       weightTensor = current weight tensor
#       inputData    = input data from input layer
def forwardProp(weightTensor, bias, inputData):
    hiddenActive = np.zeros((hiddenLayerSize, hiddenSize), dtype=float)
    outputActive = np.zeros((outputSize), dtype=float)
    activeNode = ReLUIng(np.matmul(weightTensor[0], inputData) + bias[0])
    hiddenActive[0] = activeNode
    for i in range(1, hiddenLayerSize):
        activeNode = ReLUIng(np.matmul(weightTensor[1][i - 1], hiddenActive[i - 1]) + bias[1][i - 1])
        hiddenActive[i] = activeNode
    outputActive = (np.matmul(np.transpose(weightTensor[2]), hiddenActive[-1]) + bias[2])
    predict = np.round(ReLUIng(outputActive), 3)
    return predict, hiddenActive

def transposeArray(a):
    temp = np.copy(a)
    temp = np.transpose(np.matrix(a))
    return temp

# derivative of the ReLU function
def dActivation(z):
    temp = np.copy(z)
    temp[temp > 0] = 1
    return temp

# Back propagation
# Our loss function is RMSE (y_hat - y) ^ 2
# Args
#   weightTensor - current weight
#   biasTensor - current bias
#   activeNodes - list contain [hiddenActive, outputActive]
#   predict - guessed value
#   label   - answer value
def backProp(weightTensor, biasTensor, inputData, activeNodes, predict, label):
    learningRate = 0.01
    tempLenght = hiddenLayerSize - 2
    # Last Layer
    dLoss = 2 * (predict - label)
    dCostBydWeight = transposeArray(activeNodes[-1]) * dActivation(label) * dLoss
    dCostBydBias   = 1 * dActivation(label) * dLoss
    weightTensor[2] -= dCostBydWeight * learningRate
    biasTensor[2]   -= dCostBydBias * learningRate
    # Hidden layers
    for i in range(tempLenght, -1, -1):
        # Last Hidden Layer
        if i == tempLenght:
            dLoss = np.sum(weightTensor[2] * dActivation(label) * dLoss, axis=1)
            dCostBydWeight = transposeArray(activeNodes[i]) * dActivation(activeNodes[i + 1])
            dCostBydBias   = 1 * dActivation(label) * dLoss
        # Hidden Layers Else
        else:
            dLoss = np.sum(weightTensor[1][i + 1] * dActivation(activeNodes[i]) * dLoss, axis=1)
            dCostBydWeight = transposeArray(activeNodes[i]) * dActivation(activeNodes[i + 1])
            dCostBydBias   = 1 * dActivation(activeNodes[i + 2]) * dLoss
        # formating the weight tensor
        for j, value in enumerate(dLoss.copy()):
            dCostBydWeight[:,j] = dCostBydWeight[:,j] * value
        weightTensor[1][i] -= dCostBydWeight * learningRate
        biasTensor[1][i] -= dCostBydBias * learningRate
    # First Layers
    dLoss = np.sum(weightTensor[1][0] * dActivation(activeNodes[i]) * dLoss, axis=1)
    dCostBydWeight = transposeArray(inputData) * dActivation(activeNodes[0])
    dCostBydBias   = 1 * dActivation(activeNodes[1]) * dLoss
    for j, value in enumerate(dLoss.copy()):
        dCostBydWeight[:,j] = dCostBydWeight[:,j] * value
    weightTensor[0] -= np.transpose(dCostBydWeight) * learningRate
    biasTensor[0]   -= dCostBydBias * learningRate
    
    return weightTensor, biasTensor

# MAIN SCRIPT
# get saved bias and weight data from saved data
if (haveSaved == False):
    weightTensor, biasTensor = initializeNetwork()
else:
    # TODO get the saved data
    pass

# test
label = 1
inputData = np.array([1, 1, 1])

predict, activeNodes = forwardProp(weightTensor, biasTensor, inputData)
weightTensor, biasTensor = backProp(weightTensor, biasTensor, inputData, activeNodes, predict, label)

if save:
    print(weightTensor)
    pass

print("\nPredict", predict)